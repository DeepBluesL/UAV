import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from project.artifacts import load_policy, save_policy
from project.baselines import observation_state
from project.config import EnvConfig, PPOConfig, RewardConfig
from project.core import MAPPOActorCritic
from project.env import DualUAVEnv
from project.observations import build_observations, decode_belief


class ObservationVersionTests(unittest.TestCase):
    def test_v1_keeps_legacy_dimensions_and_reward_scaled_uncertainty(self):
        cfg = EnvConfig()
        low, _, _ = DualUAVEnv(cfg, RewardConfig(rho_ref=.01), seed=4).reset(seed=4)
        high, state, _ = DualUAVEnv(cfg, RewardConfig(rho_ref=10.), seed=4).reset(seed=4)
        self.assertEqual(low.shape, (2, 31))
        self.assertEqual(state.shape, (58,))
        self.assertFalse(np.array_equal(low[:, 16:19], high[:, 16:19]))

    def test_v2_prefix_navigation_fields_match_v1_and_uncertainty_is_decoupled(self):
        base = EnvConfig()
        v1, _, _ = DualUAVEnv(base, RewardConfig(), seed=5).reset(seed=5)
        cfg = replace(base, observation_version="v2")
        low, state, _ = DualUAVEnv(cfg, RewardConfig(rho_ref=.01), seed=5).reset(seed=5)
        high, _, _ = DualUAVEnv(cfg, RewardConfig(rho_ref=10.), seed=5).reset(seed=5)
        self.assertEqual(low.shape, (2, 67))
        self.assertEqual(state.shape, (72,))
        np.testing.assert_array_equal(low[:, :16], v1[:, :16])
        np.testing.assert_array_equal(low[:, 19:31], v1[:, 19:31])
        np.testing.assert_array_equal(low[:, 16:19], high[:, 16:19])
        for actual, expected in zip(observation_state(low, cfg), observation_state(v1, base)):
            np.testing.assert_array_equal(actual, expected)

    def test_v2_decodes_complete_public_belief_without_target_truth(self):
        cfg = EnvConfig(observation_version="v2")
        env = DualUAVEnv(cfg, seed=6)
        observation, _, _ = env.reset(seed=6)
        decoded = decode_belief(observation[0], cfg)
        np.testing.assert_allclose(decoded, env.pcrb_matrix, rtol=2e-6, atol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(decoded).min(), -1e-6)
        before = observation.copy()
        env.target_state += 1000.0
        after, _ = build_observations(env)
        np.testing.assert_array_equal(after, before)

    def test_v2_checkpoint_roundtrip_and_dimension_validation(self):
        cfg, reward = EnvConfig(observation_version="v2"), RewardConfig()
        ppo = PPOConfig(hidden_sizes=(8,))
        model = MAPPOActorCritic(67, 72, hidden_sizes=(8,))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "v2.pt"
            save_policy(path, model, cfg, reward, ppo)
            loaded, loaded_cfg, _, _ = load_policy(path)
            self.assertEqual(loaded.pi[0].mu_net[0].in_features, 67)
            self.assertEqual(loaded_cfg.observation_version, "v2")
            with self.assertRaises(ValueError):
                save_policy(path, MAPPOActorCritic(31, 58), cfg, reward, ppo)


if __name__ == "__main__":
    unittest.main()
