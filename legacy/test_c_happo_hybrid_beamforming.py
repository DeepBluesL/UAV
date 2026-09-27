import unittest

import numpy as np

from c_happo_hybrid_beamforming import (
    AGENT_BS,
    AGENT_NAMES,
    CurriculumScheduler,
    HybridBeamformingConfig,
    HybridCsiDualUAVEnv,
)


class CHAPPOHybridBeamformingTests(unittest.TestCase):
    def test_reset_returns_heterogeneous_observations(self):
        env = HybridCsiDualUAVEnv(HybridBeamformingConfig(max_steps=2), seed=1)

        observations, global_state = env.reset()

        self.assertEqual(set(observations), set(AGENT_NAMES))
        self.assertEqual(global_state.shape, (env.config.global_state_dim,))
        self.assertEqual(
            observations[AGENT_BS].shape, (env.config.bs_observation_dim,)
        )
        self.assertEqual(
            observations["uav_0"].shape, (env.config.uav_observation_dim,)
        )

    def test_zero_action_uses_matched_beam_fallback_and_returns_finite_metrics(self):
        config = HybridBeamformingConfig(max_steps=2)
        env = HybridCsiDualUAVEnv(config, seed=2)
        actions = env.zero_actions()

        _, _, reward, done, info = env.step(actions)

        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(np.isfinite(info.rho))
        self.assertEqual(info.communication_sinrs.shape, (2,))
        self.assertEqual(info.sensing_sinrs.shape, (3,))
        self.assertAlmostEqual(info.power, config.total_power, places=6)
        self.assertEqual(info.goal_distances.shape, (2,))
        self.assertIn("goal_distance", info.reward_parts)
        self.assertIn("hbf_spectral_efficiency", info.reward_parts)
        self.assertLessEqual(
            np.max(np.linalg.norm(info.uav_velocities, axis=1)),
            config.max_uav_speed + 1.0e-9,
        )

    def test_random_bs_action_is_projected_to_power_budget(self):
        config = HybridBeamformingConfig(max_steps=1)
        env = HybridCsiDualUAVEnv(config, seed=3)
        rng = np.random.default_rng(3)

        beam_matrix = env.decode_bs_action(
            rng.normal(size=config.bs_action_dim)
        )

        self.assertEqual(beam_matrix.shape, (config.num_antennas, config.num_beams))
        power = float(np.real(np.sum(np.abs(beam_matrix) ** 2)))
        self.assertAlmostEqual(power, config.total_power, places=6)

    def test_curriculum_scheduler_advances_by_interval(self):
        config = HybridBeamformingConfig(
            curriculum_stages=3,
            curriculum_advance_every=2,
        )
        scheduler = CurriculumScheduler(config)

        self.assertEqual(scheduler.stage.index, 0)
        self.assertFalse(scheduler.maybe_advance(1))
        self.assertTrue(scheduler.maybe_advance(2))
        self.assertEqual(scheduler.stage.index, 1)


if __name__ == "__main__":
    unittest.main()
