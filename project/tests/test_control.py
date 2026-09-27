import unittest

import numpy as np
import torch

from project.config import EnvConfig
from project.control import ActionAdapter
from project.core import MAPPOActorCritic
from project.env import DualUAVEnv
from project.evaluate import GoalController
from project.observations import build_observations


class ActionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.obs = np.zeros((2, 31), dtype=np.float32)
        self.obs[:, 6:9] = np.array([[1., 0., 0.], [0., 1., 0.]])

    def test_pure_mode_preserves_raw_gaussian_action(self):
        raw = np.array([[2., -3., .5], [1., 2., 3.]], dtype=np.float32)
        actual = ActionAdapter(self.cfg, "pure").policy_action(raw, self.obs, [1, 1])
        np.testing.assert_array_equal(actual, raw)

    def test_zero_residual_reproduces_goal_acceleration_after_env_tanh(self):
        action = ActionAdapter(self.cfg, "residual", .25).policy_action(
            np.zeros((2, 3)), self.obs, [1, 1])
        acceleration = np.tanh(action) * self.cfg.max_uav_acceleration
        np.testing.assert_allclose(acceleration, [[4.995, 0., 0.], [0., 4.995, 0.]], atol=1e-5)

    def test_residual_actor_starts_with_zero_mean_but_keeps_gaussian_logprob(self):
        torch.manual_seed(1)
        ac = MAPPOActorCritic(31, 58, hidden_sizes=(8,), environment=self.cfg,
                              control_mode="residual")
        raw, logp = ac.step(self.obs, [True, True], deterministic=True)
        np.testing.assert_array_equal(raw, np.zeros((2, 3), dtype=np.float32))
        self.assertTrue(np.isfinite(logp).all())
        env_action, _ = ac.act(self.obs, [True, True], deterministic=True)
        self.assertFalse(np.allclose(env_action, raw))

    def test_environment_update_changes_action_scaling(self):
        adapter = ActionAdapter(self.cfg, "residual", 0.)
        before = adapter.policy_action(np.zeros((2, 3)), self.obs, [1, 1])
        changed = EnvConfig(max_uav_speed=2.5, max_uav_acceleration=2.5, slot_duration=2.)
        adapter.set_environment(changed)
        after = adapter.policy_action(np.zeros((2, 3)), self.obs, [1, 1])
        self.assertFalse(np.allclose(before, after))

    def test_zero_residual_matches_goal_motion_and_energy_with_reverse_velocity(self):
        cfg = EnvConfig(uav_initial_velocities=np.array([[-5., 0., 0.], [0., 5., 0.]]))
        goal_env, residual_env = DualUAVEnv(cfg), DualUAVEnv(cfg)
        goal_obs, _, _ = goal_env.reset(seed=12)
        residual_obs, _, _ = residual_env.reset(seed=12)
        goal_action, _ = GoalController(cfg).act(goal_obs, goal_env.active)
        residual_action = ActionAdapter(cfg, "residual", .25).policy_action(
            np.zeros((2, 3)), residual_obs, residual_env.active)
        *_, goal_info = goal_env.step(goal_action)
        *_, residual_info = residual_env.step(residual_action)
        np.testing.assert_allclose(goal_info["positions"], residual_info["positions"], atol=1e-7)
        np.testing.assert_allclose(goal_info["energy_step"], residual_info["energy_step"], atol=1e-7)

    def test_zero_residual_keeps_inactive_action_zero(self):
        cfg = EnvConfig(uav_initial=np.array([[150., 70., 90.], [30., -10., 70.]]))
        env = DualUAVEnv(cfg)
        obs, _ = build_observations(env)
        action = ActionAdapter(cfg, "residual", .25).policy_action(
            np.ones((2, 3)), obs, env.active)
        np.testing.assert_array_equal(action[0], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
