import unittest

import numpy as np

from project.annealing import SimulatedAnnealingController
from project.config import EnvConfig
from project.env import DualUAVEnv
from project.planning import executed_acceleration, physical_to_raw, rollout_cost


def config(**overrides):
    values = dict(
        max_steps=20, min_uav_distance=2., goal_tolerance=.3,
        uav_initial=np.array([[0., -3., 10.], [0., 3., 10.]]),
        uav_goal_positions=np.array([[14., 3., 10.], [14., -3., 10.]]),
        world_low=np.array([-20., -20., 1.]), world_high=np.array([20., 20., 30.]),
        target_initial_state=np.array([6., 8., 12., 0., 0., 0.]),
        target_acceleration=np.zeros(3), antenna_x=2, antenna_y=2,
        imperfect_csi_beta=0., csi_beta_jitter=0.)
    values.update(overrides)
    return EnvConfig(**values)


class TestAnnealing(unittest.TestCase):
    def test_reset_reproduces_search_and_inactive_is_zero(self):
        cfg = config(uav_goal_positions=np.array([[0., -3., 10.], [14., -3., 10.]]))
        env = DualUAVEnv(cfg, seed=2)
        obs, _, info = env.reset(seed=2)
        policy = SimulatedAnnealingController(cfg, seed=11)
        first = [policy.act(obs, info["active"])[0] for _ in range(2)]
        policy.reset(11)
        second = [policy.act(obs, info["active"])[0] for _ in range(2)]
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0][0], 0.)
        self.assertTrue(np.isfinite(first[0]).all())

    def test_planner_action_mapping_matches_environment(self):
        cfg = config(uav_initial=np.array([[18., -4., 10.], [0., 4., 10.]]),
                     uav_goal_positions=np.array([[0., -4., 10.], [14., 4., 10.]]),
                     uav_initial_velocities=np.array([[3., 0., 0.], [1., -1., 0.]]))
        env = DualUAVEnv(cfg, seed=3)
        physical = np.array([[5., 0., 0.], [0., -5., 0.]])
        raw = physical_to_raw(physical, env.active, cfg)
        expected_acc = executed_acceleration(physical, cfg)
        actual_acc, _, boundary, _, _ = env._move(raw)
        np.testing.assert_allclose(actual_acc, expected_acc, atol=1e-7)
        np.testing.assert_array_equal(boundary, [True, False])

    def test_collision_sequence_has_large_cost(self):
        cfg = config(min_uav_distance=3.)
        env = DualUAVEnv(cfg, seed=4)
        obs, _, info = env.reset(seed=4)
        position = info["positions"]
        velocity = info["velocities"]
        goals = cfg.uav_goal_positions
        toward = np.zeros((1, 2, 2, 3))
        toward[:, :, 0, 1] = 5.
        toward[:, :, 1, 1] = -5.
        safe = np.zeros_like(toward)
        weights = dict(navigation=1., effort=.03, clearance=.12, boundary=.2, collision=1.e4)
        unsafe_cost = rollout_cost(toward, position, velocity, goals, info["active"], cfg, weights)
        safe_cost = rollout_cost(safe, position, velocity, goals, info["active"], cfg, weights)
        self.assertGreater(unsafe_cost[0], safe_cost[0] + 1000.)

    def test_crossing_first_action_avoids_environment_intervention(self):
        cfg = config(min_uav_distance=3.)
        env = DualUAVEnv(cfg, seed=5)
        obs, _, info = env.reset(seed=5)
        policy = SimulatedAnnealingController(cfg, seed=5)
        action, logp = policy.act(obs, info["active"])
        _, _, _, _, _, result = env.step(action)
        self.assertFalse(result["safety_intervention"])
        np.testing.assert_array_equal(logp, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
