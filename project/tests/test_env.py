import unittest

import numpy as np

from project.config import EnvConfig
from project.env import DualUAVEnv, swept_distance
from project.observations import build_observations


def simple_config(**overrides):
    """构造确定、短小且不会意外触发避碰的直线场景。"""
    values = dict(
        max_steps=4,
        max_uav_speed=5.0,
        max_uav_acceleration=5.0,
        min_uav_distance=1.0,
        goal_tolerance=0.2,
        uav_initial=np.array([[0.0, 0.0, 10.0], [0.0, 3.0, 10.0]]),
        uav_goal_positions=np.array([[5.0, 0.0, 10.0], [15.0, 3.0, 10.0]]),
        world_low=np.array([-20.0, -20.0, 1.0]),
        world_high=np.array([20.0, 20.0, 30.0]),
        target_initial_state=np.array([6.0, 8.0, 12.0, 0.0, 0.0, 0.0]),
        target_acceleration=np.zeros(3),
        antenna_x=2,
        antenna_y=2,
        imperfect_csi_beta=0.0,
        csi_beta_jitter=0.0,
    )
    values.update(overrides)
    return EnvConfig(**values)


class TestDualUAVEnv(unittest.TestCase):
    def test_asynchronous_arrival_freezes_first_uav(self):
        env = DualUAVEnv(simple_config(), seed=1)
        raw = np.array([[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        raw_copy = raw.copy()

        _, _, _, done, truncated, first = env.step(raw)
        np.testing.assert_array_equal(raw, raw_copy)
        np.testing.assert_array_equal(first["active_before"], [True, True])
        np.testing.assert_array_equal(first["newly_arrived"], [True, False])
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(first["reward_parts"]["arrival"], 40.0)
        self.assertGreater(first["energy_step"][0], 0.0)

        frozen_position = env.uav_positions[0].copy()
        frozen_path = env.path_lengths[0]
        frozen_energy = env.energy_proxies[0]
        frozen_steps = env.active_steps[0]
        inactive_placeholder = np.array([[999.0, -999.0, 999.0], [20.0, 0.0, 0.0]])

        _, _, _, done, _, second = env.step(inactive_placeholder)
        self.assertFalse(done)
        np.testing.assert_array_equal(second["active_before"], [False, True])
        np.testing.assert_array_equal(second["newly_arrived"], [False, False])
        np.testing.assert_allclose(env.uav_positions[0], frozen_position)
        self.assertEqual(env.path_lengths[0], frozen_path)
        self.assertEqual(env.energy_proxies[0], frozen_energy)
        self.assertEqual(env.active_steps[0], frozen_steps)
        self.assertEqual(second["reward_parts"]["arrival"], 0.0)
        np.testing.assert_array_equal(second["sensing_source_mask"], [True, False, True])
        np.testing.assert_allclose(second["intended_beams"][:, 0], 0.0)
        np.testing.assert_allclose(second["actual_beams"][:, 0], 0.0)
        self.assertEqual(second["communication_sinrs"][0], 0.0)

        _, _, _, done, _, third = env.step(raw)
        self.assertTrue(done)
        self.assertEqual(third["termination_reason"], "success")
        np.testing.assert_array_equal(third["newly_arrived"], [False, True])

    def test_deadline_success_has_priority_and_timeout_counts_unfinished(self):
        both_goals = np.array([[5.0, 0.0, 10.0], [5.0, 3.0, 10.0]])
        action = np.array([[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        success = DualUAVEnv(simple_config(max_steps=1, uav_goal_positions=both_goals), seed=2)
        _, _, _, done, truncated, info = success.step(action)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "success")
        self.assertEqual(info["reward_parts"]["timeout"], 0.0)

        deadline = DualUAVEnv(simple_config(max_steps=1), seed=2)
        _, _, _, done, truncated, info = deadline.step(action)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "deadline")
        np.testing.assert_array_equal(info["arrived"], [True, False])
        self.assertEqual(info["reward_parts"]["timeout"], -50.0)

    def test_reset_marks_uav_already_at_goal(self):
        initial = np.array([[0.0, 0.0, 10.0], [0.0, 3.0, 10.0]])
        goals = np.array([[0.0, 0.0, 10.0], [15.0, 3.0, 10.0]])
        env = DualUAVEnv(simple_config(uav_initial=initial, uav_goal_positions=goals), seed=3)
        np.testing.assert_array_equal(env.arrived, [True, False])
        np.testing.assert_array_equal(env.active, [False, True])
        np.testing.assert_array_equal(env.arrival_steps, [0, -1])
        self.assertFalse(env.terminated)

    def test_inactive_uav_still_blocks_collision_path(self):
        initial = np.array([[5.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
        goals = np.array([[5.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
        cfg = simple_config(
            uav_initial=initial,
            uav_goal_positions=goals,
            min_uav_distance=2.0,
        )
        env = DualUAVEnv(cfg, seed=4)
        before = env.uav_positions.copy()
        action = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        _, _, _, done, _, info = env.step(action)
        self.assertTrue(info["safety_intervention"])
        self.assertFalse(info["collision"])
        self.assertFalse(done)
        np.testing.assert_allclose(env.uav_positions, before)

    def test_swept_distance_detects_head_on_pass_through(self):
        start = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        end = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(swept_distance(start, end), 0.0)
        self.assertEqual(np.linalg.norm(end[0] - end[1]), 10.0)

    def test_boundary_request_clips_position_without_mutating_action(self):
        initial = np.array([[18.0, 0.0, 10.0], [0.0, 5.0, 10.0]])
        goals = np.array([[0.0, 0.0, 10.0], [0.0, -5.0, 10.0]])
        env = DualUAVEnv(simple_config(uav_initial=initial, uav_goal_positions=goals), seed=5)
        action = np.array([[20.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        original = action.copy()
        _, _, _, _, _, info = env.step(action)
        np.testing.assert_array_equal(action, original)
        np.testing.assert_array_equal(info["boundary_requests"], [True, False])
        self.assertEqual(env.uav_positions[0, 0], env.config.world_high[0])

    def test_observation_shape_time_and_no_target_truth_leak(self):
        env = DualUAVEnv(simple_config(), seed=6)
        obs0, state0, _ = env.reset(seed=6)
        self.assertEqual(obs0.shape, (2, 31))
        self.assertEqual(state0.shape, (58,))
        self.assertEqual(float(obs0[0, 9]), 1.0)

        action = np.zeros((2, 3))
        obs1, state1, _, _, _, _ = env.step(action)
        self.assertEqual(float(obs1[0, 9]), 0.75)
        self.assertEqual(state1.shape, (58,))

        before_obs, before_state = build_observations(env)
        env.target_state += np.array([1000.0, -500.0, 700.0, 9.0, 8.0, 7.0])
        after_obs, after_state = build_observations(env)
        np.testing.assert_array_equal(after_obs, before_obs)
        np.testing.assert_array_equal(after_state, before_state)


if __name__ == "__main__":
    unittest.main()
