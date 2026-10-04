import unittest

import numpy as np

from project.config import EnvConfig
from project.control import goal_raw_action
from project.diagnostic_paths import PATH_NAMES, WaypointController, waypoint_for
from project.sensing_diagnostics import aggregate_sources, run_episode


class TestSensingDiagnostics(unittest.TestCase):
    def config(self, **overrides):
        values = dict(max_steps=3, sensing_mode="ekf", min_uav_distance=1., goal_tolerance=.2,
                      uav_initial=np.array([[0., 0., 10.], [0., 4., 10.]]),
                      uav_goal_positions=np.array([[8., 0., 10.], [8., 4., 10.]]),
                      world_low=np.array([-20., -20., 1.]), world_high=np.array([20., 20., 30.]),
                      target_initial_state=np.array([6., 8., 12., 0., 0., 0.]),
                      target_acceleration=np.zeros(3), antenna_x=2, antenna_y=2,
                      imperfect_csi_beta=0., csi_beta_jitter=0.)
        values.update(overrides)
        return EnvConfig(**values)

    def test_waypoints_are_feasible_and_terminal_goal_stays_in_observation(self):
        cfg = self.config()
        for name in PATH_NAMES[1:]:
            waypoint = waypoint_for(cfg, name)
            self.assertTrue(np.all(waypoint >= cfg.world_low))
            self.assertTrue(np.all(waypoint <= cfg.world_high))
        obs = np.zeros((2, 31), dtype=np.float32)
        obs[:, 6] = 1.
        action, _ = WaypointController(cfg, "lateral_plus").act(obs, np.ones(2, dtype=bool))
        self.assertEqual(action.shape, (2, 3))
        self.assertTrue(np.isfinite(action).all())

    def test_goal_path_exactly_matches_shared_goal_action(self):
        cfg = self.config()
        obs = np.zeros((2, 31), dtype=np.float32)
        obs[:, 3:9] = np.array([[.1, 0., 0., .5, .2, 0.],
                                [0., -.1, 0., .4, -.3, .1]])
        active = np.array([True, False])
        actual, _ = WaypointController(cfg, "goal").act(obs, active)
        np.testing.assert_array_equal(actual, goal_raw_action(obs, active, cfg))

    def test_waypoint_phase_switches_back_to_actual_goal(self):
        cfg = self.config()
        controller = WaypointController(cfg, "altitude_split")
        obs = np.zeros((2, 31), dtype=np.float32)
        obs[:, :3] = controller.waypoint / cfg.position_scale
        obs[:, 6:9] = np.array([[.3, 0., -.1], [.2, .1, .1]])
        actual, _ = controller.act(obs, np.ones(2, dtype=bool))
        self.assertFalse(controller.using_waypoint.any())
        np.testing.assert_array_equal(actual, goal_raw_action(obs, np.ones(2, dtype=bool), cfg))

    def test_short_episode_keeps_prefix_fields_nan_and_records_deliveries(self):
        episode, steps, delivered = run_episode(self.config(), "test", "goal", seed=4)
        self.assertEqual(len(steps), episode["episode_steps"])
        self.assertTrue(np.isnan(episode["rmse_first10"]))
        self.assertTrue(np.isnan(episode["rho_mean_first10"]))
        self.assertTrue(np.isnan(episode["position_nees_mean_first10"]))
        self.assertTrue(np.isnan(episode["communication_fraction_first10"]))
        self.assertTrue(all(np.isfinite(row["position_nees"]) for row in steps))
        self.assertGreater(len(delivered), 0)
        self.assertEqual(episode["bs_updates"], len(steps))

    def test_source_aggregation_is_only_over_delivered_measurements(self):
        rows = aggregate_sources([("nominal", "goal", 0, "range", .5, 1.),
                                  ("nominal", "goal", 0, "range", 2., 1.)])
        self.assertEqual(rows[0]["delivered_samples"], 2)
        self.assertEqual(rows[0]["floor_hit_fraction"], .5)
        self.assertEqual(rows[0]["raw_to_floor_median"], 1.25)


if __name__ == "__main__":
    unittest.main()
