import unittest

import numpy as np

from project.config import EnvConfig
from project.crb import crb_from_sensing_sinr
from project.env import DualUAVEnv
from project.tracking import (TrackingEKF, geometric_measurement,
                              measurement_covariance, measurement_jacobian, wrap_angle)


def ekf_config(**overrides):
    values = dict(
        sensing_mode="ekf", max_steps=4, antenna_x=2, antenna_y=2,
        min_uav_distance=1.0, goal_tolerance=.2,
        uav_initial=np.array([[0., -5., 10.], [0., 5., 10.]]),
        uav_goal_positions=np.array([[15., -5., 10.], [15., 5., 10.]]),
        world_low=np.array([-20., -20., 1.]), world_high=np.array([30., 20., 30.]),
        target_initial_state=np.array([8., 2., 12., .5, -.2, .1]),
        target_acceleration=np.zeros(3), imperfect_csi_beta=0., csi_beta_jitter=0.,
        tracking_initial_position_std=4., tracking_initial_velocity_std=.8,
        measurement_range_std_floor=.5,
        measurement_angle_std_floor=np.deg2rad(.2),
        measurement_range_rate_std_floor=.2,
    )
    values.update(overrides)
    return EnvConfig(**values)


class TrackingGeometryTests(unittest.TestCase):
    def test_crb_polar_and_azimuth_variances_are_reordered(self):
        cfg = ekf_config(measurement_range_std_floor=1e-9,
                         measurement_angle_std_floor=1e-9,
                         measurement_range_rate_std_floor=1e-9)
        crb = crb_from_sensing_sinr(
            2., bandwidth=1., kappa_d=4., kappa_theta=8., kappa_phi=12., kappa_v=16.)
        covariance = measurement_covariance(crb, cfg)
        # tracking顺序是range, azimuth(phi), elevation(theta), range-rate。
        np.testing.assert_allclose(np.diag(covariance), crb[[0, 2, 1, 3]])

    def test_single_and_bistatic_measurements_use_sensor_geometry(self):
        state = np.array([3., 4., 0., 1., 0., 0.])
        bs = np.zeros(3)
        mono = geometric_measurement(state, bs, bs, np.zeros(3), False)
        self.assertAlmostEqual(mono[0], 10.)
        self.assertAlmostEqual(mono[3], 1.2)
        sensor = np.array([3., 0., 0.])
        bistatic = geometric_measurement(state, bs, sensor, np.zeros(3), True)
        self.assertAlmostEqual(bistatic[0], 9.)
        self.assertAlmostEqual(bistatic[3], .6)

    def test_analytic_jacobian_matches_finite_difference(self):
        state = np.array([8., 3., 12., .5, -.2, .1])
        bs, sensor = np.zeros(3), np.array([1., -4., 8.])
        velocity = np.array([.1, .2, 0.])
        analytic = measurement_jacobian(state, bs, sensor, velocity, True)
        numeric = np.zeros((4, 6))
        for index in range(6):
            plus, minus = state.copy(), state.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            difference = (geometric_measurement(plus, bs, sensor, velocity, True)
                          - geometric_measurement(minus, bs, sensor, velocity, True))
            difference[1:3] = [wrap_angle(value) for value in difference[1:3]]
            numeric[:, index] = difference / 2e-5
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


class TrackingLoopTests(unittest.TestCase):
    def test_prediction_only_never_reinjects_truth(self):
        env = DualUAVEnv(ekf_config(collect_measurements=False), seed=21)
        initial_error = env.estimated_target_state - env.target_state
        self.assertGreater(np.linalg.norm(initial_error), 0.)
        _, _, _, _, _, info = env.step(np.zeros((2, 3)))
        np.testing.assert_array_equal(info["estimated_target_state"],
                                      info["prior_estimated_target_state"])
        self.assertEqual(info["measurement_count"], 0)
        np.testing.assert_array_equal(info["measurement_counts"], [0, 0, 0])
        self.assertGreater(info["tracking_position_error"], 0.)
        self.assertEqual(info["uncertainty_kind"], "ekf_covariance")

    def test_bs_only_when_uav_upload_is_disabled_or_communication_fails(self):
        for cfg in (ekf_config(tracking_use_uav_measurements=False),
                    ekf_config(gamma_min=1e100)):
            with self.subTest(gamma_min=cfg.gamma_min,
                              use_uav=cfg.tracking_use_uav_measurements):
                env = DualUAVEnv(cfg, seed=22)
                _, _, _, _, _, info = env.step(np.zeros((2, 3)))
                np.testing.assert_array_equal(info["measurement_source_mask"], [True, False, False])
                np.testing.assert_array_equal(info["measurement_counts"], [1, 0, 0])

    def test_measurements_reduce_covariance_and_are_reproducible(self):
        measured = DualUAVEnv(ekf_config(), seed=23)
        predicted = DualUAVEnv(ekf_config(collect_measurements=False), seed=23)
        action = np.zeros((2, 3))
        _, _, _, _, _, measured_info = measured.step(action)
        _, _, _, _, _, predicted_info = predicted.step(action)
        self.assertLess(np.trace(measured_info["tracking_covariance"]),
                        np.trace(predicted_info["tracking_covariance"]))
        self.assertEqual(measured_info["measurement_count"], 3)
        replay = DualUAVEnv(ekf_config(), seed=23)
        _, _, _, _, _, replay_info = replay.step(action)
        np.testing.assert_allclose(measured_info["estimated_target_state"],
                                   replay_info["estimated_target_state"])
        np.testing.assert_allclose(measured_info["tracking_covariance"],
                                   replay_info["tracking_covariance"])

    def test_joseph_update_keeps_covariance_symmetric_positive(self):
        cfg = ekf_config()
        tracker = TrackingEKF(cfg, np.random.default_rng(5))
        tracker.initialize(cfg.target_initial_state)
        tracker.predict()
        positions = cfg.uav_initial.copy()
        crbs = np.full((3, 4), 1e-4)
        _, covariance = tracker.update(
            cfg.target_initial_state, positions, np.zeros((2, 3)), crbs,
            np.ones(3, dtype=bool), np.zeros((3, 4)))
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), -1e-10)


if __name__ == "__main__":
    unittest.main()
