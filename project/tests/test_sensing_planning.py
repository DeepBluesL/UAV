import unittest

import numpy as np

from project.annealing import SimulatedAnnealingController
from project.config import EnvConfig
from project.env import DualUAVEnv
from project.mpc import MPCController
from project.nominal_links import nominal_link_metrics
from project.sensing_planning import (BeliefTrajectoryCost,
                                      SensingAwareMPCController,
                                      SensingAwareSAController)


def config(**overrides):
    values = dict(
        max_steps=10, min_uav_distance=2., goal_tolerance=.2,
        uav_initial=np.array([[0., -3., 10.], [0., 3., 10.]]),
        uav_goal_positions=np.array([[14., -3., 10.], [14., 3., 10.]]),
        world_low=np.array([-20., -20., 1.]), world_high=np.array([20., 20., 30.]),
        target_initial_state=np.array([6., 8., 12., 0., 0., 0.]),
        target_acceleration=np.zeros(3), antenna_x=2, antenna_y=2,
        imperfect_csi_beta=0., csi_beta_jitter=0.)
    values.update(overrides)
    return EnvConfig(**values)


class TestSensingPlanning(unittest.TestCase):
    def test_zero_weights_exactly_match_navigation_controllers(self):
        cfg = config()
        env = DualUAVEnv(cfg, seed=4)
        obs, _, info = env.reset(seed=4)
        pairs = [
            (SimulatedAnnealingController(cfg, seed=9),
             SensingAwareSAController(
                 cfg, seed=9, sensing_weight=0., communication_weight=0.)),
            (MPCController(cfg), SensingAwareMPCController(
                cfg, sensing_weight=0., communication_weight=0.)),
        ]
        for navigation, sensing in pairs:
            sensing.set_belief(
                info["estimated_target_state"], info["tracking_covariance"])
            np.testing.assert_array_equal(
                navigation.act(obs, info["active"])[0],
                sensing.act(obs, info["active"])[0])

    def test_both_planners_have_no_target_truth_or_acceleration_oracle(self):
        first_cfg = config(target_acceleration=np.array([9., -8., 7.]))
        second_cfg = config(
            target_acceleration=np.array([-4., 5., -6.]),
            target_initial_state=np.array([-10., 1., 4., 3., 2., 1.]))
        env = DualUAVEnv(first_cfg, seed=7)
        obs, _, info = env.reset(seed=7)
        for controller_type, kwargs in (
                (SensingAwareSAController, {"seed": 12}),
                (SensingAwareMPCController, {})):
            controllers = [controller_type(first_cfg, **kwargs),
                           controller_type(second_cfg, **kwargs)]
            for controller in controllers:
                controller.set_belief(
                    info["estimated_target_state"], info["tracking_covariance"])
            np.testing.assert_array_equal(
                controllers[0].act(obs, info["active"])[0],
                controllers[1].act(obs, info["active"])[0])

    def test_nominal_links_are_finite_and_inactive_source_is_zero(self):
        cfg = config()
        communication, sensing = nominal_link_metrics(
            cfg.uav_initial, np.array([6., 8., 12.]), np.array([True, False]), cfg)
        self.assertTrue(np.isfinite(communication).all())
        self.assertTrue(np.isfinite(sensing).all())
        self.assertEqual(communication[1], 0.)
        self.assertEqual(sensing[2], np.finfo(float).tiny)

    def test_belief_cost_responds_to_covariance(self):
        cfg = config()
        positions = np.broadcast_to(cfg.uav_initial, (1, 1, 2, 3)).copy()
        velocities = np.zeros_like(positions)
        active = np.ones((1, 1, 2), dtype=bool)
        estimate = cfg.target_initial_state.copy()
        low = BeliefTrajectoryCost(cfg, estimate, np.eye(6), .2, 0.)
        high = BeliefTrajectoryCost(cfg, estimate, 100. * np.eye(6), .2, 0.)
        self.assertGreater(high(positions, velocities, active)[0],
                           low(positions, velocities, active)[0])

    def test_cost_stops_after_both_predicted_arrivals(self):
        cfg = config()
        positions = np.broadcast_to(cfg.uav_initial, (1, 3, 2, 3)).copy()
        changed = positions.copy()
        changed[:, 1:] += 1000.
        velocities = np.zeros_like(positions)
        active = np.array([[[True, True], [False, False], [False, False]]])
        cost = BeliefTrajectoryCost(
            cfg, cfg.target_initial_state, np.eye(6), .2, .5, uncertainty_ref=1.)
        np.testing.assert_allclose(
            cost(positions, velocities, active), cost(changed, velocities, active))

    def test_measurement_update_respects_upload_and_source_toggles(self):
        mean, covariance = config().target_initial_state, 10. * np.eye(6)
        positions, velocities = config().uav_initial, np.zeros((2, 3))
        active, sensing = np.ones(2, dtype=bool), np.ones(3)

        enabled = BeliefTrajectoryCost(config(), mean, covariance, .2, .5)
        below = enabled._measurement_update(
            mean, covariance, positions, velocities, active, np.zeros(2), sensing)
        above = enabled._measurement_update(
            mean, covariance, positions, velocities, active,
            np.full(2, enabled.config.gamma_min * 2), sensing)
        self.assertLess(np.trace(above), np.trace(below))

        bs_only = BeliefTrajectoryCost(
            config(tracking_use_uav_measurements=False), mean, covariance, .2, .5)
        np.testing.assert_allclose(
            bs_only._measurement_update(
                mean, covariance, positions, velocities, active,
                np.full(2, bs_only.config.gamma_min * 2), sensing),
            bs_only._measurement_update(
                mean, covariance, positions, velocities, active, np.zeros(2), sensing))

        disabled = BeliefTrajectoryCost(
            config(collect_measurements=False), mean, covariance, .2, .5)
        np.testing.assert_array_equal(
            disabled._measurement_update(
                mean, covariance, positions, velocities, active,
                np.full(2, disabled.config.gamma_min * 2), sensing), covariance)

    def test_mpc_trace_keeps_arrival_step_displacement_velocity(self):
        class CaptureMPC(MPCController):
            def _trajectory_cost_enabled(self):
                return True

            def _trajectory_cost(self, positions, velocities, active):
                self.trace = positions, velocities, active
                return np.zeros(len(positions))

        cfg = config(
            uav_goal_positions=np.array([[5., -3., 10.], [14., 3., 10.]]),
            goal_tolerance=.2)
        env = DualUAVEnv(cfg, seed=2)
        obs, _, info = env.reset(seed=2)
        controller = CaptureMPC(cfg, horizon=2)
        controller.act(obs, info["active"])
        positions, velocities, active = controller.trace
        arrived_first = (active[:, 0, 0]
                         & (np.linalg.norm(positions[:, 0, 0] - cfg.uav_goal_positions[0],
                                           axis=1) <= cfg.goal_tolerance))
        self.assertTrue(np.any(arrived_first))
        self.assertTrue(np.all(np.linalg.norm(velocities[arrived_first, 0, 0], axis=1) > 0.))

    def test_uncertainty_reference_is_explicit_metadata(self):
        controller = SensingAwareSAController(config(), uncertainty_ref=2.5)
        self.assertEqual(controller.settings["uncertainty_ref"], 2.5)
        self.assertIn("horizon_mean", controller.settings["trajectory_cost_units"])


if __name__ == "__main__":
    unittest.main()
