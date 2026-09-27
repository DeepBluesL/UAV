import unittest

import numpy as np

from uav_isac_basic import (
    BasicUAVISACEnv,
    spherical_measurement,
)


class BasicUAVISACTests(unittest.TestCase):
    def test_spherical_measurement(self):
        state = np.array([3.0, 4.0, 12.0, 0.3, 0.4, 1.2])
        measurement = spherical_measurement(state)

        self.assertAlmostEqual(measurement[0], 13.0)
        self.assertAlmostEqual(measurement[1], np.arccos(12.0 / 13.0))
        self.assertAlmostEqual(measurement[2], np.arctan2(4.0, 3.0))
        self.assertAlmostEqual(measurement[3], 16.9 / 13.0)

    def test_uav_delta_is_clipped_to_speed_constraint(self):
        env = BasicUAVISACEnv(seed=1)
        start = env.uav_position.copy()
        _, info = env.step([100.0, 0.0, 0.0])

        moved = np.linalg.norm(info.uav_position - start)
        self.assertAlmostEqual(moved, env.config.max_uav_step)
        self.assertTrue(info.constraints["uav_speed"])

    def test_beam_power_is_normalized(self):
        env = BasicUAVISACEnv(seed=1)
        links = env.compute_link_metrics(env.uav_position, env.true_target_state, 0.25)

        self.assertAlmostEqual(
            links.comm_power + links.sensing_power,
            env.config.total_power,
        )

    def test_more_sensing_power_improves_sensing_sinr(self):
        env = BasicUAVISACEnv(seed=1)
        low = env.compute_link_metrics(env.uav_position, env.true_target_state, 0.20)
        high = env.compute_link_metrics(env.uav_position, env.true_target_state, 0.80)

        self.assertGreater(high.sinr_sensing_bs, low.sinr_sensing_bs)
        self.assertLess(high.sinr_comm, low.sinr_comm)

    def test_step_outputs_finite_pcrb_trace(self):
        env = BasicUAVISACEnv(seed=1)
        _, info = env.step([1.0, 0.0, 0.0], sensing_power_fraction=0.5)

        self.assertTrue(np.isfinite(info.rho))
        self.assertGreater(info.rho, 0.0)
        self.assertTrue(info.constraints["bs_total_power"])


if __name__ == "__main__":
    unittest.main()
