import unittest

import numpy as np

from project.config import EnvConfig
from project.window_metrics import window_metrics


class WindowMetricsTests(unittest.TestCase):
    def test_arrival_step_and_early_late_errors_are_accounted_for(self):
        trace = {
            "rho_pos": np.ones(6), "source_mask": np.array([[1, 1, 1]] * 2 + [[1, 0, 1]] * 3),
            "communication_sinrs": np.array([[4, 2], [2, 4], [0, 4], [0, 4], [0, 4]]),
            "estimated_target_positions": np.array([[0., 0., 0.]] + [[2., 0., 0.]] * 3 + [[1., 0., 0.]] * 2),
            "target_positions": np.zeros((6, 3)), "uncertainty_kind": "ekf_covariance",
            "tracking_covariances": np.broadcast_to(np.eye(6), (6, 6, 6)),
        }
        result = window_metrics(trace, EnvConfig(), 5)
        self.assertAlmostEqual(result["communication_prefix_rate"], 5 / 7)
        self.assertEqual(result["tracking_prefix_early_rmse"], 2.)
        self.assertEqual(result["tracking_prefix_late_rmse"], 1.)
        self.assertAlmostEqual(result["position_nees_prefix_mean"], 14 / 5)
        self.assertEqual(result["position_coverage_prefix_rate"], 1.)
        self.assertTrue(all(value is None for value in window_metrics(trace, EnvConfig(), 6).values()))


if __name__ == "__main__":
    unittest.main()
