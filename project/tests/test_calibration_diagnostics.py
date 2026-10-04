import unittest

import numpy as np

from project.calibration_diagnostics import run_episode, summarize
from project.config import EnvConfig
from project.tracking import TrackingEKF


class TestCalibrationDiagnostics(unittest.TestCase):
    def test_storing_diagnostics_does_not_change_filter_state(self):
        cfg = EnvConfig(sensing_mode="ekf")
        truth = cfg.target_initial_state.copy()
        positions = cfg.uav_initial.copy()
        velocities = np.zeros((2, 3))
        crbs = np.ones((3, 4))
        mask = np.array([True, True, False])
        noise = np.arange(12, dtype=float).reshape(3, 4) / 10.
        first = TrackingEKF(cfg, np.random.default_rng(2))
        second = TrackingEKF(cfg, np.random.default_rng(2))
        for tracker in (first, second):
            tracker.initialize(truth); tracker.predict()
        expected = first.update(truth, positions, velocities, crbs, mask, noise)
        actual = second.update(truth, positions, velocities, crbs, mask, noise)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(actual[0], [61.92709570444356, 6.616312428097613,
                                              58.35186703809467, -1.745140753383597,
                                              2.467661463909002, .05171518799439911], rtol=1e-13)
        np.testing.assert_allclose(np.diag(actual[1]),
                                   [4.805254768364986, 71.43762271194522,
                                    4.661789819867988, .395798492946867,
                                    1.073359208247552, .582654342810588], rtol=1e-13)
        self.assertTrue(np.isfinite(second.last_residuals[:2]).all())
        self.assertTrue(np.isnan(second.last_residuals[2]).all())

    def test_short_goal_probe_has_position_and_delivered_source_statistics(self):
        cfg = EnvConfig(sensing_mode="ekf", max_steps=2)
        rows = run_episode(cfg, "nominal", 4101)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(np.isfinite(row["position_nees"]) for row in rows))
        summary = summarize(rows)
        position = next(row for row in summary
                        if row["window"] == "steps_1_3" and row["source"] == "position")
        self.assertEqual(position["samples"], 2)


if __name__ == "__main__":
    unittest.main()
