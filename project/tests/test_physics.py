import unittest

import numpy as np

from project.config import EnvConfig
from project.physics import ISACPhysics


POSITIONS = np.array([[30.0, 10.0, 70.0], [30.0, -10.0, 70.0]])
TARGET = np.array([60.0, 0.0, 60.0, -1.0, 1.0, -1.0])


class TestISACPhysics(unittest.TestCase):
    def test_link_metrics_respect_active_sources(self):
        cases = [
            ([True, True], [True, True, True], []),
            ([True, False], [True, True, False], [1]),
            ([False, False], [True, False, False], [0, 1]),
        ]
        for active, expected_mask, zero_columns in cases:
            with self.subTest(active=active):
                physics = ISACPhysics(EnvConfig(), np.random.default_rng(12))
                metrics = physics.link_metrics(
                    POSITIONS, TARGET, TARGET, np.array(active)
                )

                self.assertEqual(metrics["intended_beams"].shape, (64, 3))
                self.assertEqual(metrics["actual_beams"].shape, (64, 3))
                self.assertEqual(
                    metrics["sensing_source_mask"].tolist(), expected_mask
                )
                self.assertAlmostEqual(metrics["power"], 5.0)
                valid = np.array(expected_mask)[[1, 2, 0]]
                column_powers = np.sum(
                    np.abs(metrics["actual_beams"]) ** 2, axis=0
                )
                np.testing.assert_allclose(
                    column_powers[valid], 5.0 / np.count_nonzero(valid)
                )
                for idx in zero_columns:
                    self.assertEqual(
                        np.count_nonzero(metrics["intended_beams"][:, idx]), 0
                    )
                    self.assertEqual(
                        np.count_nonzero(metrics["actual_beams"][:, idx]), 0
                    )
                    self.assertEqual(metrics["communication_sinrs"][idx], 0.0)
                    self.assertEqual(metrics["sensing_sinrs"][idx + 1], 0.0)

    def test_only_bs_builds_one_crb_and_finite_pcrb_update(self):
        physics = ISACPhysics(EnvConfig(), np.random.default_rng(7))
        result = physics.update(
            POSITIONS,
            TARGET,
            TARGET,
            np.array([False, False]),
            TARGET - np.array([-1.0, 1.0, -1.0, 0.0, 0.0, 0.0]),
            np.eye(6),
        )

        self.assertTrue(np.all(result["source_crbs"][0] > 0.0))
        self.assertEqual(np.count_nonzero(result["source_crbs"][1:]), 0)
        np.testing.assert_allclose(result["fused_crb"], result["source_crbs"][0])
        self.assertAlmostEqual(
            result["rho_pos"], np.trace(result["pcrb"][:3, :3])
        )
        self.assertAlmostEqual(result["rho_all"], np.trace(result["pcrb"]))
        self.assertGreaterEqual(result["rho_all"], result["rho_pos"])
        self.assertGreater(result["rho_pos"], 0.0)
        self.assertTrue(np.all(np.isfinite(result["j"])))
        self.assertTrue(np.all(np.isfinite(result["pcrb"])))

    def test_noisy_estimate_is_reproducible_and_does_not_mutate_truth(self):
        truth = TARGET.copy()
        first = ISACPhysics(EnvConfig(), np.random.default_rng(3)).noisy_estimate(
            truth
        )
        second = ISACPhysics(EnvConfig(), np.random.default_rng(3)).noisy_estimate(
            truth
        )

        np.testing.assert_allclose(first, second)
        self.assertFalse(np.array_equal(first, truth))
        np.testing.assert_allclose(truth, TARGET)


if __name__ == "__main__":
    unittest.main()