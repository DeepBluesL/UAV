import unittest
from types import SimpleNamespace

import numpy as np

from project.benchmark_metrics import enrich_episode, paired_comparisons, summarize


def base_row(success=1, length=4):
    return {
        "episode_return": 8., "length": length, "team_success": success,
        "rho_pos_mean": 9., "safety_interventions": 1, "collisions": 0,
        "boundary_requests": 2, "path_length_0": 3., "path_length_1": 5.,
        "energy_proxy_0": 7., "energy_proxy_1": 11.,
        "goal_distance_0": 2., "goal_distance_1": 6.,
        "active_steps_0": 4, "active_steps_1": 2,
        "communication_rate_0": .5, "communication_rate_1": 1.,
    }


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(slot_duration=.5, max_steps=20)

    def test_derived_totals_weighted_rate_and_success_time(self):
        result = enrich_episode(base_row(), {"rho_pos": np.arange(7.)}, self.config, 3)
        self.assertEqual(result["restricted_team_time"], 2.)
        self.assertEqual(result["success_time"], 2.)
        self.assertEqual(result["total_path_length"], 8.)
        self.assertEqual(result["total_energy_proxy"], 18.)
        self.assertEqual(result["goal_distance_mean"], 4.)
        self.assertAlmostEqual(result["communication_rate"], 2 / 3)
        self.assertEqual(result["rho_pos_mean"], 9.)

    def test_failure_uses_horizon_penalty(self):
        result = enrich_episode(base_row(0, 3), {"rho_pos": np.arange(4.)}, self.config, 3)
        self.assertEqual(result["restricted_team_time"], 10.)
        self.assertIsNone(result["success_time"])

    def test_prefix_excludes_initial_and_does_not_pad_short_trace(self):
        full = enrich_episode(base_row(), {"rho_pos": [100., 1., 2., 3.]}, self.config, 3)
        short = enrich_episode(base_row(), {"rho_pos": [100., 1., 2.]}, self.config, 3)
        self.assertEqual(full["rho_prefix_mean"], 2.)
        self.assertIsNone(short["rho_prefix_mean"])


class SummaryTests(unittest.TestCase):
    def test_empty_metric_single_sample_and_wilson_boundary(self):
        row = enrich_episode(base_row(), {"rho_pos": [0., 1., 2.]},
                             SimpleNamespace(slot_duration=1., max_steps=4), 2)
        row.update(scenario="easy", method="goal", seed=1, policy_ms_per_step=None)
        summary = summarize([row])[0]
        self.assertEqual(summary["success_rate"], 1.)
        self.assertLess(summary["success_ci_low"], 1.)
        self.assertEqual(summary["success_time_n"], 1)
        self.assertIsNone(summary["success_time_std"])
        self.assertEqual(summary["policy_ms_per_step_n"], 0)
        self.assertIsNone(summary["policy_ms_per_step_mean"])

    def test_sample_std_and_reproducible_bootstrap(self):
        rows = []
        for seed, value in enumerate([1., 3.]):
            row = base_row()
            row.update(scenario="s", method="m", seed=seed, episode_return=value,
                       restricted_team_time=value, success_time=value,
                       total_path_length=value, total_energy_proxy=value,
                       goal_distance_mean=value, communication_rate=value,
                       rho_prefix_mean=value, policy_ms_per_step=value)
            rows.append(row)
        first, second = summarize(rows)[0], summarize(rows)[0]
        self.assertAlmostEqual(first["episode_return_std"], np.sqrt(2))
        self.assertEqual(first["episode_return_ci_low"], second["episode_return_ci_low"])


class PairedTests(unittest.TestCase):
    def test_difference_is_method_minus_reference_on_seed_intersection(self):
        rows = [
            {"scenario": "s", "method": "goal", "seed": 1,
             "episode_return": 2., "restricted_team_time": 8., "rho_prefix_mean": 4.},
            {"scenario": "s", "method": "learned", "seed": 1,
             "episode_return": 5., "restricted_team_time": 6., "rho_prefix_mean": 3.},
            {"scenario": "s", "method": "learned", "seed": 2,
             "episode_return": 99., "restricted_team_time": 1., "rho_prefix_mean": 1.},
        ]
        result = paired_comparisons(rows)[0]
        self.assertEqual(result["episode_return_mean"], 3.)
        self.assertEqual(result["restricted_team_time_mean"], -2.)
        self.assertEqual(result["rho_prefix_mean_mean"], -1.)
        self.assertEqual(result["episode_return_n"], 1)

    def test_duplicate_seed_key_is_rejected(self):
        row = {"scenario": "s", "method": "goal", "seed": 1}
        with self.assertRaises(ValueError):
            paired_comparisons([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
