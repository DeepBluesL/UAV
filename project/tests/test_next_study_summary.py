import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.next_study_summary import (_across_training_seeds, _paired_ablations, _seed_row,
                                        summarize_next_study)
from project.next_study_plots import plot_next_study
from project.next_study_io import variant_epochs


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode(seed, success, scenario="nominal", method="mappo"):
    return {
        "seed": seed, "scenario": scenario, "method": method,
        "team_success": success, "restricted_team_time": 10 + seed % 2,
        "safety_interventions": 0, "collisions": 0, "boundary_requests": 0,
        "tracking_prefix_rmse": 1., "rho_prefix_mean": .5,
        "communication_prefix_rate": .9, "tracking_prefix_late_rmse": 1.2,
        "position_nees_prefix_mean": 3., "episode_return": 5.,
    }


class NextStudySummaryTests(unittest.TestCase):
    def test_variant_epochs_respect_each_planned_maximum(self):
        manifest = {"primary_epoch": 100, "budget_epochs": [100, 300, 500]}
        self.assertEqual(variant_epochs({"name": "short", "max_epochs": 100}, manifest), [100])
        self.assertEqual(variant_epochs({"name": "long", "max_epochs": 500}, manifest),
                         [100, 300, 500])

    def test_hierarchical_average_equal_weights_uneven_episode_counts(self):
        spec = {"name": "nav_v2", "demo": True}
        first = _seed_row("nav_v2", 7, 100, "nominal",
                          [episode(1, True)] * 2, spec, 100)
        second = _seed_row("nav_v2", 17, 100, "nominal",
                           [episode(1, False)] * 8, spec, 100)
        result = _across_training_seeds([first, second], [7, 17])[0]
        self.assertEqual(result["evaluation_episodes"], 10)
        self.assertEqual(result["success_rate_mean"], .5)
        self.assertAlmostEqual(result["success_rate_std"], np.sqrt(.5))
        self.assertTrue(result["demo"])

    def test_paired_ablations_keep_metric_specific_n_and_std(self):
        spec, rows = {}, []
        for seed, left, right in ((7, 10., 13.), (17, 20., 19.)):
            for method, elapsed in (("left", left), ("right", right)):
                row = _seed_row(method, seed, 100, "nominal",
                                [episode(5101, True)], spec, 100)
                row["restricted_team_time"] = elapsed
                rows.append(row)
        result = _paired_ablations(rows, {
            "training_seeds": [7, 17],
            "paired_ablations": [{"left": "left", "right": "right"}],
        })[0]
        self.assertEqual(result["restricted_team_time_difference_n"], 2)
        self.assertAlmostEqual(result["restricted_team_time_difference_mean"], 1.)
        self.assertAlmostEqual(result["restricted_team_time_difference_std"], np.sqrt(8.))

    def test_summary_refuses_missing_requested_evaluation_seed(self):
        with tempfile.TemporaryDirectory(dir="project/tests") as directory:
            output = Path(directory)
            manifest = {
                "status": "complete", "variants": [{"name": "nav_v2", "epoch": 100}],
                "primary_epoch": 100, "budget_epochs": [100],
                "training_seeds": [7], "evaluation_seeds": [5101, 5102],
            }
            (output / "study_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            learned = output / "eval" / "nav_v2_seed7_epoch100" / "episodes.csv"
            write_csv(learned, [episode(5101, True)])
            rules = output / "eval" / "rules" / "episodes.csv"
            write_csv(rules, [episode(5101, True, method="goal"),
                              episode(5102, True, method="goal")])
            with self.assertRaisesRegex(ValueError, "Incomplete evaluation seeds"):
                summarize_next_study(output)

    def test_complete_fixture_writes_bilingual_reports(self):
        with tempfile.TemporaryDirectory(dir="project/tests") as directory:
            output = Path(directory)
            manifest = {
                "status": "complete", "variants": [{"name": "nav_v2", "epoch": 100}],
                "primary_epoch": 100, "budget_epochs": [100],
                "training_seeds": [7], "evaluation_seeds": [5101, 5102],
            }
            (output / "study_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            learned = output / "eval" / "nav_v2_seed7_epoch100" / "episodes.csv"
            write_csv(learned, [episode(5101, True), episode(5102, False)])
            rules = output / "eval" / "rules" / "episodes.csv"
            write_csv(rules, [episode(5101, True, method="goal"),
                              episode(5102, True, method="goal")])
            result = summarize_next_study(output)
            self.assertEqual(result["rows"][0]["training_seeds"], 1)
            self.assertTrue((output / "REPORT.md").exists())
            self.assertTrue((output / "REPORT.en.md").exists())
            figures = plot_next_study(output)
            self.assertTrue(all(path.exists() for path in figures))


if __name__ == "__main__":
    unittest.main()
