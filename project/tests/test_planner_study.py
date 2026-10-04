import unittest

from project.planner_study import configurations, controller, family_pairs
from project.sensing_planning import SensingAwareMPCController, SensingAwareSAController


class TestPlannerStudy(unittest.TestCase):
    def test_grid_controller_settings_preserve_family_budgets(self):
        configs, _ = configurations(["nominal"])
        cfg = configs["nominal"]
        sa = controller("sa", {"family": "sa", "sensing_weight": 20,
                               "communication_weight": 0}, cfg, 1.)
        mpc = controller("mpc", {"family": "mpc", "sensing_weight": 2000,
                                 "communication_weight": 0}, cfg, 1.)
        self.assertIsInstance(sa, SensingAwareSAController)
        self.assertIsInstance(mpc, SensingAwareMPCController)
        self.assertEqual((sa.horizon, sa.iterations), (4, 16))
        self.assertEqual(mpc.horizon, 3)
        self.assertEqual(sa.communication_weight, 0.)
        self.assertEqual(mpc.uncertainty_ref, 1.)

    def test_family_pairs_use_same_family_navigation_reference(self):
        rows = []
        for method, value in (("sa_nav", 2.), ("sa_sense", 1.),
                              ("mpc_nav", 4.), ("mpc_sense", 3.)):
            rows.append({"scenario": "nominal", "method": method, "seed": 1,
                         "tracking_prefix_rmse": value})
        grid = {"methods": {"sa_nav": {"family": "sa"}, "sa_sense": {"family": "sa"},
                            "mpc_nav": {"family": "mpc"}, "mpc_sense": {"family": "mpc"}},
                "family_references": {"sa": "sa_nav", "mpc": "mpc_nav"}}
        pairs = family_pairs(rows, grid)
        self.assertEqual({row["reference"] for row in pairs}, {"sa_nav", "mpc_nav"})
        self.assertTrue(all(row["tracking_prefix_rmse_mean"] == -1. for row in pairs))

    def test_optional_search_settings_reach_constructor(self):
        configs, _ = configurations(["nominal"])
        policy = controller("sa64", {"family": "sa", "sensing_weight": 2.,
                                     "communication_weight": 0.,
                                     "search": {"horizon": 4, "iterations": 64}},
                            configs["nominal"], 1.)
        self.assertEqual((policy.horizon, policy.iterations), (4, 64))
        self.assertEqual(policy.settings["model_evaluations_per_action"], 69)

    def test_comparison_groups_separate_search_budgets(self):
        rows = [{"scenario": "x", "method": method, "seed": 1,
                 "tracking_prefix_rmse": value}
                for method, value in (("nav16", 2.), ("sense16", 1.),
                                      ("nav64", 4.), ("sense64", 2.))]
        grid = {"methods": {
            "nav16": {"family": "sa", "comparison_group": "sa16"},
            "sense16": {"family": "sa", "comparison_group": "sa16"},
            "nav64": {"family": "sa", "comparison_group": "sa64"},
            "sense64": {"family": "sa", "comparison_group": "sa64"}},
            "family_references": {"sa16": "nav16", "sa64": "nav64"}}
        pairs = family_pairs(rows, grid)
        self.assertEqual({row["reference"] for row in pairs}, {"nav16", "nav64"})


if __name__ == "__main__":
    unittest.main()
