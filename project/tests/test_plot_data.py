import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from project.config import EnvConfig, RewardConfig
from project.env import DualUAVEnv
from project.evaluate import GoalController, rollout_policy
from project.replot import load_trace


class PlotDataTests(unittest.TestCase):
    def test_trace_separates_truth_estimate_and_nonzero_bs(self):
        cfg = EnvConfig(max_steps=2, bs_position=np.array([11., -7., 3.]))
        env = DualUAVEnv(cfg, RewardConfig(), seed=4)
        _, trace = rollout_policy(GoalController(cfg), env, seed=4)
        np.testing.assert_array_equal(trace["bs_position"], cfg.bs_position)
        self.assertEqual(trace["target_positions"].shape,
                         trace["estimated_target_positions"].shape)
        self.assertFalse(np.shares_memory(trace["target_positions"],
                                          trace["estimated_target_positions"]))
        self.assertTrue(np.any(trace["target_positions"] !=
                               trace["estimated_target_positions"]))

    def test_old_trace_loads_only_bs_from_config(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            output = Path(folder)
            np.savez(output / "trajectory.npz", target_positions=np.zeros((2, 3)))
            (output / "config.json").write_text(json.dumps({
                "environment": {"bs_position": [9, 8, 7]}}), encoding="utf-8")
            trace = load_trace(output)
            np.testing.assert_array_equal(trace["bs_position"], [9, 8, 7])
            self.assertNotIn("estimated_target_positions", trace)


if __name__ == "__main__":
    unittest.main()
