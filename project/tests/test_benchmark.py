import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from project.artifacts import save_policy
from project.baselines import RandomController
from project.benchmark import evaluate_episode, run_benchmark
from project.benchmark_setup import load_experiment
from project.config import EnvConfig, PPOConfig, RewardConfig
from project.core import MAPPOActorCritic
from project.env import DualUAVEnv
from project.observations import OBS_DIM, STATE_DIM


class BenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self):
        tests_dir = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(dir=tests_dir)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.suite = self.root / "suite.json"
        self.output = self.root / "result"
        environment = {
            "max_steps": 3,
            "antenna_x": 2,
            "antenna_y": 2,
            "min_uav_distance": 2.0,
            "goal_tolerance": 0.2,
            "uav_initial": [[0, -10, 10], [0, 10, 10]],
            "uav_goal_positions": [[20, -10, 10], [20, 10, 10]],
            "world_low": [-30, -30, 1],
            "world_high": [30, 30, 30],
            "target_initial_state": [10, 0, 12, 0, 0, 0],
            "target_acceleration": [0, 0, 0],
        }
        self.config.write_text(json.dumps({"environment": environment}), encoding="utf-8")
        self.suite.write_text(json.dumps({
            "prefix_steps": 2,
            "scenarios": {"short": {"description": "Short test.", "environment": {}}},
            "methods": {"goal": {}, "random": {}},
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def args(self, **changes):
        values = dict(
            checkpoint=None, config=self.config, suite=self.suite,
            methods=["goal", "random"], scenarios=["short"], seed_start=11,
            episodes=2, eval_seeds=None, reference="goal", device="cpu",
            output=self.output, no_plots=True,
        )
        values.update(changes)
        return Namespace(**values)

    @staticmethod
    def without_timing(row):
        return {key: value for key, value in row.items() if key != "policy_ms_per_step"}

    def test_run_writes_complete_outputs_and_first_seed_trace(self):
        rows, summaries = run_benchmark(self.args())
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(summaries), 2)
        self.assertEqual({(row["method"], row["seed"]) for row in rows},
                         {(method, seed) for method in ("goal", "random") for seed in (11, 12)})

        expected = {"manifest.json", "episodes.csv", "summary.csv", "paired.csv",
                    "summary.json", "paired.json", "REPORT.md"}
        self.assertTrue(expected.issubset({path.name for path in self.output.iterdir()}))
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["total_episodes"], 4)
        self.assertEqual(manifest["evaluation_seeds"], [11, 12])
        self.assertEqual(manifest["methods"], ["goal", "random"])
        self.assertEqual(manifest["prefix_steps"], 2)
        self.assertIn("controller_settings", manifest)
        self.assertIn("## short", (self.output / "REPORT.md").read_text(encoding="utf-8"))

        with (self.output / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        with (self.output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            summary_csv = list(csv.DictReader(stream))
        self.assertEqual(len(summary_csv), 2)
        self.assertTrue(all(row["episodes"] == "2" and row["rho_prefix_mean_n"] == "2"
                            for row in summary_csv))
        with (self.output / "paired.csv").open(encoding="utf-8-sig", newline="") as stream:
            paired = list(csv.DictReader(stream))
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["episode_return_n"], "2")

        _, _, scenarios, _, _, metadata = load_experiment(self.args())
        cfg = scenarios["short"]
        expected_row, expected_trace = evaluate_episode(
            RandomController(cfg), DualUAVEnv(cfg, RewardConfig()), 11, metadata["prefix_steps"])
        saved = np.load(self.output / "trajectories" / "short" / "random" / "trajectory.npz")
        self.assertEqual(set(saved.files), set(expected_trace))
        for key, value in expected_trace.items():
            np.testing.assert_array_equal(saved[key], value)
        actual = next(row for row in rows if row["method"] == "random" and row["seed"] == 11)
        self.assertEqual(self.without_timing(actual),
                         {**self.without_timing(expected_row), "scenario": "short", "method": "random",
                          "prefix_steps": 2})

    def test_random_episode_reset_is_reproducible_except_timing(self):
        _, reward, scenarios, _, _, metadata = load_experiment(self.args())
        cfg = scenarios["short"]
        policy = RandomController(cfg, seed=999)
        env = DualUAVEnv(cfg, reward)
        first, first_trace = evaluate_episode(policy, env, 12, metadata["prefix_steps"])
        second, second_trace = evaluate_episode(policy, env, 12, metadata["prefix_steps"])
        self.assertEqual(self.without_timing(first), self.without_timing(second))
        for key in first_trace:
            np.testing.assert_array_equal(first_trace[key], second_trace[key])

    def test_existing_output_directory_is_rejected(self):
        run_benchmark(self.args())
        with self.assertRaises(FileExistsError):
            run_benchmark(self.args())

    def test_checkpoint_supplies_base_config_before_scenario_override(self):
        checkpoint = self.root / "policy.pt"
        checkpoint_env = EnvConfig(max_steps=7, slot_duration=.5, antenna_x=2, antenna_y=2)
        checkpoint_reward = RewardConfig(progress=17.)
        ppo = PPOConfig(hidden_sizes=(8,), steps_per_epoch=1, epochs=2, seed=19)
        model = MAPPOActorCritic(OBS_DIM, STATE_DIM, hidden_sizes=ppo.hidden_sizes)
        save_policy(checkpoint, model, checkpoint_env, checkpoint_reward, ppo)
        conflicting = self.root / "conflicting.json"
        conflicting.write_text(json.dumps({
            "environment": {"max_steps": 99}, "reward": {"progress": 999.}
        }), encoding="utf-8")
        checkpoint_suite = self.root / "checkpoint_suite.json"
        checkpoint_suite.write_text(json.dumps({
            "prefix_steps": 2,
            "scenarios": {"changed_dt": {
                "description": "Override one field.", "environment": {"slot_duration": .25}}},
        }), encoding="utf-8")
        args = self.args(checkpoint=checkpoint, config=conflicting, suite=checkpoint_suite,
                         scenarios=["changed_dt"], methods=["goal"])
        _, reward, scenarios, methods, seeds, metadata = load_experiment(args)
        self.assertEqual(scenarios["changed_dt"].max_steps, 7)
        self.assertEqual(scenarios["changed_dt"].slot_duration, .25)
        self.assertEqual(reward.progress, 17.)
        self.assertEqual(methods, ["goal"])
        self.assertEqual(seeds, [11, 12])
        self.assertEqual(metadata["checkpoint"]["training_seed"], 19)
        self.assertEqual(metadata["base_config"]["environment"]["max_steps"], 7)


if __name__ == "__main__":
    unittest.main()
