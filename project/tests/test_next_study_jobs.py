import json
from pathlib import Path
import tempfile
import unittest

from project.config import EnvConfig, PPOConfig, RewardConfig
from project.next_study_jobs import PROJECT, prepare_study


class StudyPreparationTests(unittest.TestCase):
    def test_generated_configs_keep_total_demo_budget_and_short_snapshots(self):
        protocol = json.loads((PROJECT / "next_study_protocol.json").read_text(encoding="utf-8"))
        protocol["status"] = "frozen"
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as temp:
            manifest, tasks = prepare_study(protocol, Path(temp) / "study", 3)
            self.assertEqual(len(tasks), len(protocol["variants"]) * len(protocol["training_seeds"]))
            self.assertEqual(len(manifest["frozen_scenarios"]), 15)
            for name, path, _ in tasks:
                settings = json.loads(path.read_text(encoding="utf-8"))
                EnvConfig(**settings["environment"])
                RewardConfig(**settings["reward"])
                ppo = PPOConfig(**settings["ppo"])
                expected = (100,) if ppo.epochs == 100 else (100, 300, 500)
                self.assertEqual(ppo.checkpoint_epochs, expected)
                self.assertEqual(ppo.device, "cuda")
                self.assertEqual(ppo.rollout_device, "cpu")
                if name.startswith("pure_goal_bc"):
                    self.assertEqual(ppo.demonstration_steps, 8192)
                    self.assertEqual(100 * ppo.steps_per_epoch - ppo.demonstration_steps, 196608)
            self.assertTrue((Path(temp) / "study" / "source.zip").is_file())

    def test_development_placeholders_cannot_launch_training(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as temp:
            output = Path(temp) / "study"
            with self.assertRaises(ValueError):
                prepare_study({"status": "developing"}, output, 3)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
