import unittest

import numpy as np

from project.config import EnvConfig
from project.training_scenarios import TrainingScenarioSampler


class TrainingScenarioTests(unittest.TestCase):
    def test_fixed_returns_equivalent_independent_config(self):
        base = EnvConfig()
        sampled = TrainingScenarioSampler(base, "fixed", 3).sample()
        self.assertIsNot(sampled, base)
        np.testing.assert_array_equal(sampled.uav_initial, base.uav_initial)

    def test_randomized_stream_is_reproducible_and_configs_remain_valid(self):
        left = TrainingScenarioSampler(EnvConfig(), "randomized", 9)
        right = TrainingScenarioSampler(EnvConfig(), "randomized", 9)
        for _ in range(30):
            a, b = left.sample(), right.sample()
            np.testing.assert_array_equal(a.uav_initial, b.uav_initial)
            np.testing.assert_array_equal(a.uav_goal_positions, b.uav_goal_positions)
            self.assertEqual(a.max_steps, b.max_steps)
            self.assertEqual(left.current_kind, right.current_kind)
            self.assertEqual(a.max_uav_speed, 5.)
            self.assertEqual(a.world_high.tolist(), EnvConfig().world_high.tolist())
            minimum_goal_gap = a.min_uav_distance + 2 * a.goal_tolerance
            self.assertGreaterEqual(
                np.linalg.norm(a.uav_goal_positions[0] - a.uav_goal_positions[1]),
                minimum_goal_gap)
            distances = np.linalg.norm(a.uav_goal_positions - a.uav_initial, axis=1)
            lower = np.max(np.ceil(
                np.maximum(0., distances - a.goal_tolerance)
                / (a.max_uav_speed * a.slot_duration)))
            self.assertGreaterEqual(a.max_steps, lower + 10)

    def test_scenario_rng_does_not_touch_numpy_global_rng(self):
        np.random.seed(42)
        expected = np.random.random()
        np.random.seed(42)
        sampler = TrainingScenarioSampler(EnvConfig(), "randomized", 2)
        for _ in range(5):
            sampler.sample()
        self.assertEqual(np.random.random(), expected)

    def test_continuous_distribution_includes_reverse_direction_goals(self):
        sampler = TrainingScenarioSampler(EnvConfig(), "randomized", 19)
        offsets = []
        while len(offsets) < 20:
            config = sampler.sample()
            if sampler.current_kind == "continuous":
                offsets.extend((config.uav_goal_positions - config.uav_initial)[:, 0])
        self.assertTrue(any(offset < 0 for offset in offsets))
        self.assertTrue(any(offset > 0 for offset in offsets))


if __name__ == "__main__":
    unittest.main()
