import unittest

import numpy as np

from project.config import EnvConfig
from project.domain_randomization import DOMAIN_RANGES
from project.env import DualUAVEnv
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

    def test_domain_randomized_ranges_and_geometry_feasibility(self):
        sampler = TrainingScenarioSampler(EnvConfig(), "domain_randomized", 23)
        limits = DOMAIN_RANGES
        for _ in range(40):
            cfg = sampler.sample()
            self.assertTrue(limits["max_uav_speed"][0] <= cfg.max_uav_speed
                            <= limits["max_uav_speed"][1])
            self.assertTrue(limits["max_uav_acceleration"][0] <= cfg.max_uav_acceleration
                            <= limits["max_uav_acceleration"][1])
            self.assertTrue(np.all(cfg.world_low >= limits["world_low"][0]))
            self.assertTrue(np.all(cfg.world_low <= limits["world_low"][1]))
            self.assertTrue(np.all(cfg.world_high >= limits["world_high"][0]))
            self.assertTrue(np.all(cfg.world_high <= limits["world_high"][1]))
            self.assertTrue(np.all(cfg.target_initial_state[:3]
                                   >= limits["target_position"][0]))
            self.assertTrue(np.all(cfg.target_initial_state[:3]
                                   <= limits["target_position"][1]))
            self.assertTrue(np.all(cfg.target_initial_state[3:]
                                   >= limits["target_velocity"][0]))
            self.assertTrue(np.all(cfg.target_initial_state[3:]
                                   <= limits["target_velocity"][1]))
            self.assertTrue(np.all(cfg.target_acceleration
                                   >= limits["target_acceleration"][0]))
            self.assertTrue(np.all(cfg.target_acceleration
                                   <= limits["target_acceleration"][1]))
            self.assertTrue(np.all(cfg.uav_initial > cfg.world_low))
            self.assertTrue(np.all(cfg.uav_initial < cfg.world_high))
            self.assertTrue(np.all(cfg.uav_goal_positions > cfg.world_low))
            self.assertTrue(np.all(cfg.uav_goal_positions < cfg.world_high))
            self.assertGreaterEqual(
                np.linalg.norm(cfg.uav_goal_positions[0] - cfg.uav_goal_positions[1]),
                cfg.min_uav_distance + 2 * cfg.goal_tolerance)
            distances = np.linalg.norm(cfg.uav_goal_positions - cfg.uav_initial, axis=1)
            lower = np.max(np.ceil(
                np.maximum(0., distances - cfg.goal_tolerance)
                / (cfg.max_uav_speed * cfg.slot_duration)))
            self.assertGreaterEqual(cfg.max_steps, lower + 10)
            self.assertTrue(.8 <= cfg.measurement_range_std_floor <= 1.2)
            self.assertTrue(np.deg2rad(.4) <= cfg.measurement_angle_std_floor
                            <= np.deg2rad(.7))
            self.assertTrue(.4 <= cfg.measurement_range_rate_std_floor <= .7)
            self.assertTrue(2. <= cfg.gamma_min <= 5.)
            self.assertTrue(.05 <= cfg.imperfect_csi_beta <= .31)

    def test_domain_stream_is_reproducible(self):
        left = TrainingScenarioSampler(EnvConfig(), "domain_randomized", 31)
        right = TrainingScenarioSampler(EnvConfig(), "domain_randomized", 31)
        for _ in range(10):
            a, b = left.sample(), right.sample()
            for name in ("uav_initial", "uav_goal_positions", "world_low", "world_high",
                         "target_initial_state", "target_acceleration"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            self.assertEqual(a.max_uav_speed, b.max_uav_speed)

    def test_curriculum_absolute_interaction_phases(self):
        sampler = TrainingScenarioSampler(EnvConfig(), "curriculum", 5)
        self.assertEqual(sampler.phase, "easy")
        easy = sampler.sample()
        np.testing.assert_array_equal(easy.uav_initial, EnvConfig().uav_initial)
        sampler.advance(50_000)
        self.assertEqual(sampler.phase, "geometry")
        sampler.sample()
        self.assertTrue(sampler.current_kind.startswith("curriculum_geometry_"))
        sampler.advance(99_999)
        self.assertEqual(sampler.phase, "geometry")
        sampler.advance(1)
        self.assertEqual(sampler.phase, "domain")
        domain = sampler.sample()
        self.assertTrue(sampler.current_kind.startswith("curriculum_domain_"))
        self.assertTrue(4. <= domain.max_uav_speed <= 7.)

    def test_maneuver_parameters_are_not_added_to_observations(self):
        base = EnvConfig(target_turn_step=None)
        turn = EnvConfig(target_turn_step=0,
                         target_acceleration_after_turn=np.array([1., 2., 3.]))
        left = DualUAVEnv(base, seed=17)
        right = DualUAVEnv(turn, seed=17)
        left_obs, left_state, _ = left.reset(seed=17)
        right_obs, right_state, _ = right.reset(seed=17)
        np.testing.assert_array_equal(left_obs, right_obs)
        np.testing.assert_array_equal(left_state, right_state)
        self.assertEqual(left_obs.shape, right_obs.shape)
        self.assertEqual(left_state.shape, right_state.shape)


if __name__ == "__main__":
    unittest.main()
