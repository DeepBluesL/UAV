import unittest

import numpy as np

from project.config import EnvConfig, RewardConfig
from project.rewards import team_reward


class TestTeamReward(unittest.TestCase):
    def setUp(self):
        self.env = EnvConfig()
        self.weights = RewardConfig()
        self.base = dict(
            previous_distances=np.array([100.0, 100.0]),
            distances=np.array([100.0, 100.0]),
            active_before=np.array([True, True]),
            newly_arrived=np.array([False, False]),
            arrived=np.array([False, False]),
            movement_velocities=np.zeros((2, 3)),
            accelerations=np.zeros((2, 3)),
            communication_sinrs=np.full(2, self.env.gamma_min),
            rho_pos=self.weights.rho_ref,
            safety_intervention=False,
            boundary_requests=np.array([False, False]),
            completed=False,
            timed_out=False,
        )

    def parts(self, **changes):
        args = self.base.copy()
        args.update(changes)
        return team_reward(self.env, self.weights, **args)

    def test_progress_has_known_scale_and_is_monotonic(self):
        stationary = self.parts()
        one_uav_moves_five = self.parts(distances=np.array([95.0, 100.0]))
        both_move_five = self.parts(distances=np.array([95.0, 95.0]))
        self.assertEqual(stationary["progress"], 0.0)
        self.assertAlmostEqual(one_uav_moves_five["progress"], self.weights.progress / 2)
        self.assertAlmostEqual(both_move_five["progress"], self.weights.progress)
        self.assertLess(stationary["progress"], one_uav_moves_five["progress"])
        self.assertLess(one_uav_moves_five["progress"], both_move_five["progress"])

    def test_remaining_distance_penalty_grows_with_distance(self):
        near = self.parts(distances=np.array([10.0, 10.0]))["distance"]
        middle = self.parts(distances=np.array([100.0, 100.0]))["distance"]
        far = self.parts(distances=np.array([1000.0, 1000.0]))["distance"]
        self.assertGreater(near, middle)
        self.assertGreater(middle, far)
        self.assertAlmostEqual(middle, -self.weights.distance / 2)

    def test_lower_rho_is_better_and_sensing_is_bounded(self):
        zero = self.parts(rho_pos=0.0)["sensing"]
        low = self.parts(rho_pos=self.weights.rho_ref / 10)["sensing"]
        high = self.parts(rho_pos=self.weights.rho_ref * 10)["sensing"]
        self.assertEqual(zero, 0.0)
        self.assertGreater(low, high)
        self.assertGreaterEqual(low, -self.weights.sensing)
        self.assertGreater(high, -self.weights.sensing)
        self.assertLessEqual(high, 0.0)

    def test_log1p_sensing_penalty_uses_reward_rho_reference(self):
        self.weights = RewardConfig(sensing=.2, rho_ref=2., sensing_penalty="log1p")
        self.base["rho_pos"] = 6.
        self.assertAlmostEqual(self.parts()["sensing"], -.2 * np.log1p(3.))

    def test_two_uav_denominator_does_not_change_with_active_count(self):
        both = self.parts(active_before=np.array([True, True]))["time"]
        one = self.parts(active_before=np.array([True, False]))["time"]
        self.assertAlmostEqual(both, -self.weights.time)
        self.assertAlmostEqual(one, -self.weights.time / 2)

    def test_inactive_placeholders_do_not_change_activity_costs(self):
        active = np.array([True, False])
        clean = self.parts(active_before=active)
        placeholders = self.parts(
            active_before=active,
            previous_distances=np.array([100.0, -1.0e9]),
            distances=np.array([100.0, 1.0e9]),
            movement_velocities=np.array([[0.0, 0.0, 0.0], [1.0e9, -1.0e9, 1.0e9]]),
            accelerations=np.array([[0.0, 0.0, 0.0], [1.0e9, 1.0e9, 1.0e9]]),
            communication_sinrs=np.array([self.env.gamma_min, -1.0e9]),
            boundary_requests=np.array([False, True]),
        )
        for name in ("progress", "distance", "time", "energy", "communication", "boundary"):
            self.assertEqual(placeholders[name], clean[name], name)

    def test_arrival_completion_and_timeout_weights(self):
        one_arrival = self.parts(newly_arrived=np.array([True, False]))
        self.assertEqual(one_arrival["arrival"], self.weights.arrival / 2)

        all_arrive = self.parts(
            newly_arrived=np.array([True, True]),
            arrived=np.array([True, True]),
            completed=True,
        )
        self.assertEqual(all_arrive["arrival"], self.weights.arrival)
        self.assertEqual(all_arrive["completion"], self.weights.completion)
        self.assertEqual(all_arrive["timeout"], 0.0)

        one_unfinished = self.parts(
            arrived=np.array([True, False]),
            timed_out=True,
        )
        both_unfinished = self.parts(timed_out=True)
        self.assertEqual(one_unfinished["timeout"], -self.weights.timeout / 2)
        self.assertEqual(both_unfinished["timeout"], -self.weights.timeout)


if __name__ == "__main__":
    unittest.main()
