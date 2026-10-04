import unittest

import numpy as np

from project.config import EnvConfig
from project.target_motion import advance_target_state


class TargetMotionTests(unittest.TestCase):
    def test_default_matches_legacy_constant_acceleration_arithmetic(self):
        cfg = EnvConfig()
        state = cfg.target_initial_state.copy()
        expected = state.copy()
        expected[:3] += state[3:] * cfg.slot_duration + (
            .5 * cfg.target_acceleration * cfg.slot_duration ** 2)
        expected[3:] += cfg.target_acceleration * cfg.slot_duration
        np.testing.assert_array_equal(advance_target_state(state, cfg, 0), expected)

    def test_optional_turn_switches_after_absolute_step(self):
        cfg = EnvConfig(
            target_turn_step=1,
            target_acceleration_after_turn=np.array([.5, -.4, .3]))
        first = advance_target_state(cfg.target_initial_state, cfg, 0)
        expected_first_velocity = cfg.target_initial_state[3:] + cfg.target_acceleration
        np.testing.assert_allclose(first[3:], expected_first_velocity)
        second = advance_target_state(first, cfg, 1)
        np.testing.assert_allclose(
            second[3:], first[3:] + cfg.target_acceleration_after_turn)

    def test_invalid_turn_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            EnvConfig(target_turn_step=-1)
        with self.assertRaises(ValueError):
            EnvConfig(target_acceleration_after_turn=np.zeros(2))


if __name__ == "__main__":
    unittest.main()
