import unittest

import numpy as np

from project.baselines import (PDController, PotentialFieldController, RandomController,
                               acceleration_to_action)
from project.config import EnvConfig
from project.env import DualUAVEnv
from project.mpc import MPCController


def config(**overrides):
    values = dict(
        max_steps=10, min_uav_distance=2.0, goal_tolerance=.2,
        uav_initial=np.array([[0., -2., 10.], [0., 2., 10.]]),
        uav_goal_positions=np.array([[12., 2., 10.], [12., -2., 10.]]),
        world_low=np.array([-20., -20., 1.]), world_high=np.array([20., 20., 30.]),
        target_initial_state=np.array([6., 8., 12., 0., 0., 0.]),
        target_acceleration=np.zeros(3), antenna_x=2, antenna_y=2,
        imperfect_csi_beta=0., csi_beta_jitter=0.)
    values.update(overrides)
    return EnvConfig(**values)


class TestBaselines(unittest.TestCase):
    def test_interface_is_finite_and_inactive_is_stationary(self):
        cfg = config(uav_goal_positions=np.array([[0., -2., 10.], [12., -2., 10.]]))
        env = DualUAVEnv(cfg, seed=2)
        obs, _, info = env.reset(seed=2)
        self.assertFalse(info["active"][0])
        for policy in (RandomController(cfg), PDController(cfg),
                       PotentialFieldController(cfg), MPCController(cfg)):
            action, value = policy.act(obs, info["active"])
            self.assertEqual(action.shape, (2, 3))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_array_equal(action[0], 0.)
            np.testing.assert_array_equal(value, np.zeros(2))
            before = env.uav_positions[0].copy()
            env.step(action)
            np.testing.assert_allclose(env.uav_positions[0], before)
            obs, _, info = env.reset(seed=2)

    def test_random_reset_reproduces_episode_sequence(self):
        policy = RandomController(config(), seed=13)
        obs = np.zeros((2, 31), dtype=np.float32)
        first = [policy.act(obs, np.ones(2, dtype=bool))[0] for _ in range(3)]
        policy.reset(13)
        second = [policy.act(obs, np.ones(2, dtype=bool))[0] for _ in range(3)]
        np.testing.assert_array_equal(first, second)

    def test_potential_field_responds_to_head_on_threat(self):
        cfg = config()
        env = DualUAVEnv(cfg, seed=3)
        obs, _, info = env.reset(seed=3)
        clear = obs.copy()
        clear[:, 19:22] *= 5.0
        policy = PotentialFieldController(cfg)
        threat, _ = policy.act(obs, info["active"])
        distant, _ = policy.act(clear, info["active"])
        # 相向目标下，近距离排斥应增强两机 y 方向分离。
        self.assertLess(threat[0, 1], distant[0, 1])
        self.assertGreater(threat[1, 1], distant[1, 1])

    def test_mpc_first_step_does_not_cross_teammate(self):
        cfg = config(min_uav_distance=3.0)
        env = DualUAVEnv(cfg, seed=4)
        obs, _, info = env.reset(seed=4)
        action, _ = MPCController(cfg).act(obs, info["active"])
        _, _, _, _, _, result = env.step(action)
        self.assertFalse(result["safety_intervention"])
        self.assertGreaterEqual(result["minimum_separation"], cfg.min_uav_distance)

    def test_mpc_uses_observation_only(self):
        cfg = config()
        env = DualUAVEnv(cfg, seed=5)
        obs, _, info = env.reset(seed=5)
        policy = MPCController(cfg)
        first = policy.act(obs, info["active"])[0]
        env.target_state[:] = 999.0
        env.uav_positions[:] += 7.0
        second = policy.act(obs, info["active"])[0]
        np.testing.assert_array_equal(first, second)

    def test_mpc_prediction_matches_environment_move(self):
        cfg = config(
            uav_initial=np.array([[18., -5., 10.], [0., 5., 10.]]),
            uav_initial_velocities=np.array([[3., 0., 0.], [1., -1., 0.]]),
            uav_goal_positions=np.array([[0., -5., 10.], [12., 5., 10.]]))
        env = DualUAVEnv(cfg, seed=6)
        policy = MPCController(cfg)
        requested_acceleration = np.array([[5., 0., 0.], [0., -5., 0.]])
        raw, _ = acceleration_to_action(requested_acceleration, env.active, cfg)
        position, velocity, active = (env.uav_positions.copy(),
                                      env.uav_velocities.copy(), env.active.copy())
        predicted = policy._step(position[None], velocity[None],
                                 requested_acceleration[None], active[None],
                                 cfg.uav_goal_positions[None])
        env._move(raw)
        np.testing.assert_allclose(predicted[0][0], env.uav_positions, atol=1e-7)
        np.testing.assert_allclose(predicted[1][0], env.uav_velocities, atol=1e-7)
        self.assertEqual(bool(predicted[3][0]), True)

    def test_mpc_predicted_arrival_freezes_next_step_like_env(self):
        cfg = config(
            uav_goal_positions=np.array([[5., -2., 10.], [12., -2., 10.]]),
            min_uav_distance=2.)
        env = DualUAVEnv(cfg, seed=7)
        policy = MPCController(cfg)
        requested = np.array([[5., 0., 0.], [0., 0., 0.]])
        raw, _ = acceleration_to_action(requested, env.active, cfg)
        position = env.uav_positions[None].copy()
        velocity = env.uav_velocities[None].copy()
        active = env.active[None].copy()
        goals = cfg.uav_goal_positions[None]
        first = policy._step(position, velocity, requested[None], active, goals)
        env.step(raw)
        np.testing.assert_array_equal(first[2][0], env.active)
        second = policy._step(first[0], first[1], requested[None], first[2], goals)
        frozen = env.uav_positions[0].copy()
        env.step(raw)
        np.testing.assert_allclose(second[0][0], env.uav_positions)
        np.testing.assert_allclose(env.uav_positions[0], frozen)


if __name__ == "__main__":
    unittest.main()
