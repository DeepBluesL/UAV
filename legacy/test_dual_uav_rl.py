import unittest

import numpy as np

from dual_uav_rl import (
    DualUAVConfig,
    DualUAVISACEnv,
    QLearningAgent,
    rollout_greedy_policy,
    train_independent_q_learning,
)


class DualUAVRLEnvTests(unittest.TestCase):
    def test_reset_returns_two_discrete_local_states(self):
        env = DualUAVISACEnv(DualUAVConfig(max_steps=2), seed=1)

        states = env.reset()

        self.assertEqual(len(states), 2)
        self.assertEqual(len(states[0]), 7)
        self.assertTrue(all(isinstance(value, int) for value in states[0]))

    def test_step_moves_uav_and_returns_finite_paper_metrics(self):
        config = DualUAVConfig(max_steps=2)
        env = DualUAVISACEnv(config, seed=1)
        start = env.uav_positions.copy()

        _, reward, done, info = env.step([1, 0])  # UAV1 沿 +x，UAV2 悬停。

        np.testing.assert_allclose(
            info.uav_positions[0],
            start[0]
            + np.array(
                [config.max_uav_speed * config.slot_duration, 0.0, 0.0]
            ),
        )
        np.testing.assert_allclose(info.uav_positions[1], start[1])
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(np.isfinite(info.rho))
        self.assertEqual(info.sensing_sinrs.shape, (3,))
        self.assertEqual(info.communication_sinrs.shape, (2,))
        self.assertLessEqual(np.max(info.uav_speeds), config.max_uav_speed)
        self.assertFalse(done)

    def test_collision_action_is_rejected(self):
        initial = np.array([[30.0, 5.0, 70.0], [30.0, -5.0, 70.0]])
        config = DualUAVConfig(
            max_steps=1,
            uav_initial=initial,
            min_uav_distance=5.0,
        )
        env = DualUAVISACEnv(config, seed=1)

        # UAV1 沿 -y、UAV2 沿 +y，会落在同一点，因此环境拒绝联合动作。
        _, _, _, info = env.step([4, 3])

        self.assertTrue(info.collision)
        np.testing.assert_allclose(info.uav_positions, initial)

    def test_q_learning_update_matches_td_equation(self):
        agent = QLearningAgent(
            num_actions=3,
            learning_rate=0.5,
            discount_factor=0.9,
            seed=1,
        )
        state = (0,)
        next_state = (1,)
        agent.q_table[next_state][2] = 2.0

        agent.update(state, action=1, reward=1.0, next_state=next_state, done=False)

        # 0 + 0.5 * (1 + 0.9 * 2 - 0) = 1.4
        self.assertAlmostEqual(agent.q_table[state][1], 1.4)

    def test_short_training_populates_both_q_tables(self):
        env = DualUAVISACEnv(DualUAVConfig(max_steps=2), seed=2)

        agents, history = train_independent_q_learning(
            env, episodes=2, seed=2, verbose=False
        )

        self.assertEqual(len(history.episode_returns), 2)
        self.assertGreater(len(agents[0].q_table), 0)
        self.assertGreater(len(agents[1].q_table), 0)

        trace = rollout_greedy_policy(env, agents)
        self.assertEqual(trace.uav_positions.shape, (3, 2, 3))
        self.assertEqual(trace.uav_speeds.shape, (2, 2))
        self.assertTrue(np.all(trace.uav_speeds <= env.config.max_uav_speed))


if __name__ == "__main__":
    unittest.main()
