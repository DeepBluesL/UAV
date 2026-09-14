"""采样边界和模型保存的集成测试，使用短确定性场景而不是长训练。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from project.artifacts import load_policy, save_policy
from project.config import EnvConfig, PPOConfig, RewardConfig
from project.core import MAPPOActorCritic
from project.env import DualUAVEnv
from project.evaluate import GoalController, evaluate
from project.metrics import EpisodeRecorder
from project.ppo import MAPPOBuffer
from project.train import collect_epoch


class ValuedController(GoalController):
    def __init__(self, config):
        super().__init__(config)
        self.value_calls = 0

    def value(self, state):
        self.value_calls += 1
        return 10.


def scenario(max_steps=10):
    return EnvConfig(
        max_steps=max_steps, goal_tolerance=.1,
        uav_initial=np.array([[30., 10., 70.], [30., -10., 70.]]),
        uav_goal_positions=np.array([[35., 10., 70.], [45., -10., 70.]]),
        imperfect_csi_beta=0., csi_beta_jitter=0.)


class TrainingIntegrationTests(unittest.TestCase):
    def test_cutoff_bootstraps_and_keeps_partially_finished_team(self):
        cfg = scenario()
        env = DualUAVEnv(cfg)
        obs, state, info = env.reset(seed=11)
        policy = ValuedController(cfg)
        buffer = MAPPOBuffer(31, 58, 3, 1, gamma=.9)
        obs, state, recorder, episodes, _ = collect_epoch(
            env, policy, buffer, obs, state, EpisodeRecorder(info))
        self.assertEqual(env.step_count, 1)  # 未因批次结束 reset。
        np.testing.assert_array_equal(env.active, [False, True])
        np.testing.assert_array_equal(buffer.active_buf[0], [True, True])
        self.assertEqual(episodes, [])
        self.assertEqual(recorder.steps, 1)
        self.assertAlmostEqual(buffer.ret_buf[0], buffer.rew_buf[0] + 9., places=5)
        self.assertEqual(policy.value_calls, 2)  # 当前 V 与末端 bootstrap V。
        buffer.get()

        # 第二批不再给首机生成有效样本，仍继续同一团队回合。
        _, _, recorder, episodes, _ = collect_epoch(env, policy, buffer, obs, state, recorder)
        self.assertEqual(env.step_count, 2)
        np.testing.assert_array_equal(buffer.active_buf[0], [False, True])
        self.assertEqual(recorder.steps, 2)
        self.assertFalse(episodes)

    def test_deadline_coinciding_with_cutoff_has_zero_bootstrap(self):
        cfg = scenario(max_steps=1)
        env = DualUAVEnv(cfg)
        obs, state, info = env.reset(seed=11)
        policy = ValuedController(cfg)
        buffer = MAPPOBuffer(31, 58, 3, 1, gamma=.9)
        _, _, recorder, episodes, _ = collect_epoch(
            env, policy, buffer, obs, state, EpisodeRecorder(info))
        self.assertEqual(policy.value_calls, 1)
        self.assertEqual(buffer.ret_buf[0], buffer.rew_buf[0])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["termination_reason"], "deadline")
        self.assertEqual(episodes[0]["team_success"], 0)
        self.assertEqual(recorder.steps, 0)  # 真结束后，准备新回合。
        self.assertEqual(env.step_count, 0)

    def test_checkpoint_roundtrip_and_frozen_evaluation_are_reproducible(self):
        torch.manual_seed(2)
        cfg, reward, ppo = scenario(), RewardConfig(), PPOConfig(hidden_sizes=(8, 8))
        ac = MAPPOActorCritic(31, 58, hidden_sizes=ppo.hidden_sizes)
        parameters = [value.detach().clone() for value in ac.parameters()]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "policy.pt"
            save_policy(path, ac, cfg, reward, ppo)
            loaded, loaded_env, loaded_reward, _ = load_policy(path)
            before, _, trace_before = evaluate(ac, cfg, reward, [21, 22])
            after, _, trace_after = evaluate(loaded, loaded_env, loaded_reward, [21, 22])
        self.assertEqual(before, after)
        np.testing.assert_array_equal(trace_before["positions"], trace_after["positions"])
        for parameter, original in zip(ac.parameters(), parameters):
            torch.testing.assert_close(parameter, original)

    def test_goal_controller_confirms_full_episode_success(self):
        cfg = scenario()
        summary, rows, _ = evaluate(GoalController(cfg), cfg, RewardConfig(), [1, 2])
        self.assertEqual(summary["team_success_rate"], 1.)
        self.assertEqual(summary["mean_collisions"], 0.)
        self.assertLess(rows[0]["arrival_time_0"], rows[0]["arrival_time_1"])


if __name__ == "__main__":
    unittest.main()