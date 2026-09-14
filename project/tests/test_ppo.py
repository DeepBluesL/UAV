"""MAPPO 核心算法的快速单元测试。"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.core import MAPPOActorCritic
from project.ppo import MAPPOBuffer, PPOTrainer


def small_config(**overrides):
    values = dict(
        device="cpu",
        pi_lr=3e-3,
        vf_lr=3e-3,
        clip_ratio=0.2,
        train_pi_iters=2,
        train_v_iters=2,
        target_kl=0.1,
        entropy_coef=0.0,
        max_grad_norm=0.5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_training_data(ac, active):
    """生成带 NaN 非活动行的自洽旧策略数据。"""
    count = len(active)
    obs = torch.randn(count, 2, 3)
    active = torch.as_tensor(active, dtype=torch.bool)
    obs[~active] = torch.nan
    actions = torch.zeros(count, 2, 2)
    logp = torch.zeros(count, 2)
    with torch.no_grad():
        for agent_index in range(2):
            mask = active[:, agent_index]
            if not mask.any():
                continue
            distribution = ac.pi[agent_index]._distribution(obs[mask, agent_index])
            actions[mask, agent_index] = distribution.sample()
            logp[mask, agent_index] = ac.pi[
                agent_index
            ]._log_prob_from_distribution(
                distribution, actions[mask, agent_index]
            )
    return {
        "obs": obs,
        "state": torch.randn(count, 5),
        "act": actions,
        "ret": torch.linspace(-1.0, 1.0, count),
        "adv": torch.tensor([-1.0, 0.5, 2.0, -0.4])[:count],
        "logp": logp,
        "active": active,
    }


class ActorCriticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        np.random.seed(3)

    def test_raw_action_logp_ratio_is_initially_one(self):
        ac = MAPPOActorCritic(3, 5, act_dim=2, hidden_sizes=(8,))
        obs = torch.randn(6, 3)
        distribution = ac.pi[0]._distribution(obs)
        actions = distribution.sample()
        old_logp = ac.pi[0]._log_prob_from_distribution(distribution, actions)
        _, new_logp = ac.pi[0](obs, actions)
        np.testing.assert_allclose(
            torch.exp(new_logp - old_logp).detach().numpy(), np.ones(6)
        )

    def test_inactive_actor_is_never_called(self):
        ac = MAPPOActorCritic(3, 5, act_dim=2, hidden_sizes=(8,))
        obs = np.array([[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan]])
        with mock.patch.object(
            ac.pi[1], "_distribution", side_effect=AssertionError("被错误调用")
        ):
            actions, logp = ac.act(obs, [True, False], deterministic=True)
        np.testing.assert_array_equal(actions[1], np.zeros(2))
        self.assertEqual(logp[1], 0.0)
        self.assertTrue(np.isfinite(actions[0]).all())

    def test_value_uses_only_critic(self):
        ac = MAPPOActorCritic(3, 5, act_dim=2, hidden_sizes=(8,))
        with mock.patch.object(
            ac.pi[0], "forward", side_effect=AssertionError("策略不应参与 value")
        ):
            value = ac.value(np.zeros(5, dtype=np.float32))
        self.assertIsInstance(value, float)


class BufferTests(unittest.TestCase):
    def _store_path(self, buffer, rewards, values=None, active=None):
        values = np.zeros(len(rewards)) if values is None else values
        active = active or [[True, True]] * len(rewards)
        for reward, value, flags in zip(rewards, values, active):
            buffer.store(
                np.zeros((2, 2)), np.zeros(3), np.zeros((2, 1)),
                reward, value, np.zeros(2), flags,
            )

    def test_agent_arrival_does_not_truncate_team_gae(self):
        buffer = MAPPOBuffer(2, 3, 1, size=3, gamma=1.0, lam=1.0)
        self._store_path(
            buffer,
            [1.0, 2.0, 3.0],
            active=[[True, True], [True, False], [True, False]],
        )
        buffer.finish_path(0.0)
        np.testing.assert_allclose(buffer.adv_buf, [6.0, 5.0, 3.0])

    def test_bootstrap_changes_gae_and_returns(self):
        terminal = MAPPOBuffer(2, 3, 1, size=2, gamma=0.9, lam=0.95)
        cutoff = MAPPOBuffer(2, 3, 1, size=2, gamma=0.9, lam=0.95)
        self._store_path(terminal, [1.0, 1.0])
        self._store_path(cutoff, [1.0, 1.0])
        terminal.finish_path(0.0)
        cutoff.finish_path(4.0)
        self.assertFalse(np.allclose(terminal.adv_buf, cutoff.adv_buf))
        self.assertFalse(np.allclose(terminal.ret_buf, cutoff.ret_buf))
        np.testing.assert_allclose(cutoff.ret_buf, [5.14, 4.6], rtol=1e-6)

    def test_return_is_discounted_reward_to_go(self):
        buffer = MAPPOBuffer(2, 3, 1, size=2, gamma=0.9, lam=0.5)
        self._store_path(buffer, [1.0, 2.0], values=[5.0, 6.0])
        buffer.finish_path(0.0)
        np.testing.assert_allclose(buffer.ret_buf, [2.8, 2.0], rtol=1e-6)
        self.assertFalse(np.allclose(buffer.ret_buf, buffer.adv_buf + buffer.val_buf))

    def test_get_keeps_raw_advantage_and_bool_mask(self):
        buffer = MAPPOBuffer(2, 3, 1, size=2, gamma=1.0, lam=1.0)
        self._store_path(buffer, [1.0, 3.0])
        buffer.finish_path()
        data = buffer.get()
        np.testing.assert_allclose(data["adv"].numpy(), [4.0, 3.0])
        self.assertEqual(data["active"].dtype, torch.bool)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.ac = MAPPOActorCritic(3, 5, act_dim=2, hidden_sizes=(8,))

    def test_masked_update_filters_nan_placeholders(self):
        active = [[True, False], [False, True], [True, True], [False, False]]
        data = make_training_data(self.ac, active)
        stats = PPOTrainer(self.ac, small_config()).update(data)
        self.assertEqual(stats["valid_samples_0"], 2)
        self.assertEqual(stats["valid_samples_1"], 2)
        for name, value in stats.items():
            self.assertTrue(np.isfinite(value), name)

    def test_actor_with_no_valid_samples_is_unchanged(self):
        active = [[True, False]] * 4
        data = make_training_data(self.ac, active)
        trainer = PPOTrainer(self.ac, small_config())
        before = [parameter.detach().clone() for parameter in self.ac.pi[1].parameters()]
        with mock.patch.object(
            trainer.pi_optimizers[1], "step", wraps=trainer.pi_optimizers[1].step
        ) as step:
            stats = trainer.update(data)
        step.assert_not_called()
        self.assertEqual(stats["valid_samples_1"], 0)
        for old, new in zip(before, self.ac.pi[1].parameters()):
            torch.testing.assert_close(old, new)

    def test_critic_receives_every_team_timestep(self):
        data = make_training_data(self.ac, [[False, False]] * 4)
        trainer = PPOTrainer(self.ac, small_config())
        with mock.patch.object(
            self.ac.v, "forward", wraps=self.ac.v.forward
        ) as critic_forward:
            stats = trainer.update(data)
        self.assertGreaterEqual(critic_forward.call_count, 2)
        for call in critic_forward.call_args_list:
            self.assertEqual(call.args[0].shape[0], 4)
        self.assertTrue(np.isfinite(stats["value_loss"]))


if __name__ == "__main__":
    unittest.main()
