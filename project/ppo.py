"""双智能体 MAPPO 的经验缓存与 PPO 更新器。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .core import MAPPOActorCritic, combined_shape, discount_cumsum


class MAPPOBuffer:
    """保存一条共享团队轨迹，并计算团队 GAE 与 reward-to-go。"""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        self.obs_buf = np.zeros((size, 2, obs_dim), dtype=np.float32)
        self.state_buf = np.zeros(combined_shape(size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, 2, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros((size, 2), dtype=np.float32)
        self.active_buf = np.zeros((size, 2), dtype=bool)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, state, act, rew, val, logp, active) -> None:
        """追加一个团队时间步。"""
        if self.ptr >= self.max_size:
            raise IndexError("MAPPOBuffer 已满")
        self.obs_buf[self.ptr] = obs
        self.state_buf[self.ptr] = state
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.active_buf[self.ptr] = active
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        """结束团队轨迹；个体退出不会切断团队奖励传播。"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rew_buf[path_slice], np.float32(last_val))
        values = np.append(self.val_buf[path_slice], np.float32(last_val))
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # value target 按参考 PPO 直接使用折扣 reward-to-go，而不是 A + V。
        self.ret_buf[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
        """返回完整缓存；优势留待每个策略按有效样本分别标准化。"""
        if self.ptr != self.max_size:
            raise RuntimeError("必须填满 MAPPOBuffer 后才能 get")
        data = {
            "obs": self.obs_buf,
            "state": self.state_buf,
            "act": self.act_buf,
            "rew": self.rew_buf,
            "ret": self.ret_buf,
            "adv": self.adv_buf,
            "val": self.val_buf,
            "logp": self.logp_buf,
            "active": self.active_buf,
        }
        self.ptr, self.path_start_idx = 0, 0
        return {
            key: torch.as_tensor(value, dtype=torch.bool if key == "active" else torch.float32)
            for key, value in data.items()
        }


class PPOTrainer:
    """分别优化两个策略，并用所有团队时间步训练共享 critic。"""

    def __init__(self, ac: MAPPOActorCritic, config: Any) -> None:
        self.ac = ac
        self.config = config
        self.device = torch.device(config.device)
        self.ac.to(self.device)
        self.pi_optimizers = [
            Adam(actor.parameters(), lr=config.pi_lr) for actor in self.ac.pi
        ]
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=config.vf_lr)

    def _policy_loss(
        self, data: dict[str, torch.Tensor], agent_index: int
    ) -> tuple[torch.Tensor, dict[str, float]] | None:
        active = data["active"][:, agent_index]
        valid_samples = int(active.sum().item())
        if valid_samples == 0:
            return None

        # 必须先筛选再送入网络，避免 NaN 占位污染前向传播和梯度。
        obs = data["obs"][active, agent_index]
        act = data["act"][active, agent_index]
        old_logp = data["logp"][active, agent_index]
        advantage = data["adv"][active]
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )

        distribution, logp = self.ac.pi[agent_index](obs, act)
        ratio = torch.exp(logp - old_logp)
        clipped_advantage = torch.clamp(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        ) * advantage
        entropy = distribution.entropy().sum(dim=-1).mean()
        loss = -torch.minimum(ratio * advantage, clipped_advantage).mean()
        loss = loss - self.config.entropy_coef * entropy
        clipped = (ratio > 1.0 + self.config.clip_ratio) | (
            ratio < 1.0 - self.config.clip_ratio
        )
        info = {
            "kl": float((old_logp - logp).mean().detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "clipfrac": float(clipped.float().mean().detach().cpu()),
            "valid_samples": valid_samples,
        }
        return loss, info

    def _value_loss(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        return ((self.ac.v(data["state"]) - data["ret"]) ** 2).mean()

    def update(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        """执行一次批量更新，返回便于记录的扁平统计字典。"""
        data = {key: value.to(self.device) for key, value in data.items()}
        stats: dict[str, float] = {}

        for agent_index, optimizer in enumerate(self.pi_optimizers):
            valid_samples = int(data["active"][:, agent_index].sum().item())
            if valid_samples == 0:
                stats.update(
                    {
                        f"pi_loss_{agent_index}": 0.0,
                        f"kl_{agent_index}": 0.0,
                        f"entropy_{agent_index}": 0.0,
                        f"clipfrac_{agent_index}": 0.0,
                        f"valid_samples_{agent_index}": 0,
                    }
                )
                continue

            for _ in range(self.config.train_pi_iters):
                result = self._policy_loss(data, agent_index)
                assert result is not None
                loss, info = result
                if info["kl"] > 1.5 * self.config.target_kl:
                    break
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.pi[agent_index].parameters(), self.config.max_grad_norm
                )
                optimizer.step()

            final_result = self._policy_loss(data, agent_index)
            assert final_result is not None
            final_loss, final_info = final_result
            stats[f"pi_loss_{agent_index}"] = float(final_loss.detach().cpu())
            for name, value in final_info.items():
                stats[f"{name}_{agent_index}"] = value

        for _ in range(self.config.train_v_iters):
            self.vf_optimizer.zero_grad()
            value_loss = self._value_loss(data)
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.ac.v.parameters(), self.config.max_grad_norm)
            self.vf_optimizer.step()
        stats["value_loss"] = float(self._value_loss(data).detach().cpu())
        return stats
