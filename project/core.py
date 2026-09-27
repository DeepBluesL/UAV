"""MAPPO 使用的轻量 PyTorch 网络组件。

网络结构沿用 OpenAI Spinning Up 的 PPO 实现，但环境接口固定为两个
同构智能体和连续三维动作，因此不依赖 Gym、MPI 或 SciPy。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .control import ActionAdapter


def combined_shape(length: int, shape=None) -> tuple[int, ...]:
    """构造经验缓存数组的形状。"""
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, int(shape))
    return (length, *shape)


def mlp(
    sizes: Iterable[int],
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    """创建全连接网络，最后一层默认不使用激活函数。"""
    sizes = list(sizes)
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layer_activation = activation if index < len(sizes) - 2 else output_activation
        layers.extend((nn.Linear(sizes[index], sizes[index + 1]), layer_activation()))
    return nn.Sequential(*layers)


def discount_cumsum(values: np.ndarray, discount: float) -> np.ndarray:
    """仅用 NumPy 反向计算折扣累计和。"""
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(values, dtype=np.float32)
    running = np.zeros(values.shape[1:], dtype=np.float32)
    for index in range(len(values) - 1, -1, -1):
        running = values[index] + discount * running
        result[index] = running
    return result


class Actor(nn.Module):
    """Spinning Up 风格的策略基类。"""

    def _distribution(self, obs: torch.Tensor):
        raise NotImplementedError

    def _log_prob_from_distribution(self, distribution, act: torch.Tensor):
        raise NotImplementedError

    def forward(self, obs: torch.Tensor, act: torch.Tensor | None = None):
        distribution = self._distribution(obs)
        logp = None
        if act is not None:
            logp = self._log_prob_from_distribution(distribution, act)
        return distribution, logp


class MLPGaussianActor(Actor):
    """均值由 MLP 给出、标准差可学习的高斯策略。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.mu_net = mlp([obs_dim, *hidden_sizes, act_dim], activation)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mu_net(obs), torch.exp(self.log_std))

    def _log_prob_from_distribution(
        self, distribution: Normal, act: torch.Tensor
    ) -> torch.Tensor:
        # 多维连续动作视为独立联合高斯，log-prob 对最后一维求和。
        return distribution.log_prob(act).sum(dim=-1)


class MLPCritic(nn.Module):
    """用全局状态估计共享团队价值。"""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Iterable[int],
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.v_net = mlp([state_dim, *hidden_sizes, 1], activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v_net(state).squeeze(-1)


class MAPPOActorCritic(nn.Module):
    """两个独立策略与一个共享状态价值网络。"""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        act_dim: int = 3,
        hidden_sizes: Iterable[int] = (128, 128),
        environment=None,
        control_mode: str = "pure",
        residual_scale: float = .25,
    ) -> None:
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        self.pi = nn.ModuleList(
            MLPGaussianActor(obs_dim, act_dim, hidden_sizes) for _ in range(2)
        )
        self.v = MLPCritic(state_dim, hidden_sizes)
        self.act_dim = act_dim
        self.action_adapter = (
            ActionAdapter(environment, control_mode, residual_scale)
            if environment is not None else None)
        self.control_mode = control_mode
        if control_mode == "residual":
            # Start exactly at the useful goal controller; PPO learns deviations.
            for actor in self.pi:
                output = actor.mu_net[-2]
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def step(self, obs, active, deterministic: bool = False):
        """Sample raw Gaussian actions retained unchanged for PPO log-probs."""
        if torch.is_tensor(active):
            active_flags = active.detach().cpu().numpy().astype(bool)
        else:
            active_flags = np.asarray(active, dtype=bool)
        if active_flags.shape != (2,):
            raise ValueError("active 必须是形状 (2,) 的布尔数组")

        actions = np.zeros((2, self.act_dim), dtype=np.float32)
        logps = np.zeros(2, dtype=np.float32)
        with torch.no_grad():
            for agent_index in range(2):
                if not active_flags[agent_index]:
                    continue
                # 先按掩码取行，保证非活动行中的 NaN 从不进入策略网络。
                agent_obs = torch.as_tensor(
                    obs[agent_index], dtype=torch.float32, device=self.device
                )
                distribution = self.pi[agent_index]._distribution(agent_obs)
                action = distribution.mean if deterministic else distribution.sample()
                logp = self.pi[agent_index]._log_prob_from_distribution(
                    distribution, action
                )
                actions[agent_index] = action.cpu().numpy()
                logps[agent_index] = float(logp.item())
        return actions, logps

    def act(self, obs, active, deterministic: bool = False):
        """Return actions in the environment's raw-action coordinate system."""
        raw, logps = self.step(obs, active, deterministic)
        if self.action_adapter is None:
            return raw, logps
        return self.action_adapter.policy_action(raw, obs, active), logps

    def set_environment(self, environment):
        """Use scenario-specific dynamics and observation scales for action mapping."""
        if self.action_adapter is not None:
            self.action_adapter.set_environment(environment)

    def value(self, state) -> float:
        """在模型当前设备上计算单个全局状态的价值。"""
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            )
            return float(self.v(state_tensor).item())

    state_value = value
