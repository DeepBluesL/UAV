"""面向初学者的双 UAV 协作强化学习示例。

这个文件把仓库 ``test.py`` 中已经复现的论文公式包装成一个小型 RL 环境：

* Eq. (6)：基站感知 SINR；
* Eq. (10)：两架 UAV 的通信 SINR；
* Eq. (12)：两架 UAV 的感知 SINR；
* Eq. (13)-(16)：CRB 与信息融合；
* Eq. (17)-(28)：目标运动与 PCRB。

为避免一开始就引入 PyTorch、MADDPG 或 MAPPO，本例只学习 UAV 的离散轨迹，
波束仍使用 ``test.py`` 中的匹配波束基线。两个 UAV 各自维护一张 Q 表，但使用同一个
团队奖励，这种入门方法通常称为 Independent Q-Learning (IQL)。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np

# 直接复用当前论文复现代码，而不是另外写一套与论文无关的距离奖励环境。
from test import (
    bs_sensing_sinr_eq6_parts,
    ci_fusion_crb,
    crb_from_sensing_sinr,
    db_to_linear,
    dbm_to_watt,
    make_matched_beams,
    measurement_noise_cov_from_fused_crb,
    pcrb_eq24_to_eq28_parts,
    safe_inverse,
    state_transition_matrix,
    uav_comm_sinr_eq10_parts,
    uav_sensing_sinr_eq12_parts,
)


# 每个 UAV 有 7 个离散动作。动作 0 表示悬停，其余动作沿一个坐标轴移动。
# 真实论文中的连续速度可在后续用 MADDPG/MAPPO 替代；离散动作更适合先理解 RL 闭环。
ACTION_DELTAS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)

ACTION_NAMES = (
    "hover",
    "+x",
    "-x",
    "+y",
    "-y",
    "+z",
    "-z",
)

# 离散状态由若干整数桶组成，使用 tuple 后可以直接作为 Q 表的字典键。
DiscreteState = Tuple[int, ...]


@dataclass
class DualUAVConfig:
    """双 UAV 环境参数；默认值尽量与 ``test.py`` 的仿真设置一致。"""

    max_steps: int = 30
    slot_duration: float = 1.0
    # 当前复现脚本 test.py 使用 5 m/s。每时隙最大位移严格等于 v_max * dt。
    max_uav_speed: float = 5.0
    min_uav_distance: float = 5.0

    bs_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    uav_initial: np.ndarray = field(
        default_factory=lambda: np.array(
            [[30.0, 10.0, 70.0], [30.0, -10.0, 70.0]], dtype=float
        )
    )
    target_initial_state: np.ndarray = field(
        default_factory=lambda: np.array(
            [60.0, 0.0, 60.0, 0.5, 0.1, 0.0], dtype=float
        )
    )
    world_low: np.ndarray = field(
        default_factory=lambda: np.array([-20.0, -80.0, 20.0], dtype=float)
    )
    world_high: np.ndarray = field(
        default_factory=lambda: np.array([150.0, 80.0, 120.0], dtype=float)
    )

    # 论文通信/感知参数。
    antenna_x: int = 8
    antenna_y: int = 8
    total_power: float = 5.0
    bandwidth: float = 100.0e6
    rician_k: float = 1.0e6  # 近似 LOS，使教学训练稳定且仍保留极小衰落。
    beta0_c: float = field(default_factory=lambda: db_to_linear(-50.0))
    beta0_s: float = field(default_factory=lambda: db_to_linear(-50.0))
    alpha_target: float = 0.9
    alpha_uav: float = 0.9
    awgn_power: float = field(default_factory=lambda: dbm_to_watt(-80.0))
    residual_noise: float = field(default_factory=lambda: dbm_to_watt(-70.0))
    gamma_min: float = field(default_factory=lambda: 10.0 ** (5.0 / 10.0))

    kappa_d: float = 1.0
    kappa_theta: float = 1.0e-6
    kappa_phi: float = 1.0e-6
    kappa_v: float = 1.0e-6
    process_noise_intensity: float = 0.1
    initial_covariance: float = 1.0
    propulsion_energy_weight: float = 0.01

    # 状态离散化阈值：误差小于 near 时记为 0，大于 far 时记为 +/-2。
    state_near: float = 5.0
    state_far: float = 20.0

    def __post_init__(self) -> None:
        self.bs_position = np.asarray(self.bs_position, dtype=float)
        self.uav_initial = np.asarray(self.uav_initial, dtype=float)
        self.target_initial_state = np.asarray(self.target_initial_state, dtype=float)
        self.world_low = np.asarray(self.world_low, dtype=float)
        self.world_high = np.asarray(self.world_high, dtype=float)

        if self.uav_initial.shape != (2, 3):
            raise ValueError("uav_initial 必须是形状为 (2, 3) 的数组。")
        if self.target_initial_state.shape != (6,):
            raise ValueError("target_initial_state 必须为 [x,y,z,vx,vy,vz]。")
        if np.any(self.world_low >= self.world_high):
            raise ValueError("world_low 必须逐元素小于 world_high。")
        if self.slot_duration <= 0.0 or self.max_uav_speed <= 0.0:
            raise ValueError("slot_duration 和 max_uav_speed 必须为正数。")


@dataclass(frozen=True)
class DualUAVStepInfo:
    """一步仿真的详细物理指标，便于训练后画图或调奖励。"""

    t: int
    uav_positions: np.ndarray
    uav_velocities: np.ndarray
    uav_speeds: np.ndarray
    target_state: np.ndarray
    communication_sinrs: np.ndarray
    sensing_sinrs: np.ndarray  # 顺序为 [BS, UAV1, UAV2]
    fused_crb: np.ndarray
    rho: float
    mean_target_distance: float
    uav_distance: float
    collision: bool
    boundary_violations: int
    communication_ok: np.ndarray
    propulsion_energy_proxy: float
    reward_parts: Dict[str, float]


class DualUAVISACEnv:
    """不依赖 Gym 的最小双 UAV RL 环境。

    API 与常见环境相似：

    ``states = env.reset()``
    ``next_states, reward, done, info = env.step([a1, a2])``

    两个局部状态分别交给两个智能体；奖励为共享标量，鼓励协作。
    """

    num_agents = 2
    num_actions = len(ACTION_DELTAS)

    def __init__(self, config: DualUAVConfig | None = None, seed: int = 0):
        self.config = config or DualUAVConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> Tuple[DiscreteState, DiscreteState]:
        """恢复初始位置、目标状态和上一时隙的 Fisher 信息矩阵。"""

        self.t = 0
        self.uav_positions = self.config.uav_initial.copy()
        self.target_state = self.config.target_initial_state.copy()

        p0 = self.config.initial_covariance * np.eye(6, dtype=float)
        self.j_prev = safe_inverse(p0)

        initial_metrics = self._paper_link_metrics(self.target_state[:3])
        self.last_communication_sinrs = initial_metrics["communication_sinrs"]
        self.last_rho = float(np.trace(p0))
        return self._observations()

    def step(
        self, joint_actions: Sequence[int]
    ) -> Tuple[Tuple[DiscreteState, DiscreteState], float, bool, DualUAVStepInfo]:
        """执行两架 UAV 的联合动作并推进一个论文时隙。"""

        actions = np.asarray(joint_actions, dtype=int)
        if actions.shape != (2,):
            raise ValueError("joint_actions 必须包含 UAV1 和 UAV2 的两个动作。")
        if np.any(actions < 0) or np.any(actions >= self.num_actions):
            raise ValueError(f"动作编号必须在 [0, {self.num_actions - 1}] 内。")

        # 1) 轨迹动作：最大位移 = 最大速度 * 时隙长度，边界通过 clip 保证。
        previous_positions = self.uav_positions.copy()
        max_displacement = self.config.max_uav_speed * self.config.slot_duration
        requested_positions = (
            self.uav_positions + ACTION_DELTAS[actions] * max_displacement
        )
        next_positions = np.clip(
            requested_positions, self.config.world_low, self.config.world_high
        )
        boundary_violations = int(
            np.count_nonzero(np.any(next_positions != requested_positions, axis=1))
        )

        # 2) 防碰撞约束：若联合动作会使间距小于 d_min，则两架 UAV 都保持原位。
        candidate_distance = float(np.linalg.norm(next_positions[0] - next_positions[1]))
        collision = candidate_distance < self.config.min_uav_distance
        if collision:
            next_positions = self.uav_positions.copy()
        self.uav_positions = next_positions
        uav_velocities = (
            self.uav_positions - previous_positions
        ) / self.config.slot_duration
        uav_speeds = np.linalg.norm(uav_velocities, axis=1)
        # 教学用推进能耗代理：速度平方归一化。它不是特定机型的电池功率模型。
        propulsion_energy_proxy = float(
            np.sum((uav_speeds / self.config.max_uav_speed) ** 2)
        )

        # 3) 先按常速度模型预测本时隙目标，再在新几何位置上计算论文指标。
        target_state_before_step = self.target_state.copy()
        f = state_transition_matrix(self.config.slot_duration)
        predicted_target = f @ target_state_before_step
        metrics = self._paper_link_metrics(predicted_target[:3])

        # 4) Eq. (14)-(16)：由 BS、UAV1、UAV2 的感知 SINR 得到融合 CRB。
        crbs = [
            crb_from_sensing_sinr(
                sensing_sinr=sinr,
                bandwidth=self.config.bandwidth,
                kappa_d=self.config.kappa_d,
                kappa_theta=self.config.kappa_theta,
                kappa_phi=self.config.kappa_phi,
                kappa_v=self.config.kappa_v,
            )
            for sinr in metrics["sensing_sinrs"]
        ]
        fused_crb = ci_fusion_crb(crbs)
        psi = measurement_noise_cov_from_fused_crb(fused_crb)

        # 5) Eq. (17)-(28)：更新 PCRB，rho=trace(PCRB) 越小表示跟踪下界越好。
        pcrb_parts = pcrb_eq24_to_eq28_parts(
            zeta_prev=target_state_before_step,
            J_prev=self.j_prev,
            Psi=psi,
            dt=self.config.slot_duration,
            sigma2_zeta=self.config.process_noise_intensity,
            use_inverse_covariance=True,
        )
        self.target_state = pcrb_parts["zeta_pred"]
        self.j_prev = pcrb_parts["J"]
        rho = float(pcrb_parts["rho"])

        communication_sinrs = metrics["communication_sinrs"]
        sensing_sinrs = metrics["sensing_sinrs"]
        communication_ok = communication_sinrs >= self.config.gamma_min
        target_distances = np.linalg.norm(
            self.uav_positions - self.target_state[:3], axis=1
        )
        mean_target_distance = float(np.mean(target_distances))
        uav_distance = float(np.linalg.norm(self.uav_positions[0] - self.uav_positions[1]))

        reward_parts = self._reward(
            mean_target_distance=mean_target_distance,
            rho=rho,
            sensing_sinrs=sensing_sinrs,
            communication_ok=communication_ok,
            collision=collision,
            boundary_violations=boundary_violations,
            propulsion_energy_proxy=propulsion_energy_proxy,
            propulsion_energy_weight=self.config.propulsion_energy_weight,
        )
        reward = float(sum(reward_parts.values()))

        self.t += 1
        self.last_communication_sinrs = communication_sinrs
        self.last_rho = rho
        done = self.t >= self.config.max_steps

        info = DualUAVStepInfo(
            t=self.t,
            uav_positions=self.uav_positions.copy(),
            uav_velocities=uav_velocities.copy(),
            uav_speeds=uav_speeds.copy(),
            target_state=self.target_state.copy(),
            communication_sinrs=communication_sinrs.copy(),
            sensing_sinrs=sensing_sinrs.copy(),
            fused_crb=fused_crb.copy(),
            rho=rho,
            mean_target_distance=mean_target_distance,
            uav_distance=uav_distance,
            collision=collision,
            boundary_violations=boundary_violations,
            communication_ok=communication_ok.copy(),
            propulsion_energy_proxy=propulsion_energy_proxy,
            reward_parts=reward_parts,
        )
        return self._observations(), reward, done, info

    def _paper_link_metrics(self, target_position: np.ndarray) -> Dict[str, np.ndarray]:
        """调用现有论文公式，计算一次双 UAV 的通信与感知链路。"""

        cfg = self.config
        w0, w_comm = make_matched_beams(
            u_bs=cfg.bs_position,
            uav_positions=self.uav_positions,
            target=target_position,
            Mx=cfg.antenna_x,
            My=cfg.antenna_y,
            P=cfg.total_power,
        )

        bs_sensing = bs_sensing_sinr_eq6_parts(
            u_bs=cfg.bs_position,
            target=target_position,
            W_comm=w_comm,
            w0=w0,
            Mx=cfg.antenna_x,
            My=cfg.antenna_y,
            beta0_s=cfg.beta0_s,
            K=cfg.rician_k,
            alpha0=cfg.alpha_target,
            sigma_ubs=cfg.residual_noise,
            sigma0=cfg.awgn_power,
            rng=self.rng,
        )["sinr"]

        communication_sinrs = np.array(
            [
                uav_comm_sinr_eq10_parts(
                    u_bs=cfg.bs_position,
                    uav_positions=self.uav_positions,
                    target=target_position,
                    n_idx=agent_id,
                    W_comm=w_comm,
                    w0=w0,
                    Mx=cfg.antenna_x,
                    My=cfg.antenna_y,
                    beta0_c=cfg.beta0_c,
                    beta0_s=cfg.beta0_s,
                    K=cfg.rician_k,
                    alpha0=cfg.alpha_target,
                    alpha1=cfg.alpha_uav,
                    sigma0=cfg.awgn_power,
                    rng=self.rng,
                    include_direct_sensing_beam=False,
                )["sinr"]
                for agent_id in range(self.num_agents)
            ],
            dtype=float,
        )

        uav_sensing = np.array(
            [
                uav_sensing_sinr_eq12_parts(
                    u_bs=cfg.bs_position,
                    uav=uav_position,
                    target=target_position,
                    W_comm=w_comm,
                    w0=w0,
                    Mx=cfg.antenna_x,
                    My=cfg.antenna_y,
                    beta0_s=cfg.beta0_s,
                    K=cfg.rician_k,
                    alpha0=cfg.alpha_target,
                    sigma_un=cfg.residual_noise,
                    sigma0=cfg.awgn_power,
                    rng=self.rng,
                )["sinr"]
                for uav_position in self.uav_positions
            ],
            dtype=float,
        )

        return {
            "communication_sinrs": communication_sinrs,
            "sensing_sinrs": np.concatenate(([bs_sensing], uav_sensing)),
        }

    def _observations(self) -> Tuple[DiscreteState, DiscreteState]:
        """构造去中心化局部状态：目标相对位置、队友方向和通信约束。"""

        states: List[DiscreteState] = []
        for agent_id in range(self.num_agents):
            other_id = 1 - agent_id
            relative_target = self.target_state[:3] - self.uav_positions[agent_id]
            relative_teammate = self.uav_positions[other_id] - self.uav_positions[agent_id]

            # 目标相对位置使用 5 桶，队友方向只使用 3 桶，控制 Q 表规模。
            target_buckets = tuple(self._bucket_5(value) for value in relative_target)
            teammate_buckets = tuple(self._bucket_3(value) for value in relative_teammate)
            comm_ok = int(
                self.last_communication_sinrs[agent_id] >= self.config.gamma_min
            )
            states.append(target_buckets + teammate_buckets + (comm_ok,))
        return states[0], states[1]

    def _bucket_5(self, value: float) -> int:
        if value < -self.config.state_far:
            return -2
        if value < -self.config.state_near:
            return -1
        if value <= self.config.state_near:
            return 0
        if value <= self.config.state_far:
            return 1
        return 2

    def _bucket_3(self, value: float) -> int:
        if value < -self.config.state_near:
            return -1
        if value > self.config.state_near:
            return 1
        return 0

    @staticmethod
    def _reward(
        mean_target_distance: float,
        rho: float,
        sensing_sinrs: np.ndarray,
        communication_ok: np.ndarray,
        collision: bool,
        boundary_violations: int,
        propulsion_energy_proxy: float,
        propulsion_energy_weight: float,
    ) -> Dict[str, float]:
        """团队奖励：距离/PCRB/感知收益与通信/安全约束的加权和。

        各分量分开返回，便于看到智能体为什么得到当前奖励。真实复现实验中应对这些
        权重做消融或采用拉格朗日约束；这里的数值只服务于一个可学的教学示例。
        """

        return {
            "tracking": -mean_target_distance / 100.0,
            "pcrb": -0.10 * float(np.log1p(max(rho, 0.0))),
            "sensing": 0.10 * float(np.tanh(np.mean(sensing_sinrs))),
            "communication": -0.75 * float(np.count_nonzero(~communication_ok)),
            "energy": -propulsion_energy_weight * propulsion_energy_proxy,
            "collision": -2.0 if collision else 0.0,
            "boundary": -0.5 * float(boundary_violations),
        }


class QLearningAgent:
    """一张稀疏 Q 表对应一架 UAV。

    Q(s,a) 表示在局部状态 s 执行动作 a 后，未来累计团队奖励的估计值。
    """

    def __init__(
        self,
        num_actions: int,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.rng = np.random.default_rng(seed)
        self.q_table: DefaultDict[DiscreteState, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=float)
        )

    def choose_action(self, state: DiscreteState, epsilon: float) -> int:
        """epsilon-greedy：以 epsilon 随机探索，否则选择当前最优动作。"""

        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_actions))

        q_values = self.q_table[state]
        # 随机打破并列，避免初始 Q 全为 0 时总选择悬停。
        best_actions = np.flatnonzero(np.isclose(q_values, np.max(q_values)))
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        done: bool,
    ) -> None:
        """标准 Q-learning 更新：Q <- Q + alpha * (TD target - Q)。"""

        current_q = self.q_table[state][action]
        bootstrap = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.discount_factor * bootstrap
        self.q_table[state][action] += self.learning_rate * (td_target - current_q)


@dataclass
class TrainingHistory:
    episode_returns: List[float]
    final_rhos: List[float]
    final_distances: List[float]
    communication_success_rates: List[float]
    episode_energy_proxies: List[float]


@dataclass
class EpisodeTrace:
    """一个贪心回合的完整轨迹，供三维场景和时序指标绘图。"""

    times: np.ndarray
    uav_positions: np.ndarray
    target_positions: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    rhos: np.ndarray
    mean_target_distances: np.ndarray
    uav_distances: np.ndarray
    communication_sinrs: np.ndarray
    sensing_sinrs: np.ndarray
    uav_speeds: np.ndarray
    energy_proxies: np.ndarray
    reward_parts: Dict[str, np.ndarray]


def train_independent_q_learning(
    env: DualUAVISACEnv,
    episodes: int = 300,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[List[QLearningAgent], TrainingHistory]:
    """训练两个 IQL 智能体，并返回智能体和学习曲线数据。"""

    if episodes <= 0:
        raise ValueError("episodes 必须为正整数。")

    agents = [
        QLearningAgent(env.num_actions, seed=seed + 100 + agent_id)
        for agent_id in range(env.num_agents)
    ]
    history = TrainingHistory([], [], [], [], [])

    # 指数衰减能在训练前期多探索、后期更多利用已学策略。
    if episodes == 1:
        epsilon_decay = 1.0
    else:
        epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / (episodes - 1))
    report_every = max(1, episodes // 10)

    for episode in range(episodes):
        states = env.reset()
        epsilon = max(epsilon_end, epsilon_start * epsilon_decay**episode)
        episode_return = 0.0
        episode_energy = 0.0
        communication_successes = 0
        communication_total = 0
        last_info: DualUAVStepInfo | None = None

        done = False
        while not done:
            actions = [
                agents[i].choose_action(states[i], epsilon)
                for i in range(env.num_agents)
            ]
            next_states, reward, done, info = env.step(actions)

            # 两个智能体都用同一个 reward，体现协作目标。
            for i in range(env.num_agents):
                agents[i].update(states[i], actions[i], reward, next_states[i], done)

            states = next_states
            episode_return += reward
            episode_energy += info.propulsion_energy_proxy
            communication_successes += int(np.count_nonzero(info.communication_ok))
            communication_total += env.num_agents
            last_info = info

        assert last_info is not None
        history.episode_returns.append(episode_return)
        history.final_rhos.append(last_info.rho)
        history.final_distances.append(last_info.mean_target_distance)
        history.communication_success_rates.append(
            communication_successes / communication_total
        )
        history.episode_energy_proxies.append(episode_energy)

        if verbose and ((episode + 1) % report_every == 0 or episode == 0):
            window = min(report_every, len(history.episode_returns))
            mean_return = float(np.mean(history.episode_returns[-window:]))
            print(
                f"episode={episode + 1:4d}/{episodes}, "
                f"epsilon={epsilon:.3f}, "
                f"mean_return={mean_return:8.3f}, "
                f"rho={last_info.rho:.4e}, "
                f"mean_distance={last_info.mean_target_distance:6.2f} m"
            )

    return agents, history


def rollout_greedy_policy(
    env: DualUAVISACEnv,
    agents: Sequence[QLearningAgent],
) -> EpisodeTrace:
    """运行一个不学习的贪心回合，保存绘图所需的全部物理量。"""

    states = env.reset()
    positions = [env.uav_positions.copy()]
    targets = [env.target_state[:3].copy()]
    actions_history: List[np.ndarray] = []
    rewards: List[float] = []
    rhos: List[float] = []
    mean_distances: List[float] = []
    uav_distances: List[float] = []
    communication_sinrs: List[np.ndarray] = []
    sensing_sinrs: List[np.ndarray] = []
    speeds: List[np.ndarray] = []
    energy_proxies: List[float] = []
    reward_parts: DefaultDict[str, List[float]] = defaultdict(list)

    done = False
    while not done:
        actions = np.array(
            [agents[i].choose_action(states[i], epsilon=0.0) for i in range(env.num_agents)],
            dtype=int,
        )
        states, reward, done, info = env.step(actions)

        positions.append(info.uav_positions.copy())
        targets.append(info.target_state[:3].copy())
        actions_history.append(actions)
        rewards.append(reward)
        rhos.append(info.rho)
        mean_distances.append(info.mean_target_distance)
        uav_distances.append(info.uav_distance)
        communication_sinrs.append(info.communication_sinrs.copy())
        sensing_sinrs.append(info.sensing_sinrs.copy())
        speeds.append(info.uav_speeds.copy())
        energy_proxies.append(info.propulsion_energy_proxy)
        for name, value in info.reward_parts.items():
            reward_parts[name].append(value)

    times = np.arange(len(positions), dtype=float) * env.config.slot_duration
    return EpisodeTrace(
        times=times,
        uav_positions=np.stack(positions),
        target_positions=np.stack(targets),
        actions=np.stack(actions_history),
        rewards=np.asarray(rewards, dtype=float),
        rhos=np.asarray(rhos, dtype=float),
        mean_target_distances=np.asarray(mean_distances, dtype=float),
        uav_distances=np.asarray(uav_distances, dtype=float),
        communication_sinrs=np.stack(communication_sinrs),
        sensing_sinrs=np.stack(sensing_sinrs),
        uav_speeds=np.stack(speeds),
        energy_proxies=np.asarray(energy_proxies, dtype=float),
        reward_parts={
            name: np.asarray(values, dtype=float) for name, values in reward_parts.items()
        },
    )


def evaluate_policy(
    env: DualUAVISACEnv,
    agents: Sequence[QLearningAgent] | None,
    episodes: int = 5,
    random_policy: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    """评估随机策略或训练后的贪心策略，不再更新 Q 表。"""

    rng = np.random.default_rng(seed)
    returns: List[float] = []
    final_rhos: List[float] = []
    final_distances: List[float] = []
    communication_rates: List[float] = []
    collision_counts: List[int] = []
    energy_totals: List[float] = []
    mean_speeds: List[float] = []

    for _ in range(episodes):
        states = env.reset()
        done = False
        episode_return = 0.0
        comm_successes = 0
        comm_total = 0
        collisions = 0
        energy_total = 0.0
        speed_samples: List[float] = []
        last_info: DualUAVStepInfo | None = None

        while not done:
            if random_policy:
                actions = [int(rng.integers(env.num_actions)) for _ in range(env.num_agents)]
            else:
                if agents is None:
                    raise ValueError("评估贪心策略时必须传入 agents。")
                actions = [
                    agents[i].choose_action(states[i], epsilon=0.0)
                    for i in range(env.num_agents)
                ]

            states, reward, done, info = env.step(actions)
            episode_return += reward
            comm_successes += int(np.count_nonzero(info.communication_ok))
            comm_total += env.num_agents
            collisions += int(info.collision)
            energy_total += info.propulsion_energy_proxy
            speed_samples.extend(info.uav_speeds.tolist())
            last_info = info

        assert last_info is not None
        returns.append(episode_return)
        final_rhos.append(last_info.rho)
        final_distances.append(last_info.mean_target_distance)
        communication_rates.append(comm_successes / comm_total)
        collision_counts.append(collisions)
        energy_totals.append(energy_total)
        mean_speeds.append(float(np.mean(speed_samples)))

    return {
        "mean_return": float(np.mean(returns)),
        "mean_final_rho": float(np.mean(final_rhos)),
        "mean_final_distance": float(np.mean(final_distances)),
        "communication_success_rate": float(np.mean(communication_rates)),
        "mean_collisions": float(np.mean(collision_counts)),
        "mean_energy_proxy": float(np.mean(energy_totals)),
        "mean_uav_speed": float(np.mean(mean_speeds)),
    }


def _print_evaluation(title: str, result: Dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"mean return               : {result['mean_return']:.3f}")
    print(f"mean final rho            : {result['mean_final_rho']:.4e}")
    print(f"mean final target distance: {result['mean_final_distance']:.2f} m")
    print(f"communication success     : {100.0 * result['communication_success_rate']:.1f}%")
    print(f"mean collisions / episode : {result['mean_collisions']:.2f}")
    print(f"mean UAV speed            : {result['mean_uav_speed']:.2f} m/s")
    print(f"mean energy proxy         : {result['mean_energy_proxy']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="双 UAV 协作 Independent Q-Learning 示例")
    parser.add_argument("--episodes", type=int, default=300, help="训练回合数")
    parser.add_argument("--steps", type=int, default=30, help="每回合时隙数")
    parser.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--max-speed", type=float, default=5.0, help="UAV 最大速度，单位 m/s")
    parser.add_argument("--slot-duration", type=float, default=1.0, help="时隙长度，单位 s")
    parser.add_argument("--output-dir", default="output/rl", help="曲线和轨迹图输出目录")
    parser.add_argument("--no-plots", action="store_true", help="只训练，不生成 PNG 图")
    parser.add_argument("--quiet", action="store_true", help="不打印训练过程")
    args = parser.parse_args()

    config = DualUAVConfig(
        max_steps=args.steps,
        max_uav_speed=args.max_speed,
        slot_duration=args.slot_duration,
    )

    random_result = evaluate_policy(
        DualUAVISACEnv(config=config, seed=args.seed + 1),
        agents=None,
        episodes=args.eval_episodes,
        random_policy=True,
        seed=args.seed + 2,
    )

    train_env = DualUAVISACEnv(config=config, seed=args.seed)
    agents, history = train_independent_q_learning(
        train_env,
        episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet,
    )

    learned_result = evaluate_policy(
        DualUAVISACEnv(config=config, seed=args.seed + 3),
        agents=agents,
        episodes=args.eval_episodes,
        random_policy=False,
        seed=args.seed + 4,
    )

    _print_evaluation("Random policy baseline", random_result)
    _print_evaluation("Learned greedy policy", learned_result)
    print(f"\nQ-table states: UAV1={len(agents[0].q_table)}, UAV2={len(agents[1].q_table)}")
    print(f"Last 20 episode mean return: {np.mean(history.episode_returns[-20:]):.3f}")

    if not args.no_plots:
        trace = rollout_greedy_policy(
            DualUAVISACEnv(config=config, seed=args.seed + 5), agents
        )
        from uav_rl_visualization import save_run_artifacts

        output_paths = save_run_artifacts(
            history=history,
            trace=trace,
            config=config,
            random_result=random_result,
            learned_result=learned_result,
            output_dir=args.output_dir,
        )
        print("\nSaved visualizations:")
        for path in output_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()

#python dual_uav_rl.py --episodes 300 --steps 30 --eval-episodes 5 --seed 7 --max-speed 5 --output-dir output/rl