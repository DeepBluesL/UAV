"""集中修改场景、奖励和 PPO 参数；长度单位 m，时间单位 s。"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvConfig:
    max_steps: int = 100
    slot_duration: float = 1.0
    max_uav_speed: float = 5.0
    max_uav_acceleration: float = 5.0
    min_uav_distance: float = 5.0
    goal_tolerance: float = 5.0
    uav_initial: np.ndarray = field(default_factory=lambda: np.array(
        [[30., 10., 70.], [30., -10., 70.]]))
    uav_initial_velocities: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    uav_goal_positions: np.ndarray = field(default_factory=lambda: np.array(
        [[150., 70., 90.], [120., -30., 30.]]))
    world_low: np.ndarray = field(default_factory=lambda: np.array([-20., -80., 20.]))
    world_high: np.ndarray = field(default_factory=lambda: np.array([150., 80., 120.]))
    bs_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_initial_state: np.ndarray = field(default_factory=lambda: np.array(
        [60., 0., 60., -1., 1., -1.]))
    target_acceleration: np.ndarray = field(default_factory=lambda: np.array([.02, -.02, .02]))

    # 第一版固定物理难度；波束为确定性规则，没有 BS Actor。
    antenna_x: int = 8
    antenna_y: int = 8
    total_power: float = 5.0
    bandwidth: float = 100.e6
    rician_k: float = 10.0
    beta0_c: float = 1.e-5
    beta0_s: float = 1.e-5
    alpha_target: float = .9
    alpha_uav: float = .9
    awgn_power: float = 1.e-11
    residual_noise: float = 1.e-10
    gamma_min: float = 10. ** .5
    kappa_d: float = 1.0
    kappa_theta: float = 1.e-6
    kappa_phi: float = 1.e-6
    kappa_v: float = 1.e-6
    process_noise_intensity: float = .1
    initial_covariance: float = 1.0
    imperfect_csi_beta: float = .10
    csi_position_std: float = 10.0
    csi_velocity_std: float = 1.0
    csi_beta_jitter: float = .02

    # 固定观测尺度，不随当前活动机数或当前回合改变。
    position_scale: float = 150.0
    velocity_scale: float = 10.0
    sinr_log_scale: float = 10.0
    covariance_position_std: float = 1.0
    covariance_velocity_std: float = 1.0

    def __post_init__(self):
        shapes = {
            "uav_initial": (2, 3), "uav_initial_velocities": (2, 3),
            "uav_goal_positions": (2, 3), "world_low": (3,), "world_high": (3,),
            "bs_position": (3,), "target_initial_state": (6,), "target_acceleration": (3,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite array of shape {shape}")
            setattr(self, name, value.copy())
        positive = (
            "max_steps", "slot_duration", "max_uav_speed", "max_uav_acceleration",
            "min_uav_distance", "goal_tolerance", "total_power", "bandwidth",
            "antenna_x", "antenna_y", "gamma_min", "initial_covariance",
            "process_noise_intensity", "position_scale", "velocity_scale",
            "sinr_log_scale", "covariance_position_std", "covariance_velocity_std",
        )
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError("Time, limits, power and normalization scales must be positive")
        if np.any(self.world_low >= self.world_high):
            raise ValueError("world_low must be below world_high")
        for positions in (self.uav_initial, self.uav_goal_positions):
            if np.any(positions < self.world_low) or np.any(positions > self.world_high):
                raise ValueError("UAV starts and goals must lie inside the world")
        if np.linalg.norm(self.uav_initial[0] - self.uav_initial[1]) < self.min_uav_distance:
            raise ValueError("Initial UAV positions violate the separation distance")
        if np.any(np.linalg.norm(self.uav_initial_velocities, axis=1) > self.max_uav_speed):
            raise ValueError("Initial velocity exceeds max_uav_speed")
        if not 0 <= self.imperfect_csi_beta <= 1:
            raise ValueError("imperfect_csi_beta must be in [0, 1]")


@dataclass
class RewardConfig:
    progress: float = 5.0
    distance: float = .5
    time: float = .2
    energy: float = .05
    communication: float = .5
    sensing: float = .2
    arrival: float = 80.0            # 固定除以 2 后，每架首次到达得到 40。
    completion: float = 80.0
    timeout: float = 100.0           # 固定除以 2 后，每架未完成扣 50。
    collision: float = 20.0
    boundary: float = 2.0
    distance_ref: float = 100.0
    rho_ref: float = .01             # m²：物理探针给出的候选值，正式比较前固定。

    def __post_init__(self):
        if self.distance_ref <= 0 or self.rho_ref <= 0:
            raise ValueError("distance_ref and rho_ref must be positive")


@dataclass
class PPOConfig:
    hidden_sizes: tuple = (128, 128)
    gamma: float = .99
    lam: float = .95
    clip_ratio: float = .2
    pi_lr: float = 3.e-4
    vf_lr: float = 1.e-3
    train_pi_iters: int = 4
    train_v_iters: int = 4
    target_kl: float = .01
    entropy_coef: float = 0.0
    max_grad_norm: float = .5
    steps_per_epoch: int = 2048
    epochs: int = 100
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self):
        if min(self.steps_per_epoch, self.epochs, self.train_pi_iters, self.train_v_iters) < 1:
            raise ValueError("Rollout size, epochs and update counts must be positive")
        if not (0 <= self.gamma <= 1 and 0 <= self.lam <= 1):
            raise ValueError("gamma and lam must be in [0, 1]")