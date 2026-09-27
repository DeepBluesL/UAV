"""联合时间步、任务退出和安全处理；物理计算与奖励分别委托给独立模块。"""

import numpy as np

from .config import EnvConfig, RewardConfig
from .observations import OBS_DIM, STATE_DIM, build_observations
from .physics import ISACPhysics
from .rewards import team_reward


def swept_distance(start, end):
    """同步直线运动期间的两机最小距离，也适用于一架已经停驻的情形。"""
    relative = start[0] - start[1]
    change = (end[0] - start[0]) - (end[1] - start[1])
    length2 = float(change @ change)
    fraction = np.clip(-relative @ change / length2, 0., 1.) if length2 > 0 else 0.
    return float(np.linalg.norm(relative + fraction * change))


class DualUAVEnv:
    obs_dim = OBS_DIM
    state_dim = STATE_DIM
    action_dim = 3

    def __init__(self, config=None, reward_config=None, seed=7):
        self.config = config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.physics = ISACPhysics(self.config, self.rng)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.physics = ISACPhysics(self.config, self.rng)
        cfg = self.config
        self.step_count = 0
        self.uav_positions = cfg.uav_initial.copy()
        self.uav_velocities = cfg.uav_initial_velocities.copy()
        self.arrived = self.goal_distances <= cfg.goal_tolerance
        self.active = ~self.arrived
        self.uav_velocities[~self.active] = 0.
        self.arrival_steps = np.where(self.arrived, 0, -1)
        self.path_lengths = np.zeros(2)
        self.energy_proxies = np.zeros(2)
        self.active_steps = np.zeros(2, dtype=int)
        self.target_state = cfg.target_initial_state.copy()
        self.estimated_target_state, self.pcrb_matrix = self.physics.initialize_belief(
            self.target_state)
        self.prior_estimated_target_state = self.estimated_target_state.copy()
        self.j_prev = np.linalg.pinv(self.pcrb_matrix)
        self.metrics = self.physics.link_metrics(
            self.uav_positions, self.target_state, self.estimated_target_state, self.active)
        self.metrics.update(
            pcrb=self.pcrb_matrix.copy(), j=self.j_prev.copy(),
            rho_pos=float(np.trace(self.pcrb_matrix[:3, :3])),
            rho_all=float(np.trace(self.pcrb_matrix)),
            estimated_target_state=self.estimated_target_state.copy(),
            prior_estimated_target_state=self.prior_estimated_target_state.copy(),
            measurement_source_mask=np.zeros(3, dtype=bool),
            measurement_counts=np.zeros(3, dtype=int), measurement_count=0,
            uncertainty_kind="ekf_covariance" if cfg.sensing_mode == "ekf" else "proxy_pcrb")
        self.unsafe = False
        self.terminated = bool(self.arrived.all())
        self.termination_reason = "success" if self.terminated else None
        obs, state = build_observations(self)
        return obs, state, self._info()

    @property
    def goal_distances(self):
        return np.linalg.norm(self.uav_positions - self.config.uav_goal_positions, axis=1)

    def step(self, raw_actions):
        if self.terminated:
            raise RuntimeError("The team episode has ended; call reset() before step()")
        raw_actions = np.asarray(raw_actions, dtype=float)
        if raw_actions.shape != (2, 3) or not np.isfinite(raw_actions[self.active]).all():
            raise ValueError("Active UAV actions must be finite; expected shape (2, 3)")
        cfg = self.config
        # 必须在执行动作前保存：本步首次到达者仍有一个有效训练样本。
        active_before = self.active.copy()
        previous_positions = self.uav_positions.copy()
        previous_distances = self.goal_distances
        accelerations, movement_velocities, boundary, intervention, minimum_distance = self._move(raw_actions)
        self.active_steps += active_before
        self.path_lengths += np.linalg.norm(self.uav_positions - previous_positions, axis=1)
        energy = (
            np.sum(movement_velocities ** 2, axis=1) / cfg.max_uav_speed ** 2
            + .5 * np.sum(accelerations ** 2, axis=1) / cfg.max_uav_acceleration ** 2)
        self.energy_proxies += energy * active_before

        previous_target = self.target_state.copy()
        dt = cfg.slot_duration
        self.target_state[:3] += previous_target[3:] * dt + .5 * cfg.target_acceleration * dt ** 2
        self.target_state[3:] += cfg.target_acceleration * dt
        if cfg.sensing_mode == "proxy":
            self.estimated_target_state = self.physics.noisy_estimate(self.target_state)
        self.metrics = self.physics.update(
            self.uav_positions, self.target_state, self.estimated_target_state,
            active_before, previous_target, self.j_prev, movement_velocities)
        if cfg.sensing_mode == "ekf":
            self.prior_estimated_target_state = self.metrics["prior_estimated_target_state"].copy()
            self.estimated_target_state = self.metrics["estimated_target_state"].copy()
        else:
            self.prior_estimated_target_state = self.estimated_target_state.copy()
        self.j_prev = self.metrics["j"]
        self.pcrb_matrix = self.metrics["pcrb"]
        self.step_count += 1

        distances = self.goal_distances
        actual_collision = minimum_distance < cfg.min_uav_distance - 1.e-9
        self.unsafe |= actual_collision
        newly_arrived = active_before & (distances <= cfg.goal_tolerance) & (not self.unsafe)
        self.arrival_steps[newly_arrived] = self.step_count
        self.arrived |= newly_arrived
        completed = bool(self.arrived.all() and not self.unsafe)
        timed_out = bool(self.step_count >= cfg.max_steps and not completed)
        self.terminated = bool(completed or timed_out or self.unsafe)
        self.termination_reason = (
            "success" if completed else "collision" if self.unsafe else "deadline" if timed_out else None)

        reward_parts = team_reward(
            cfg, self.reward_config, previous_distances=previous_distances,
            distances=distances, active_before=active_before, newly_arrived=newly_arrived,
            arrived=self.arrived, movement_velocities=movement_velocities,
            accelerations=accelerations, communication_sinrs=self.metrics["communication_sinrs"],
            rho_pos=self.metrics["rho_pos"], safety_intervention=intervention or actual_collision,
            boundary_requests=boundary, completed=completed, timed_out=timed_out)
        self.active &= ~self.arrived
        # 任务结束后的停驻抽象；最后一步运动已经用于计能耗和路径长度。
        self.uav_velocities[~self.active] = 0.
        obs, state = build_observations(self)
        info = self._info()
        info.update(
            active_before=active_before, newly_arrived=newly_arrived,
            accelerations=accelerations, movement_velocities=movement_velocities,
            energy_step=energy, boundary_requests=boundary,
            safety_intervention=bool(intervention), collision=bool(actual_collision),
            minimum_separation=minimum_distance, reward_parts=reward_parts)
        # 本任务的有限期限属于 terminated；采样批次截止由训练器单独处理。
        return obs, state, float(sum(reward_parts.values())), self.terminated, False, info

    def _move(self, raw_actions):
        cfg = self.config
        start = self.uav_positions.copy()
        accelerations = np.zeros((2, 3))
        accelerations[self.active] = np.tanh(raw_actions[self.active]) * cfg.max_uav_acceleration
        norms = np.linalg.norm(accelerations, axis=1, keepdims=True)
        accelerations /= np.maximum(1., norms / cfg.max_uav_acceleration)
        velocity = self.uav_velocities + accelerations * cfg.slot_duration
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity /= np.maximum(1., speed / cfg.max_uav_speed)
        velocity[~self.active] = 0.
        requested = start + velocity * cfg.slot_duration
        candidate = np.clip(requested, cfg.world_low, cfg.world_high)
        boundary = np.any(candidate != requested, axis=1) & self.active
        intervention = swept_distance(start, candidate) < cfg.min_uav_distance
        if intervention:
            candidate = start.copy()  # 与旧环境一致：冲突时共同取消本步位移。
        self.uav_positions = candidate
        self.uav_velocities = (candidate - start) / cfg.slot_duration
        return (accelerations, self.uav_velocities.copy(), boundary, intervention,
                swept_distance(start, candidate))

    def _info(self):
        info = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.metrics.items()
        }
        info.update(
            step=self.step_count, active=self.active.copy(), arrived=self.arrived.copy(),
            arrival_steps=self.arrival_steps.copy(),
            arrival_times=np.where(self.arrived, self.arrival_steps * self.config.slot_duration, np.nan),
            positions=self.uav_positions.copy(), velocities=self.uav_velocities.copy(),
            target_state=self.target_state.copy(), estimated_target_state=self.estimated_target_state.copy(),
            prior_estimated_target_state=self.prior_estimated_target_state.copy(),
            tracking_covariance=self.pcrb_matrix.copy(),
            tracking_covariance_trace=float(np.trace(self.pcrb_matrix)),
            tracking_position_error=float(np.linalg.norm(
                self.estimated_target_state[:3] - self.target_state[:3])),
            prior_tracking_position_error=float(np.linalg.norm(
                self.prior_estimated_target_state[:3] - self.target_state[:3])),
            goal_distances=self.goal_distances, path_lengths=self.path_lengths.copy(),
            energy_proxies=self.energy_proxies.copy(), active_steps=self.active_steps.copy(),
            rho_pos=float(np.trace(self.pcrb_matrix[:3, :3])),
            rho_all=float(np.trace(self.pcrb_matrix)),
            team_success=bool(self.arrived.all() and not self.unsafe),
            terminated=self.terminated, termination_reason=self.termination_reason,
            communication_ok=self.metrics["communication_sinrs"] >= self.config.gamma_min,
        )
        return info
