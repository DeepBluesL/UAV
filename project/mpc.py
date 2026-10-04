"""不使用通信/感知真值的离散导航 MPC，并非完整 ISAC 优化。"""

import itertools

import numpy as np

from .baselines import acceleration_to_action, observation_state


def _swept_distance(start, end):
    """批量同步线段距离，输入末三维为 [2,3]。"""
    relative = start[..., 0, :] - start[..., 1, :]
    change = ((end[..., 0, :] - start[..., 0, :])
              - (end[..., 1, :] - start[..., 1, :]))
    length2 = np.sum(change ** 2, axis=-1)
    numerator = -np.sum(relative * change, axis=-1)
    fraction = np.divide(numerator, length2, out=np.zeros_like(length2), where=length2 > 0)
    fraction = np.clip(fraction, 0.0, 1.0)
    return np.linalg.norm(relative + fraction[..., None] * change, axis=-1)


class MPCController:
    """小候选集短时域联合规划；有限候选本身不构成安全保证。"""

    DEFAULT_SETTINGS = {
        "horizon": 3, "effort_weight": 0.03, "speed_weight": 0.02,
        "clearance_weight": 8.0, "terminal_weight": 2.0,
    }

    def __init__(self, config, horizon=3, effort_weight=.03, speed_weight=.02,
                 clearance_weight=8.0, terminal_weight=2.0):
        self.config = config
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        self.effort_weight = float(effort_weight)
        self.speed_weight = float(speed_weight)
        self.clearance_weight = float(clearance_weight)
        self.terminal_weight = float(terminal_weight)
        self.settings = dict(horizon=self.horizon, effort_weight=self.effort_weight,
                             speed_weight=self.speed_weight,
                             clearance_weight=self.clearance_weight,
                             terminal_weight=self.terminal_weight)

    def reset(self, seed=None):
        del seed

    def _trajectory_cost_enabled(self):
        return False

    def _trajectory_cost(self, positions, velocities, active):
        del positions, velocities, active
        return 0.0

    def _candidates(self, goal, velocity, is_active):
        if not is_active:
            return [np.zeros(3)]
        cfg = self.config
        distance = np.linalg.norm(goal)
        goal_direction = goal / max(distance, 1e-12)
        desired_speed = min(cfg.max_uav_speed, distance / max(cfg.slot_duration, 1e-12))
        brake = -velocity / cfg.slot_duration
        track = (goal_direction * desired_speed - velocity) / cfg.slot_duration
        side = np.cross(goal_direction, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 1e-8:
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        magnitude = cfg.max_uav_acceleration
        return [np.zeros(3), brake, track, goal_direction * magnitude,
                -goal_direction * magnitude, side * magnitude, -side * magnitude]

    def _executed_acceleration(self, acceleration):
        """复现 physical -> raw -> env physical 的两次饱和。"""
        cfg = self.config
        acceleration = np.asarray(acceleration, dtype=float).copy()
        norms = np.linalg.norm(acceleration, axis=-1, keepdims=True)
        acceleration /= np.maximum(1.0, norms / cfg.max_uav_acceleration)
        scaled = np.clip(acceleration / cfg.max_uav_acceleration, -.999, .999)
        executed = np.tanh(np.arctanh(scaled)) * cfg.max_uav_acceleration
        norms = np.linalg.norm(executed, axis=-1, keepdims=True)
        return executed / np.maximum(1.0, norms / cfg.max_uav_acceleration)

    def _step(self, position, velocity, acceleration, active, goals):
        """批量预测一步；本步到达者从下一步起冻结。"""
        cfg = self.config
        acceleration = self._executed_acceleration(acceleration)
        acceleration = np.where(active[..., None], acceleration, 0.0)
        next_velocity = velocity + acceleration * cfg.slot_duration
        speed = np.linalg.norm(next_velocity, axis=-1, keepdims=True)
        next_velocity /= np.maximum(1.0, speed / cfg.max_uav_speed)
        next_velocity = np.where(active[..., None], next_velocity, 0.0)
        requested = position + next_velocity * cfg.slot_duration
        next_position = np.clip(requested, cfg.world_low, cfg.world_high)
        safe = _swept_distance(position, next_position) >= cfg.min_uav_distance
        moved_velocity = (next_position - position) / cfg.slot_duration
        arrived = np.linalg.norm(goals - next_position, axis=-1) <= cfg.goal_tolerance
        next_active = active & ~arrived
        moved_velocity = np.where(next_active[..., None], moved_velocity, 0.0)
        return next_position, moved_velocity, next_active, safe, acceleration

    def _cost(self, position, velocity, goals, acceleration, active, terminal=False):
        cfg = self.config
        distance = np.linalg.norm(goals - position, axis=-1)
        clearance = np.linalg.norm(position[..., 0, :] - position[..., 1, :], axis=-1)
        clearance -= cfg.min_uav_distance
        cost = np.sum(distance * active, axis=-1)
        cost += self.effort_weight * np.sum(
            (acceleration / cfg.max_uav_acceleration) ** 2 * active[..., None], axis=(-2, -1))
        cost += self.speed_weight * np.sum(
            (velocity / cfg.max_uav_speed) ** 2 * active[..., None], axis=(-2, -1))
        cost += self.clearance_weight / np.maximum(clearance, .25)
        return cost * (self.terminal_weight if terminal else 1.0)

    def act(self, obs, active, deterministic=True):
        del deterministic
        active = np.asarray(active, dtype=bool)
        position, velocity, goal_delta, teammate_position, teammate_velocity = observation_state(obs, self.config)
        # 两份局部观测应一致；平均可抑制 float32 归一化误差。
        position = np.vstack(((position[0] + teammate_position[1]) / 2,
                              (position[1] + teammate_position[0]) / 2))
        velocity = np.vstack(((velocity[0] + teammate_velocity[1]) / 2,
                              (velocity[1] + teammate_velocity[0]) / 2))
        goals = position + goal_delta
        candidates = [self._candidates(goal_delta[i], velocity[i], active[i]) for i in range(2)]
        acceleration = np.asarray(list(itertools.product(*candidates)))
        count = len(acceleration)
        trial_position = np.broadcast_to(position, (count, 2, 3)).copy()
        trial_velocity = np.broadcast_to(velocity, (count, 2, 3)).copy()
        trial_active = np.broadcast_to(active, (count, 2)).copy()
        trial_goals = np.broadcast_to(goals, (count, 2, 3))
        cost, safe = np.zeros(count), np.ones(count, dtype=bool)
        collect_trace = self._trajectory_cost_enabled()
        trace_position, trace_velocity, trace_active = [], [], []
        for step in range(self.horizon):
            active_before = trial_active.copy()
            previous_position = trial_position.copy() if collect_trace else None
            trial_position, trial_velocity, trial_active, step_safe, executed = self._step(
                trial_position, trial_velocity, acceleration, trial_active, trial_goals)
            safe &= step_safe
            cost += self._cost(trial_position, trial_velocity, trial_goals, executed,
                               active_before, terminal=step == self.horizon - 1)
            if collect_trace:
                trace_position.append(trial_position.copy())
                trace_velocity.append(
                    (trial_position - previous_position) / self.config.slot_duration)
                trace_active.append(active_before)
        if collect_trace:
            cost += self._trajectory_cost(
                np.stack(trace_position, axis=1), np.stack(trace_velocity, axis=1),
                np.stack(trace_active, axis=1))
        cost[~safe] = np.inf
        if np.isfinite(cost).any():
            best = acceleration[int(np.argmin(cost))]
        else:
            # 极端情形返回制动；最终冲突仍由环境安全修正负责。
            best = -velocity / self.config.slot_duration
            best[~active] = 0.0
        return acceleration_to_action(best, active, self.config)
