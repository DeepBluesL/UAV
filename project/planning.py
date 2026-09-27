"""基于公开观测的导航规划动力学，与环境动作执行顺序保持一致。"""

import numpy as np


def physical_to_raw(acceleration, active, config):
    acceleration = np.asarray(acceleration, dtype=float).copy()
    active = np.asarray(active, dtype=bool)
    acceleration[~active] = 0.0
    norm = np.linalg.norm(acceleration, axis=-1, keepdims=True)
    acceleration /= np.maximum(1.0, norm / config.max_uav_acceleration)
    scaled = np.clip(acceleration / config.max_uav_acceleration, -.999, .999)
    raw = np.arctanh(scaled).astype(np.float32)
    raw[~active] = 0.0
    return raw


def executed_acceleration(acceleration, config):
    """复现 physical→raw→tanh→范数限制的实际加速度。"""
    acceleration = np.asarray(acceleration, dtype=float).copy()
    norm = np.linalg.norm(acceleration, axis=-1, keepdims=True)
    acceleration /= np.maximum(1.0, norm / config.max_uav_acceleration)
    scaled = np.clip(acceleration / config.max_uav_acceleration, -.999, .999)
    executed = np.tanh(np.arctanh(scaled)) * config.max_uav_acceleration
    norm = np.linalg.norm(executed, axis=-1, keepdims=True)
    return executed / np.maximum(1.0, norm / config.max_uav_acceleration)


def swept_distance(start, end):
    """同步线段最小距离；支持前置任意批量维。"""
    relative = start[..., 0, :] - start[..., 1, :]
    change = ((end[..., 0, :] - start[..., 0, :])
              - (end[..., 1, :] - start[..., 1, :]))
    length2 = np.sum(change ** 2, axis=-1)
    numerator = -np.sum(relative * change, axis=-1)
    fraction = np.divide(numerator, length2, out=np.zeros_like(length2), where=length2 > 0)
    fraction = np.clip(fraction, 0.0, 1.0)
    return np.linalg.norm(relative + fraction[..., None] * change, axis=-1)


def rollout_cost(sequences, position, velocity, goals, active, config, weights):
    """批量评估 [B,H,2,3] 控制序列，返回纯导航代价。"""
    sequences = np.asarray(sequences, dtype=float)
    if sequences.ndim == 3:
        sequences = sequences[None]
    batch, horizon = sequences.shape[:2]
    pos = np.broadcast_to(position, (batch, 2, 3)).copy()
    vel = np.broadcast_to(velocity, (batch, 2, 3)).copy()
    live = np.broadcast_to(active, (batch, 2)).copy()
    goals = np.broadcast_to(goals, (batch, 2, 3))
    cost = np.zeros(batch)
    safe = np.ones(batch, dtype=bool)
    for step in range(horizon):
        live_before = live.copy()
        acc = executed_acceleration(sequences[:, step], config)
        acc = np.where(live[..., None], acc, 0.0)
        next_vel = vel + acc * config.slot_duration
        speed = np.linalg.norm(next_vel, axis=-1, keepdims=True)
        next_vel /= np.maximum(1.0, speed / config.max_uav_speed)
        next_vel = np.where(live[..., None], next_vel, 0.0)
        requested = pos + next_vel * config.slot_duration
        next_pos = np.clip(requested, config.world_low, config.world_high)
        boundary = np.any(next_pos != requested, axis=-1) & live
        clearance = swept_distance(pos, next_pos) - config.min_uav_distance
        collision = clearance < 0.
        safe &= ~collision
        # 环境检测到扫掠冲突时共同取消该步位移。
        next_pos = np.where(collision[:, None, None], pos, next_pos)
        distance = np.linalg.norm(goals - next_pos, axis=-1) / config.position_scale
        cost += weights["navigation"] * np.sum(distance * live_before, axis=-1)
        cost += weights["effort"] * np.sum(
            (acc / config.max_uav_acceleration) ** 2, axis=(-2, -1))
        cost += weights["clearance"] / np.maximum(clearance / config.min_uav_distance, .1)
        cost += weights["boundary"] * np.sum(boundary, axis=-1)
        arrived = np.linalg.norm(goals - next_pos, axis=-1) <= config.goal_tolerance
        live &= ~arrived
        vel = np.where(live[..., None], (next_pos - pos) / config.slot_duration, 0.0)
        pos = next_pos
    cost[~safe] += weights["collision"]
    return cost
