"""团队奖励的唯一计算入口：各项已经加权，求和就是环境 reward。"""

import numpy as np

from .config import EnvConfig, RewardConfig


def team_reward(
    env: EnvConfig, weights: RewardConfig, *, previous_distances, distances,
    active_before, newly_arrived, arrived, movement_velocities, accelerations,
    communication_sinrs, rho_pos, safety_intervention, boundary_requests,
    completed, timed_out,
):
    """本步到达者仍计入本步；退出后的占位值不参与活动相关代价。"""
    active = np.asarray(active_before, dtype=bool)
    progress = (previous_distances[active] - distances[active]) / (
        env.max_uav_speed * env.slot_duration)
    remaining = distances[active] / (distances[active] + weights.distance_ref)
    energy = (
        np.sum(movement_velocities[active] ** 2, axis=1) / env.max_uav_speed ** 2
        + .5 * np.sum(accelerations[active] ** 2, axis=1) / env.max_uav_acceleration ** 2
    )
    comm_shortfall = np.maximum(0., 1. - communication_sinrs[active] / env.gamma_min)

    # 分母始终是两架 UAV，不能在一架退出后改成活动机数。
    return {
        "progress": weights.progress * float(progress.sum()) / 2,
        "distance": -weights.distance * float(remaining.sum()) / 2,
        "arrival": weights.arrival * float(np.count_nonzero(newly_arrived)) / 2,
        "completion": weights.completion * float(completed),
        "sensing": -weights.sensing * float(rho_pos / (rho_pos + weights.rho_ref)),
        "time": -weights.time * float(active.sum()) / 2,
        "energy": -weights.energy * float(energy.sum()) / 2,
        "communication": -weights.communication * float(comm_shortfall.sum()) / 2,
        "collision": -weights.collision * float(safety_intervention),
        "boundary": -weights.boundary * float(np.count_nonzero(boundary_requests[active])) / 2,
        "timeout": -weights.timeout * float(timed_out) * float(np.count_nonzero(~arrived)) / 2,
    }