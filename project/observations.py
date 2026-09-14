"""观测字段顺序及固定尺度；只读取状态和已交付消息，不读取目标真值。"""

import numpy as np

OBS_DIM = 31
STATE_DIM = 58


def build_observations(env):
    """两机局部观测 [2,31]，以及用于训练的集中状态 [58]。"""
    cfg = env.config
    remaining_time = np.array([max(0., 1. - env.step_count / cfg.max_steps)])
    flags = np.concatenate([env.active, env.arrived]).astype(float)
    comm = np.log1p(env.metrics["communication_sinrs"]) / cfg.sinr_log_scale
    sensing = np.log1p(env.metrics["sensing_sinrs"]) / cfg.sinr_log_scale
    uncertainty = np.log1p(np.diag(env.pcrb_matrix)[:3] / (env.reward_config.rho_ref / 3))
    # 每步可靠、零时延交付的模拟 BS 消息；不代表已经实现测量滤波。
    estimate = env.estimated_target_state
    observations = []
    for i in range(2):
        j = 1 - i
        observations.append(np.concatenate([
            env.uav_positions[i] / cfg.position_scale,                          # 0:3
            env.uav_velocities[i] / cfg.velocity_scale,                         # 3:6
            (cfg.uav_goal_positions[i] - env.uav_positions[i]) / cfg.position_scale,  # 6:9
            remaining_time,                                                   # 9:10
            (estimate[:3] - env.uav_positions[i]) / cfg.position_scale,         # 10:13
            estimate[3:] / cfg.velocity_scale,                                 # 13:16
            uncertainty,                                                      # 16:19
            (env.uav_positions[j] - env.uav_positions[i]) / cfg.position_scale, # 19:22
            (env.uav_velocities[j] - env.uav_velocities[i]) / cfg.velocity_scale, # 22:25
            flags,                                                            # 25:29
            comm[i:i + 1], sensing[i + 1:i + 2],                               # 29:31
        ]))

    # 完整对称矩阵的上三角保留递推信息；位置/速度单位分别归一化。
    scales = np.array([cfg.covariance_position_std] * 3 + [cfg.covariance_velocity_std] * 3)
    normalized_covariance = env.pcrb_matrix / np.outer(scales, scales)
    state = np.concatenate([
        env.uav_positions.ravel() / cfg.position_scale,
        env.uav_velocities.ravel() / cfg.velocity_scale,
        cfg.uav_goal_positions.ravel() / cfg.position_scale,
        flags, remaining_time,
        estimate[:3] / cfg.position_scale, estimate[3:] / cfg.velocity_scale,
        normalized_covariance[np.triu_indices(6)],
        comm, sensing, env.metrics["sensing_source_mask"].astype(float),
    ])
    return np.asarray(observations, dtype=np.float32), state.astype(np.float32)