"""Versioned observations; v1 stays checkpoint-compatible."""

import numpy as np

from .observation_specs import V1, observation_spec

OBS_DIM = V1.obs_dim
STATE_DIM = V1.state_dim
_OFF_DIAGONAL = np.triu_indices(6, 1)


def _belief_features(env):
    cfg = env.config
    covariance = np.asarray(env.pcrb_matrix, dtype=float)
    position_logs = np.log1p(np.diag(covariance)[:3] / (cfg.observation_rho_ref / 3))
    velocity_logs = np.log1p(
        np.diag(covariance)[3:] / cfg.observation_velocity_variance_ref)
    deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(deviations, deviations)
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    return position_logs, velocity_logs, np.clip(
        correlation[_OFF_DIAGONAL], -1.0, 1.0)


def decode_belief(observation_row, config):
    """Reconstruct v2's public 6x6 belief covariance from one UAV row."""
    row = np.asarray(observation_row, dtype=float)
    if observation_spec(config).version != "v2" or row.shape != (67,):
        raise ValueError("decode_belief expects one v2 observation row of shape (67,)")
    variances = np.r_[
        np.expm1(row[16:19]) * (config.observation_rho_ref / 3),
        np.expm1(row[31:34]) * config.observation_velocity_variance_ref,
    ]
    deviations = np.sqrt(np.maximum(variances, 0.0))
    covariance = np.diag(variances)
    products = np.outer(deviations, deviations)
    covariance[_OFF_DIAGONAL] = row[34:49] * products[_OFF_DIAGONAL]
    covariance[(_OFF_DIAGONAL[1], _OFF_DIAGONAL[0])] = covariance[_OFF_DIAGONAL]
    return covariance


def _legacy_rows(env, uncertainty):
    cfg = env.config
    remaining = np.array([max(0., 1. - env.step_count / cfg.max_steps)])
    flags = np.concatenate([env.active, env.arrived]).astype(float)
    comm = np.log1p(env.metrics["communication_sinrs"]) / cfg.sinr_log_scale
    sensing = np.log1p(env.metrics["sensing_sinrs"]) / cfg.sinr_log_scale
    estimate = env.estimated_target_state
    rows = []
    for i in range(2):
        j = 1 - i
        rows.append(np.concatenate([
            env.uav_positions[i] / cfg.position_scale,
            env.uav_velocities[i] / cfg.velocity_scale,
            (cfg.uav_goal_positions[i] - env.uav_positions[i]) / cfg.position_scale,
            remaining, (estimate[:3] - env.uav_positions[i]) / cfg.position_scale,
            estimate[3:] / cfg.velocity_scale, uncertainty,
            (env.uav_positions[j] - env.uav_positions[i]) / cfg.position_scale,
            (env.uav_velocities[j] - env.uav_velocities[i]) / cfg.velocity_scale,
            flags, comm[i:i + 1], sensing[i + 1:i + 2],
        ]))
    return rows, flags, remaining, comm, sensing, estimate


def build_observations(env):
    """Build two local observations and one centralized training state."""
    cfg, spec = env.config, observation_spec(env.config)
    legacy_uncertainty = np.log1p(
        np.diag(env.pcrb_matrix)[:3] / (env.reward_config.rho_ref / 3))
    position_logs, velocity_logs, correlations = _belief_features(env)
    uncertainty = legacy_uncertainty if spec.version == "v1" else position_logs
    rows, flags, remaining, comm, sensing, estimate = _legacy_rows(env, uncertainty)

    scales = np.array([cfg.covariance_position_std] * 3 +
                      [cfg.covariance_velocity_std] * 3)
    normalized_covariance = env.pcrb_matrix / np.outer(scales, scales)
    state = np.concatenate([
        env.uav_positions.ravel() / cfg.position_scale,
        env.uav_velocities.ravel() / cfg.velocity_scale,
        cfg.uav_goal_positions.ravel() / cfg.position_scale,
        flags, remaining, estimate[:3] / cfg.position_scale,
        estimate[3:] / cfg.velocity_scale,
        normalized_covariance[np.triu_indices(6)], comm, sensing,
        env.metrics["sensing_source_mask"].astype(float),
    ])
    if spec.version == "v2":
        seconds = max(0.0, (cfg.max_steps - env.step_count) * cfg.slot_duration)
        source_mask = env.metrics.get(
            "measurement_source_mask", env.metrics["sensing_source_mask"])
        shared = np.r_[
            velocity_logs, correlations, seconds / cfg.observation_time_scale,
            cfg.max_uav_speed / 10.0, cfg.max_uav_acceleration / 5.0,
            cfg.slot_duration,
        ]
        for i, row in enumerate(rows):
            boundary = np.r_[env.uav_positions[i] - cfg.world_low,
                             cfg.world_high - env.uav_positions[i]] / cfg.position_scale
            rows[i] = np.r_[
                row, shared, boundary,
                (cfg.bs_position - env.uav_positions[i]) / cfg.position_scale,
                source_mask.astype(float), comm[1 - i], np.log1p(cfg.gamma_min) / 10.0,
            ]
        state = np.r_[
            state, seconds / cfg.observation_time_scale,
            cfg.max_uav_speed / 10.0, cfg.max_uav_acceleration / 5.0,
            cfg.slot_duration, cfg.world_low / cfg.position_scale,
            cfg.world_high / cfg.position_scale, cfg.bs_position / cfg.position_scale,
            np.log1p(cfg.gamma_min) / 10.0,
        ]
    observations, state = np.asarray(rows, dtype=np.float32), state.astype(np.float32)
    if observations.shape != (2, spec.obs_dim) or state.shape != (spec.state_dim,):
        raise RuntimeError("Observation builder disagrees with its versioned schema")
    return observations, state
