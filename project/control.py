"""Policy-to-environment action mapping for pure and residual control."""

import numpy as np


def _norm_clip(vectors, limit):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(1., norms / limit)


def goal_raw_action(obs, active, config):
    """Existing GoalController command in the environment's raw coordinates."""
    velocity = np.asarray(obs[:, 3:6], dtype=float) * config.velocity_scale
    relative_goal = np.asarray(obs[:, 6:9], dtype=float) * config.position_scale
    distance = np.linalg.norm(relative_goal, axis=1, keepdims=True)
    desired_speed = np.minimum(config.max_uav_speed, distance / config.slot_duration)
    desired_velocity = relative_goal / np.maximum(distance, 1.e-12) * desired_speed
    acceleration = (desired_velocity - velocity) / config.slot_duration
    scaled = np.clip(acceleration / config.max_uav_acceleration, -.999, .999)
    action = np.arctanh(scaled).astype(np.float32)
    action[~np.asarray(active, dtype=bool)] = 0.
    return action


def executable_acceleration(raw_action, config):
    """Apply the tanh and vector clipping performed by DualUAVEnv._move."""
    acceleration = np.tanh(raw_action) * config.max_uav_acceleration
    return _norm_clip(acceleration, config.max_uav_acceleration)


class ActionAdapter:
    """Add a bounded learned acceleration residual to a goal controller."""

    def __init__(self, config, control_mode="pure", residual_scale=.25):
        if control_mode not in {"pure", "residual"}:
            raise ValueError("control_mode must be 'pure' or 'residual'")
        if not 0 <= residual_scale <= 1:
            raise ValueError("residual_scale must be in [0, 1]")
        self.config = config
        self.control_mode = control_mode
        self.residual_scale = float(residual_scale)

    def set_environment(self, config):
        """Update execution scales when evaluating or training in another scenario."""
        self.config = config

    def policy_action(self, raw, obs, active):
        """Map Gaussian samples to env raw actions without changing their log-prob."""
        raw = np.asarray(raw, dtype=np.float32)
        active = np.asarray(active, dtype=bool)
        if self.control_mode == "pure":
            return raw.copy()
        cfg = self.config
        base_raw = goal_raw_action(obs, active, cfg)
        base = executable_acceleration(base_raw, cfg)
        acceleration = base + self.residual_scale * np.tanh(raw) * cfg.max_uav_acceleration
        acceleration = _norm_clip(acceleration, cfg.max_uav_acceleration)
        # Only protect arctanh's domain; do not shrink Goal's existing .999 command.
        scaled = np.clip(acceleration / cfg.max_uav_acceleration, -1 + 1.e-7, 1 - 1.e-7)
        action = np.arctanh(scaled).astype(np.float32)
        action[~active] = 0.
        return action
