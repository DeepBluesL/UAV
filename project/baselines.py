"""只依赖局部观测的轻量导航基线。"""

import numpy as np


def observation_state(obs, config):
    """从公开观测恢复两机的导航量。"""
    obs = np.asarray(obs, dtype=float)
    if obs.shape != (2, 31):
        raise ValueError("obs must have shape (2, 31)")
    position = obs[:, :3] * config.position_scale
    velocity = obs[:, 3:6] * config.velocity_scale
    goal_delta = obs[:, 6:9] * config.position_scale
    teammate_position = position + obs[:, 19:22] * config.position_scale
    teammate_velocity = velocity + obs[:, 22:25] * config.velocity_scale
    return position, velocity, goal_delta, teammate_position, teammate_velocity


def acceleration_to_action(acceleration, active, config):
    """按环境的 tanh 和范数约束逆映射物理加速度。"""
    acceleration = np.asarray(acceleration, dtype=float).copy()
    active = np.asarray(active, dtype=bool)
    if acceleration.shape != (2, 3) or active.shape != (2,):
        raise ValueError("expected acceleration (2,3) and active (2,)")
    acceleration[~active] = 0.0
    norms = np.linalg.norm(acceleration, axis=1, keepdims=True)
    acceleration /= np.maximum(1.0, norms / config.max_uav_acceleration)
    scaled = np.clip(acceleration / config.max_uav_acceleration, -0.999, 0.999)
    action = np.arctanh(scaled).astype(np.float32)
    action[~active] = 0.0
    return action, np.zeros(2, dtype=np.float32)


class RandomController:
    """可按回合重置的均匀随机动作基线。"""

    DEFAULT_SETTINGS = {"seed": 0, "raw_action_limit": 1.0}

    def __init__(self, config, seed=0, raw_action_limit=1.0):
        self.config = config
        self.seed = int(seed)
        self.raw_action_limit = float(raw_action_limit)
        self.settings = {"seed": self.seed, "raw_action_limit": self.raw_action_limit}
        self.reset(self.seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            self.settings["seed"] = self.seed
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs, active, deterministic=True):
        del obs, deterministic
        action = self.rng.uniform(-self.raw_action_limit, self.raw_action_limit, (2, 3))
        action[~np.asarray(active, dtype=bool)] = 0.0
        return action.astype(np.float32), np.zeros(2, dtype=np.float32)


class PDController:
    """目标速度随剩余距离平滑下降的 PD 导航。"""

    DEFAULT_SETTINGS = {"kp": 1.0, "kd": 1.5, "slow_radius": 15.0}

    def __init__(self, config, kp=1.0, kd=1.5, slow_radius=15.0):
        self.config = config
        self.kp, self.kd, self.slow_radius = float(kp), float(kd), float(slow_radius)
        self.settings = {"kp": self.kp, "kd": self.kd, "slow_radius": self.slow_radius}

    def reset(self, seed=None):
        del seed

    def desired_acceleration(self, obs):
        _, velocity, goal, _, _ = observation_state(obs, self.config)
        distance = np.linalg.norm(goal, axis=1, keepdims=True)
        direction = goal / np.maximum(distance, 1e-12)
        desired_speed = self.config.max_uav_speed * np.minimum(1.0, distance / self.slow_radius)
        return self.kp * goal + self.kd * (direction * desired_speed - velocity)

    def act(self, obs, active, deterministic=True):
        del deterministic
        return acceleration_to_action(self.desired_acceleration(obs), active, self.config)


class PotentialFieldController(PDController):
    """目标吸引、队友与边界排斥的人工势场基线。"""

    DEFAULT_SETTINGS = {
        **PDController.DEFAULT_SETTINGS, "teammate_gain": 18.0,
        "influence_radius": 12.0, "boundary_gain": 10.0, "boundary_margin": 10.0,
    }

    def __init__(self, config, kp=1.0, kd=1.5, slow_radius=15.0,
                 teammate_gain=18.0, influence_radius=12.0,
                 boundary_gain=10.0, boundary_margin=10.0):
        super().__init__(config, kp, kd, slow_radius)
        self.teammate_gain = float(teammate_gain)
        self.influence_radius = float(influence_radius)
        self.boundary_gain = float(boundary_gain)
        self.boundary_margin = float(boundary_margin)
        self.settings.update(teammate_gain=self.teammate_gain,
                             influence_radius=self.influence_radius,
                             boundary_gain=self.boundary_gain,
                             boundary_margin=self.boundary_margin)

    def desired_acceleration(self, obs):
        acceleration = super().desired_acceleration(obs)
        position, velocity, _, teammate, teammate_velocity = observation_state(obs, self.config)
        relative = position - teammate
        distance = np.linalg.norm(relative, axis=1, keepdims=True)
        closing = np.sum((velocity - teammate_velocity) * relative, axis=1, keepdims=True) < 0
        effective = np.maximum(distance - self.config.min_uav_distance, 0.25)
        strength = np.maximum(0.0, 1.0 / effective - 1.0 / self.influence_radius)
        strength *= self.teammate_gain * (1.0 + closing)
        acceleration += strength * relative / np.maximum(distance, 1e-12)
        low_gap = position - self.config.world_low
        high_gap = self.config.world_high - position
        acceleration += self.boundary_gain * np.maximum(0.0, 1.0 / np.maximum(low_gap, .25)
                                                       - 1.0 / self.boundary_margin)
        acceleration -= self.boundary_gain * np.maximum(0.0, 1.0 / np.maximum(high_gap, .25)
                                                       - 1.0 / self.boundary_margin)
        return acceleration
