"""Observation-only path probes for sensing controllability diagnostics."""

import numpy as np

from .control import goal_raw_action


PATH_NAMES = ("goal", "lateral_plus", "lateral_minus", "altitude_plus", "altitude_minus",
              "altitude_split", "altitude_split_reverse")


def waypoint_for(config, path_name):
    """Return one mild intermediate waypoint per UAV; the terminal goal is unchanged."""
    if path_name == "goal":
        return None
    if path_name not in PATH_NAMES:
        raise ValueError(f"unknown diagnostic path: {path_name}")
    midpoint = (config.uav_initial + config.uav_goal_positions) / 2.0
    offsets = np.zeros((2, 3))
    if path_name.startswith("lateral"):
        sign = 1.0 if path_name.endswith("plus") else -1.0
        offsets[:, 1] = sign * 12.0
    elif path_name.startswith("altitude_split"):
        signs = np.array([1., -1.])
        if path_name.endswith("reverse"):
            signs *= -1.
        offsets[:, 2] = signs * 10.0
    else:
        sign = 1.0 if path_name.endswith("plus") else -1.0
        offsets[:, 2] = sign * 10.0
    return np.clip(midpoint + offsets, config.world_low, config.world_high)


class WaypointController:
    """Apply the shared goal action to a virtual waypoint encoded in copied observations."""

    def __init__(self, config, path_name="goal", switch_radius=8.0):
        self.config = config
        self.path_name = path_name
        self.waypoint = waypoint_for(config, path_name)
        self.switch_radius = float(switch_radius)
        self.reset()

    def reset(self, seed=None):
        del seed
        self.using_waypoint = np.full(2, self.waypoint is not None, dtype=bool)

    def act(self, obs, active, deterministic=True):
        del deterministic
        virtual_obs = np.asarray(obs).copy()
        position = virtual_obs[:, :3] * self.config.position_scale
        if self.waypoint is not None:
            reached = np.linalg.norm(position - self.waypoint, axis=1) <= self.switch_radius
            self.using_waypoint &= ~reached
            virtual_obs[self.using_waypoint, 6:9] = (
                (self.waypoint[self.using_waypoint] - position[self.using_waypoint])
                / self.config.position_scale)
        return goal_raw_action(virtual_obs, active, self.config), np.zeros(2, dtype=np.float32)
