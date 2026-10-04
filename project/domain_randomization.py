"""Declared domain-randomization ranges for robust-policy training only."""

from dataclasses import replace

import numpy as np


DOMAIN_RANGES = {
    "max_uav_speed": (4.0, 7.0),
    "max_uav_acceleration": (3.0, 6.0),
    "world_low": ([-35., -95., 10.], [-20., -80., 20.]),
    "world_high": ([155., 85., 125.], [190., 110., 145.]),
    "target_position": ([50., -12., 50.], [75., 12., 70.]),
    "target_velocity": ([-1.3, .5, -1.3], [-.5, 1.3, -.5]),
    "target_acceleration": ([-.03, -.03, -.03], [.03, .03, .03]),
    "measurement_range_std_floor": (.8, 1.2),
    "measurement_angle_std_floor_deg": (.4, .7),
    "measurement_range_rate_std_floor": (.4, .7),
    "gamma_min": (2.0, 5.0),
    "imperfect_csi_beta": (.05, .31),
}


def domain_randomize(config, rng):
    """Apply the documented physical ranges to an already sampled geometry."""
    limits = DOMAIN_RANGES
    speed = float(rng.uniform(*limits["max_uav_speed"]))
    acceleration = float(rng.uniform(*limits["max_uav_acceleration"]))
    world_low = rng.uniform(*limits["world_low"])
    world_high = rng.uniform(*limits["world_high"])
    target = np.r_[rng.uniform(*limits["target_position"]),
                   rng.uniform(*limits["target_velocity"])]
    target_acceleration = rng.uniform(*limits["target_acceleration"])
    distances = np.linalg.norm(config.uav_goal_positions - config.uav_initial, axis=1)
    lower_bound = int(np.max(np.ceil(
        np.maximum(0., distances - config.goal_tolerance)
        / (speed * config.slot_duration))))
    return replace(
        config,
        max_uav_speed=speed,
        max_uav_acceleration=acceleration,
        world_low=world_low,
        world_high=world_high,
        target_initial_state=target,
        target_acceleration=target_acceleration,
        measurement_range_std_floor=float(rng.uniform(
            *limits["measurement_range_std_floor"])),
        measurement_angle_std_floor=float(np.deg2rad(rng.uniform(
            *limits["measurement_angle_std_floor_deg"]))),
        measurement_range_rate_std_floor=float(rng.uniform(
            *limits["measurement_range_rate_std_floor"])),
        gamma_min=float(rng.uniform(*limits["gamma_min"])),
        imperfect_csi_beta=float(rng.uniform(*limits["imperfect_csi_beta"])),
        max_steps=max(config.max_steps, lower_bound + 10),
    )
