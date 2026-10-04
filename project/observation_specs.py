"""Versioned public observation and centralized-state layouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationSpec:
    version: str
    obs_dim: int
    state_dim: int
    observation_fields: tuple[str, ...]
    state_fields: tuple[str, ...]


V1 = ObservationSpec(
    "v1", 31, 58,
    ("legacy_public_observation_0_30",),
    ("legacy_centralized_state_0_57",),
)

V2 = ObservationSpec(
    "v2", 67, 72,
    V1.observation_fields + (
        "velocity_log_variances_31_33", "belief_correlations_34_48",
        "remaining_seconds_49", "known_motion_limits_50_52",
        "signed_boundary_distances_53_58", "bs_relative_position_59_61",
        "delivered_source_mask_62_64", "teammate_comm_sinr_65",
        "known_gamma_min_66",
    ),
    V1.state_fields + (
        "remaining_seconds_58", "known_motion_limits_59_61",
        "world_bounds_62_67", "bs_position_68_70", "known_gamma_min_71",
    ),
)

SPECS = {spec.version: spec for spec in (V1, V2)}


def observation_spec(config_or_version) -> ObservationSpec:
    version = getattr(config_or_version, "observation_version", config_or_version)
    try:
        return SPECS[version]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unsupported observation version: {version!r}") from error
