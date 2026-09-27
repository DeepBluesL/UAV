"""C-HAPPO style multi-agent RL for dual-UAV cooperative ISAC.

This module keeps the paper-formula implementation in ``test.py`` as the
physical layer and replaces the Q-table policy with continuous PPO/HAPPO
actors:

* UAV1 and UAV2 have separate trajectory actors.
* The BS has a heterogeneous beamforming actor.
* A centralized critic is used only during training.
* BS beam actions are reconstructed with Kronecker factors and QR
  orthonormalization, then normalized to the BS power budget.
* Imperfect CSI is modeled both as noisy observations and as beam/channel
  perturbation controlled by a PrecoderNet-like beta parameter.
* Curriculum stages gradually increase SINR threshold, process noise,
  CRB scale, and CSI imperfection.

PyTorch is imported lazily so the environment and tests can run on systems
where only NumPy is installed. Training requires ``torch``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

# Conda builds of NumPy/MKL and PyTorch can load two Intel OpenMP runtimes on
# Windows. Configure the runtime before importing numeric libraries.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from test import (
    bs_sensing_sinr_eq6_parts,
    ci_fusion_crb,
    crb_from_sensing_sinr,
    db_to_linear,
    dbm_to_watt,
    make_matched_beams,
    measurement_noise_cov_from_fused_crb,
    pcrb_eq24_to_eq28_parts,
    safe_inverse,
    uav_comm_sinr_eq10_parts,
    uav_sensing_sinr_eq12_parts,
)


EPS = 1.0e-9
AGENT_UAV_0 = "uav_0"
AGENT_UAV_1 = "uav_1"
AGENT_BS = "bs"
AGENT_NAMES = (AGENT_UAV_0, AGENT_UAV_1, AGENT_BS)


def _as_float_array(values: Any, shape: Tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}.")
    return arr


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / max(float(np.sum(exp_values)), EPS)


def _safe_log1p(values: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(np.maximum(values, 0.0))


def _normalize_total_power(beam_matrix: np.ndarray, total_power: float) -> np.ndarray:
    power = float(np.real(np.sum(np.abs(beam_matrix) ** 2)))
    if not np.isfinite(power) or power <= EPS:
        raise ValueError("Beam matrix has zero or non-finite power.")
    return beam_matrix * np.sqrt(total_power / power)


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    fraction: float
    gamma_min: float
    csi_beta: float
    kappa_scale: float
    process_noise_intensity: float


@dataclass
class HybridBeamformingConfig:
    """Environment and curriculum settings for two-UAV cooperative ISAC."""

    max_steps: int = 100
    slot_duration: float = 1.0
    max_uav_speed: float = 5.0
    max_uav_acceleration: float = 5.0
    min_uav_distance: float = 5.0

    bs_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    uav_initial: np.ndarray = field(
        default_factory=lambda: np.array(
            [[30.0, 10.0, 70.0], [30.0, -10.0, 70.0]], dtype=float
        )
    )
    uav_initial_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 3), dtype=float)
    )
    uav_goal_positions: np.ndarray = field(
        default_factory=lambda: np.array(
        [[150.0, 70.0, 90.0], [120.0, -30.0,30.0]], dtype=float
        )
    )
    target_initial_state: np.ndarray = field(
        default_factory=lambda: np.array(
            [60.0, 0.0, 60.0, -1.0, 1.0, -1.0], dtype=float
        )
    )
    target_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([0.02, -0.02, 0.02], dtype=float)
    )
    world_low: np.ndarray = field(
        default_factory=lambda: np.array([-20.0, -80.0, 20.0], dtype=float)
    )
    world_high: np.ndarray = field(
        default_factory=lambda: np.array([150.0, 80.0, 120.0], dtype=float)
    )

    antenna_x: int = 8
    antenna_y: int = 8
    rf_chains: int = 3
    bs_action_rank: int = 2
    total_power: float = 5.0
    bandwidth: float = 100.0e6
    rician_k: float = 10.0
    beta0_c: float = field(default_factory=lambda: db_to_linear(-50.0))
    beta0_s: float = field(default_factory=lambda: db_to_linear(-50.0))
    alpha_target: float = 0.9
    alpha_uav: float = 0.9
    awgn_power: float = field(default_factory=lambda: dbm_to_watt(-80.0))
    residual_noise: float = field(default_factory=lambda: dbm_to_watt(-70.0))
    gamma_min: float = field(default_factory=lambda: 10.0 ** (5.0 / 10.0))

    kappa_d: float = 1.0
    kappa_theta: float = 1.0e-6
    kappa_phi: float = 1.0e-6
    kappa_v: float = 1.0e-6
    process_noise_intensity: float = 0.1
    initial_covariance: float = 1.0

    imperfect_csi_beta: float = 0.10
    csi_position_std: float = 10.0
    csi_velocity_std: float = 1.0
    csi_beta_jitter: float = 0.02

    curriculum_stages: int = 2
    gamma_min_start: float = field(default_factory=lambda: 10.0 ** (1.0 / 10.0))
    csi_beta_start: float = 0.0
    kappa_scale_start: float = 0.2
    process_noise_start: float = 0.02
    curriculum_advance_every: int = 25
    curriculum_reward_threshold: float | None = -800.0

    goal_progress_weight: float = 10.0
    goal_distance_weight: float = 3.0
    goal_bonus: float = 80.0
    goal_tolerance: float = 5.0
    terminate_on_goal: bool = True
    tracking_weight: float = 0.15
    pcrb_weight: float = 0.5
    sensing_weight: float = 0.05
    spectral_efficiency_weight: float = 0.05
    communication_penalty: float = 2.0
    collision_penalty: float = 4.0
    boundary_penalty: float = 0.5
    energy_weight: float = 0.02

    position_scale: float = 150.0
    velocity_scale: float = 10.0
    sinr_log_scale: float = 10.0
    rho_log_scale: float = 10.0

    def __post_init__(self) -> None:
        self.bs_position = _as_float_array(self.bs_position, (3,))
        self.uav_initial = _as_float_array(self.uav_initial, (2, 3))
        self.uav_initial_velocities = _as_float_array(
            self.uav_initial_velocities, (2, 3)
        )
        self.uav_goal_positions = _as_float_array(self.uav_goal_positions, (2, 3))
        self.target_initial_state = _as_float_array(self.target_initial_state, (6,))
        self.target_acceleration = _as_float_array(self.target_acceleration, (3,))
        self.world_low = _as_float_array(self.world_low, (3,))
        self.world_high = _as_float_array(self.world_high, (3,))

        if np.any(self.world_low >= self.world_high):
            raise ValueError("world_low must be smaller than world_high.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.slot_duration <= 0.0 or self.max_uav_speed <= 0.0:
            raise ValueError("slot_duration and max_uav_speed must be positive.")
        if self.max_uav_acceleration <= 0.0:
            raise ValueError("max_uav_acceleration must be positive.")
        if self.min_uav_distance <= 0.0:
            raise ValueError("min_uav_distance must be positive.")
        if self.antenna_x <= 0 or self.antenna_y <= 0:
            raise ValueError("antenna_x and antenna_y must be positive.")
        if self.rf_chains < self.num_beams:
            raise ValueError("rf_chains must be at least the number of beams.")
        if self.bs_action_rank <= 0:
            raise ValueError("bs_action_rank must be positive.")
        if self.total_power <= 0.0:
            raise ValueError("total_power must be positive.")
        if self.curriculum_stages <= 0:
            raise ValueError("curriculum_stages must be positive.")
        if self.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be positive.")
        if not (0.0 <= self.imperfect_csi_beta <= 1.0):
            raise ValueError("imperfect_csi_beta must be in [0, 1].")

    @property
    def num_uavs(self) -> int:
        return 2

    @property
    def num_beams(self) -> int:
        return self.num_uavs + 1

    @property
    def num_antennas(self) -> int:
        return self.antenna_x * self.antenna_y

    @property
    def max_displacement(self) -> float:
        return self.max_uav_speed * self.slot_duration

    @property
    def bs_vbb_action_dim(self) -> int:
        return 2 * self.rf_chains * self.num_beams

    @property
    def bs_factor_action_dim(self) -> int:
        complex_factor_dim = 2 * (self.antenna_x + self.antenna_y)
        return self.rf_chains * self.bs_action_rank * complex_factor_dim

    @property
    def bs_action_dim(self) -> int:
        return self.bs_vbb_action_dim + self.bs_factor_action_dim + self.num_beams

    @property
    def bs_hbf_state_dim(self) -> int:
        return 2 * (
            self.rf_chains * self.num_beams + self.num_antennas * self.rf_chains
        )

    @property
    def bs_observation_dim(self) -> int:
        return self.global_state_dim + self.bs_hbf_state_dim

    @property
    def uav_action_dim(self) -> int:
        return 3

    @property
    def global_state_dim(self) -> int:
        return 6 + 6 + 6 + 3 + 3 + 2 + 3 + 1 + 2 + 2

    @property
    def uav_observation_dim(self) -> int:
        return 3 + 3 + 3 + 3 + 3 + 3 + 2 + 1 + 2

    def make_stage(self, index: int) -> CurriculumStage:
        index = int(np.clip(index, 0, self.curriculum_stages - 1))
        if self.curriculum_stages == 1:
            fraction = 1.0
        else:
            fraction = index / float(self.curriculum_stages - 1)

        gamma_min = self.gamma_min_start * (
            self.gamma_min / max(self.gamma_min_start, EPS)
        ) ** fraction
        csi_beta = (
            self.csi_beta_start
            + fraction * (self.imperfect_csi_beta - self.csi_beta_start)
        )
        kappa_scale = (
            self.kappa_scale_start + fraction * (1.0 - self.kappa_scale_start)
        )
        process_noise = (
            self.process_noise_start
            + fraction * (self.process_noise_intensity - self.process_noise_start)
        )
        return CurriculumStage(
            index=index,
            fraction=float(fraction),
            gamma_min=float(gamma_min),
            csi_beta=float(np.clip(csi_beta, 0.0, 1.0)),
            kappa_scale=float(max(kappa_scale, EPS)),
            process_noise_intensity=float(max(process_noise, EPS)),
        )


@dataclass(frozen=True)
class HybridStepInfo:
    t: int
    stage: CurriculumStage
    uav_positions: np.ndarray
    uav_velocities: np.ndarray
    uav_accelerations: np.ndarray
    target_state: np.ndarray
    estimated_target_state: np.ndarray
    intended_beam_matrix: np.ndarray
    actual_beam_matrix: np.ndarray
    communication_sinrs: np.ndarray
    sensing_sinrs: np.ndarray
    fused_crb: np.ndarray
    rho: float
    mean_target_distance: float
    goal_distances: np.ndarray
    mean_goal_distance: float
    goal_success: bool
    uav_distance: float
    communication_ok: np.ndarray
    collision: bool
    boundary_violations: int
    power: float
    spectral_efficiency: float
    reward_parts: Dict[str, float]


class HybridCsiDualUAVEnv:
    """Two UAVs, one BS, and one rogue UAV target with continuous actions."""

    agent_names = AGENT_NAMES

    def __init__(
        self,
        config: HybridBeamformingConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or HybridBeamformingConfig()
        self.rng = np.random.default_rng(seed)
        self.stage = self.config.make_stage(0)
        self.reset(self.stage)

    def observation_dims(self) -> Dict[str, int]:
        return {
            AGENT_UAV_0: self.config.uav_observation_dim,
            AGENT_UAV_1: self.config.uav_observation_dim,
            AGENT_BS: self.config.bs_observation_dim,
        }

    def action_dims(self) -> Dict[str, int]:
        return {
            AGENT_UAV_0: self.config.uav_action_dim,
            AGENT_UAV_1: self.config.uav_action_dim,
            AGENT_BS: self.config.bs_action_dim,
        }

    def zero_actions(self) -> Dict[str, np.ndarray]:
        return {name: np.zeros(dim, dtype=float) for name, dim in self.action_dims().items()}

    def reset(
        self, stage: CurriculumStage | None = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        self.stage = stage or self.stage
        self.t = 0
        self.uav_positions = self.config.uav_initial.copy()
        self.uav_velocities = self.config.uav_initial_velocities.copy()
        self.target_state = self.config.target_initial_state.copy()
        self.estimated_target_state = self._make_noisy_target_estimate()

        p0 = self.config.initial_covariance * np.eye(6, dtype=float)
        self.j_prev = safe_inverse(p0)
        self.last_rho = float(np.trace(p0))

        initial_beams = self._matched_beam_matrix(self.estimated_target_state[:3])
        self._set_hbf_from_beams(initial_beams)
        initial_beams = self._apply_csi_beam_error(initial_beams)
        initial_metrics = self._paper_link_metrics(
            self.target_state[:3], initial_beams
        )
        self.last_communication_sinrs = initial_metrics["communication_sinrs"]
        self.last_sensing_sinrs = initial_metrics["sensing_sinrs"]
        return self._observations(), self.global_state()

    def step(
        self, actions: Mapping[str, Sequence[float]]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, float, bool, HybridStepInfo]:
        action_dict = self._normalize_actions(actions)
        previous_positions = self.uav_positions.copy()
        previous_goal_distances = np.linalg.norm(
            previous_positions - self.config.uav_goal_positions, axis=1
        )

        requested_positions = self.uav_positions.copy()
        uav_accelerations = np.zeros_like(self.uav_velocities)
        for agent_id, name in enumerate((AGENT_UAV_0, AGENT_UAV_1)):
            acceleration = self._uav_acceleration_from_action(action_dict[name])
            uav_accelerations[agent_id] = acceleration
            velocity = self.uav_velocities[agent_id] + acceleration * self.config.slot_duration
            velocity = self._clip_velocity(velocity)
            requested_positions[agent_id] = (
                self.uav_positions[agent_id] + velocity * self.config.slot_duration
            )

        next_positions = np.clip(
            requested_positions, self.config.world_low, self.config.world_high
        )
        boundary_violations = int(
            np.count_nonzero(np.any(next_positions != requested_positions, axis=1))
        )

        uav_distance_candidate = float(np.linalg.norm(next_positions[0] - next_positions[1]))
        collision = uav_distance_candidate < self.config.min_uav_distance
        if collision:
            next_positions = self.uav_positions.copy()

        self.uav_positions = next_positions
        self.uav_velocities = (
            self.uav_positions - previous_positions
        ) / self.config.slot_duration
        uav_velocities = self.uav_velocities.copy()
        uav_speeds = np.linalg.norm(uav_velocities, axis=1)
        acceleration_norms = np.linalg.norm(uav_accelerations, axis=1)

        target_state_before_step = self.target_state.copy()
        self.target_state = self._advance_target_state(target_state_before_step)
        self.estimated_target_state = self._make_noisy_target_estimate()

        intended_beams = self.decode_bs_action(action_dict[AGENT_BS])
        actual_beams = self._apply_csi_beam_error(intended_beams)
        metrics = self._paper_link_metrics(self.target_state[:3], actual_beams)

        crbs = [
            crb_from_sensing_sinr(
                sensing_sinr=max(float(sinr), EPS),
                bandwidth=self.config.bandwidth,
                kappa_d=self.config.kappa_d * self.stage.kappa_scale,
                kappa_theta=self.config.kappa_theta * self.stage.kappa_scale,
                kappa_phi=self.config.kappa_phi * self.stage.kappa_scale,
                kappa_v=self.config.kappa_v * self.stage.kappa_scale,
            )
            for sinr in metrics["sensing_sinrs"]
        ]
        fused_crb = ci_fusion_crb(crbs)
        psi = measurement_noise_cov_from_fused_crb(fused_crb)
        pcrb_parts = pcrb_eq24_to_eq28_parts(
            zeta_prev=target_state_before_step,
            J_prev=self.j_prev,
            Psi=psi,
            dt=self.config.slot_duration,
            sigma2_zeta=self.stage.process_noise_intensity,
            use_inverse_covariance=True,
        )
        self.j_prev = pcrb_parts["J"]
        rho = float(pcrb_parts["rho"])

        communication_sinrs = metrics["communication_sinrs"]
        sensing_sinrs = metrics["sensing_sinrs"]
        communication_ok = communication_sinrs >= self.stage.gamma_min
        target_distances = np.linalg.norm(
            self.uav_positions - self.target_state[:3], axis=1
        )
        mean_target_distance = float(np.mean(target_distances))
        goal_distances = np.linalg.norm(
            self.uav_positions - self.config.uav_goal_positions, axis=1
        )
        mean_goal_distance = float(np.mean(goal_distances))
        goal_progress = float(np.mean(previous_goal_distances) - mean_goal_distance)
        goal_success = bool(np.all(goal_distances <= self.config.goal_tolerance))
        uav_distance = float(np.linalg.norm(self.uav_positions[0] - self.uav_positions[1]))
        power = float(np.real(np.sum(np.abs(actual_beams) ** 2)))
        speed_energy_proxy = float(np.sum((uav_speeds / self.config.max_uav_speed) ** 2))
        acceleration_energy_proxy = float(
            np.sum((acceleration_norms / self.config.max_uav_acceleration) ** 2)
        )
        propulsion_energy_proxy = speed_energy_proxy + 0.5 * acceleration_energy_proxy
        spectral_efficiency = float(np.sum(np.log2(1.0 + np.maximum(communication_sinrs, 0.0))))

        reward_parts = self._reward(
            rho=rho,
            mean_target_distance=mean_target_distance,
            goal_progress=goal_progress,
            mean_goal_distance=mean_goal_distance,
            goal_success=goal_success,
            sensing_sinrs=sensing_sinrs,
            communication_sinrs=communication_sinrs,
            communication_ok=communication_ok,
            collision=collision,
            boundary_violations=boundary_violations,
            propulsion_energy_proxy=propulsion_energy_proxy,
            spectral_efficiency=spectral_efficiency,
        )
        reward = float(sum(reward_parts.values()))

        self.t += 1
        self.last_rho = rho
        self.last_communication_sinrs = communication_sinrs
        self.last_sensing_sinrs = sensing_sinrs
        done = self.t >= self.config.max_steps or (
            self.config.terminate_on_goal and goal_success
        )

        info = HybridStepInfo(
            t=self.t,
            stage=self.stage,
            uav_positions=self.uav_positions.copy(),
            uav_velocities=uav_velocities.copy(),
            uav_accelerations=uav_accelerations.copy(),
            target_state=self.target_state.copy(),
            estimated_target_state=self.estimated_target_state.copy(),
            intended_beam_matrix=intended_beams.copy(),
            actual_beam_matrix=actual_beams.copy(),
            communication_sinrs=communication_sinrs.copy(),
            sensing_sinrs=sensing_sinrs.copy(),
            fused_crb=fused_crb.copy(),
            rho=rho,
            mean_target_distance=mean_target_distance,
            goal_distances=goal_distances.copy(),
            mean_goal_distance=mean_goal_distance,
            goal_success=goal_success,
            uav_distance=uav_distance,
            communication_ok=communication_ok.copy(),
            collision=collision,
            boundary_violations=boundary_violations,
            power=power,
            spectral_efficiency=spectral_efficiency,
            reward_parts=reward_parts,
        )
        return self._observations(), self.global_state(), reward, done, info

    def decode_bs_action(self, raw_action: Sequence[float]) -> np.ndarray:
        raw = np.asarray(raw_action, dtype=float).reshape(-1)
        if raw.size != self.config.bs_action_dim:
            raise ValueError(
                f"BS action has size {raw.size}; expected {self.config.bs_action_dim}."
            )

        if float(np.linalg.norm(raw)) <= EPS:
            beams = self._matched_beam_matrix(self.estimated_target_state[:3])
            self._set_hbf_from_beams(beams)
            return beams

        cursor = 0
        vbb_values = np.tanh(raw[cursor : cursor + self.config.bs_vbb_action_dim])
        cursor += self.config.bs_vbb_action_dim
        rf_values = raw[cursor : cursor + self.config.bs_factor_action_dim]
        cursor += self.config.bs_factor_action_dim
        power_logits = raw[cursor : cursor + self.config.num_beams]

        v_bb = self._complex_from_action(
            vbb_values, (self.config.rf_chains, self.config.num_beams)
        )
        v_rf = self._rf_from_kronecker_action(rf_values)
        beam_matrix = v_rf @ v_bb
        if not np.all(np.isfinite(beam_matrix)) or np.linalg.norm(beam_matrix) <= EPS:
            beams = self._matched_beam_matrix(self.estimated_target_state[:3])
            self._set_hbf_from_beams(beams)
            return beams

        try:
            q_matrix, _ = np.linalg.qr(beam_matrix)
        except np.linalg.LinAlgError:
            beams = self._matched_beam_matrix(self.estimated_target_state[:3])
            self._set_hbf_from_beams(beams)
            return beams

        powers = self.config.total_power * _softmax(power_logits)
        beam_matrix = q_matrix[:, : self.config.num_beams] * np.sqrt(powers)[None, :]
        beam_matrix = _normalize_total_power(beam_matrix, self.config.total_power)
        self.last_v_rf = v_rf.copy()
        self.last_v_bb = v_bb.copy()
        return beam_matrix

    def global_state(self) -> np.ndarray:
        cfg = self.config
        goal_distances = np.linalg.norm(
            self.uav_positions - cfg.uav_goal_positions, axis=1
        )
        values = [
            self.uav_positions.reshape(-1) / cfg.position_scale,
            self.uav_velocities.reshape(-1) / cfg.velocity_scale,
            cfg.uav_goal_positions.reshape(-1) / cfg.position_scale,
            self.estimated_target_state[:3] / cfg.position_scale,
            self.estimated_target_state[3:] / cfg.velocity_scale,
            _safe_log1p(self.last_communication_sinrs) / cfg.sinr_log_scale,
            _safe_log1p(self.last_sensing_sinrs) / cfg.sinr_log_scale,
            np.array([_safe_log1p(self.last_rho) / cfg.rho_log_scale]),
            np.array([self.stage.csi_beta, self.stage.fraction]),
            goal_distances / cfg.position_scale,
        ]
        return np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in values])

    def _observations(self) -> Dict[str, np.ndarray]:
        cfg = self.config
        observations: Dict[str, np.ndarray] = {}
        for agent_id, name in enumerate((AGENT_UAV_0, AGENT_UAV_1)):
            other_id = 1 - agent_id
            own_pos = self.uav_positions[agent_id] / cfg.position_scale
            own_vel = self.uav_velocities[agent_id] / cfg.velocity_scale
            goal_rel = (
                cfg.uav_goal_positions[agent_id] - self.uav_positions[agent_id]
            ) / cfg.position_scale
            teammate_rel = (
                self.uav_positions[other_id] - self.uav_positions[agent_id]
            ) / cfg.position_scale
            target_rel = (
                self.estimated_target_state[:3] - self.uav_positions[agent_id]
            ) / cfg.position_scale
            target_vel = self.estimated_target_state[3:] / cfg.velocity_scale
            comm = _safe_log1p(self.last_communication_sinrs) / cfg.sinr_log_scale
            rho = np.array([_safe_log1p(self.last_rho) / cfg.rho_log_scale])
            robust = np.array([self.stage.csi_beta, self.stage.fraction])
            observations[name] = np.concatenate(
                [
                    own_pos,
                    own_vel,
                    goal_rel,
                    teammate_rel,
                    target_rel,
                    target_vel,
                    comm,
                    rho,
                    robust,
                ]
            ).astype(np.float32)

        observations[AGENT_BS] = np.concatenate(
            [self.global_state(), self._hbf_state()]
        ).astype(np.float32)
        return observations

    def _normalize_actions(
        self, actions: Mapping[str, Sequence[float]]
    ) -> Dict[str, np.ndarray]:
        dims = self.action_dims()
        normalized: Dict[str, np.ndarray] = {}
        for name in self.agent_names:
            if name not in actions:
                raise ValueError(f"Missing action for agent {name}.")
            arr = np.asarray(actions[name], dtype=float).reshape(-1)
            if arr.size != dims[name]:
                raise ValueError(
                    f"Action for {name} has size {arr.size}; expected {dims[name]}."
                )
            normalized[name] = arr
        return normalized

    def _uav_acceleration_from_action(self, raw_action: np.ndarray) -> np.ndarray:
        acceleration = np.tanh(raw_action) * self.config.max_uav_acceleration
        norm = float(np.linalg.norm(acceleration))
        if norm > self.config.max_uav_acceleration:
            acceleration = acceleration / norm * self.config.max_uav_acceleration
        return acceleration

    def _clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        if speed > self.config.max_uav_speed:
            return velocity / speed * self.config.max_uav_speed
        return velocity

    def _advance_target_state(self, target_state: np.ndarray) -> np.ndarray:
        dt = self.config.slot_duration
        next_state = target_state.copy()
        next_state[:3] = (
            target_state[:3]
            + target_state[3:] * dt
            + 0.5 * self.config.target_acceleration * dt**2
        )
        next_state[3:] = target_state[3:] + self.config.target_acceleration * dt
        return next_state

    def _make_noisy_target_estimate(self) -> np.ndarray:
        beta = float(np.clip(self.stage.csi_beta, 0.0, 1.0))
        if self.config.csi_beta_jitter > 0.0 and beta > 0.0:
            beta = float(
                np.clip(
                    beta + self.rng.normal(0.0, self.config.csi_beta_jitter),
                    0.0,
                    1.0,
                )
            )
        pos_std = beta * self.config.csi_position_std
        vel_std = beta * self.config.csi_velocity_std
        noise = np.concatenate(
            [
                self.rng.normal(0.0, pos_std, size=3),
                self.rng.normal(0.0, vel_std, size=3),
            ]
        )
        return self.target_state + noise


    def _complex_from_action(self, values: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        flat_size = int(np.prod(shape))
        real = values[:flat_size]
        imag = values[flat_size : 2 * flat_size]
        return (real + 1j * imag).reshape(shape)

    def _rf_from_kronecker_action(self, values: np.ndarray) -> np.ndarray:
        values = np.tanh(np.asarray(values, dtype=float))
        cursor = 0
        v_rf = np.zeros(
            (self.config.num_antennas, self.config.rf_chains), dtype=np.complex128
        )
        for chain_idx in range(self.config.rf_chains):
            column = np.zeros(self.config.num_antennas, dtype=np.complex128)
            for _ in range(self.config.bs_action_rank):
                ux_real = values[cursor : cursor + self.config.antenna_x]
                cursor += self.config.antenna_x
                ux_imag = values[cursor : cursor + self.config.antenna_x]
                cursor += self.config.antenna_x
                uy_real = values[cursor : cursor + self.config.antenna_y]
                cursor += self.config.antenna_y
                uy_imag = values[cursor : cursor + self.config.antenna_y]
                cursor += self.config.antenna_y
                ux = ux_real + 1j * ux_imag
                uy = uy_real + 1j * uy_imag
                if np.linalg.norm(ux) <= EPS or np.linalg.norm(uy) <= EPS:
                    continue
                column += np.kron(ux / np.linalg.norm(ux), uy / np.linalg.norm(uy))
            if np.linalg.norm(column) <= EPS:
                column = (
                    self.rng.standard_normal(self.config.num_antennas)
                    + 1j * self.rng.standard_normal(self.config.num_antennas)
                )
            v_rf[:, chain_idx] = np.exp(1j * np.angle(column)) / np.sqrt(
                self.config.num_antennas
            )
        return v_rf

    def _initialize_hbf_from_beams(self, beam_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v_rf = np.zeros(
            (self.config.num_antennas, self.config.rf_chains), dtype=np.complex128
        )
        for chain_idx in range(self.config.rf_chains):
            source = beam_matrix[:, chain_idx % self.config.num_beams]
            if np.linalg.norm(source) <= EPS:
                source = (
                    self.rng.standard_normal(self.config.num_antennas)
                    + 1j * self.rng.standard_normal(self.config.num_antennas)
                )
            v_rf[:, chain_idx] = np.exp(1j * np.angle(source.reshape(-1))) / np.sqrt(
                self.config.num_antennas
            )
        v_bb = np.linalg.pinv(v_rf) @ beam_matrix
        return v_rf, v_bb

    def _set_hbf_from_beams(self, beam_matrix: np.ndarray) -> None:
        self.last_v_rf, self.last_v_bb = self._initialize_hbf_from_beams(beam_matrix)

    def _hbf_state(self) -> np.ndarray:
        arrays = [self.last_v_bb.reshape(-1), self.last_v_rf.reshape(-1)]
        return np.concatenate([np.concatenate([arr.real, arr.imag]) for arr in arrays])

    def _matched_beam_matrix(self, target_position: np.ndarray) -> np.ndarray:
        w0, w_comm = make_matched_beams(
            u_bs=self.config.bs_position,
            uav_positions=self.uav_positions,
            target=target_position,
            Mx=self.config.antenna_x,
            My=self.config.antenna_y,
            P=self.config.total_power,
        )
        return np.hstack([*w_comm, w0])

    def _apply_csi_beam_error(self, beam_matrix: np.ndarray) -> np.ndarray:
        beta = float(np.clip(self.stage.csi_beta, 0.0, 1.0))
        if beta <= EPS:
            return _normalize_total_power(beam_matrix.copy(), self.config.total_power)

        noise = (
            self.rng.standard_normal(beam_matrix.shape)
            + 1j * self.rng.standard_normal(beam_matrix.shape)
        ) / np.sqrt(2.0)
        noise = _normalize_total_power(noise, self.config.total_power)
        perturbed = np.sqrt(max(1.0 - beta**2, 0.0)) * beam_matrix + beta * noise
        return _normalize_total_power(perturbed, self.config.total_power)

    def _split_beams(self, beam_matrix: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        w_comm = [
            beam_matrix[:, idx].reshape(-1, 1) for idx in range(self.config.num_uavs)
        ]
        w0 = beam_matrix[:, self.config.num_uavs].reshape(-1, 1)
        return w0, w_comm

    def _paper_link_metrics(
        self, target_position: np.ndarray, beam_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        cfg = self.config
        w0, w_comm = self._split_beams(beam_matrix)

        bs_sensing = bs_sensing_sinr_eq6_parts(
            u_bs=cfg.bs_position,
            target=target_position,
            W_comm=w_comm,
            w0=w0,
            Mx=cfg.antenna_x,
            My=cfg.antenna_y,
            beta0_s=cfg.beta0_s,
            K=cfg.rician_k,
            alpha0=cfg.alpha_target,
            sigma_ubs=cfg.residual_noise,
            sigma0=cfg.awgn_power,
            rng=self.rng,
        )["sinr"]

        communication_sinrs = np.array(
            [
                uav_comm_sinr_eq10_parts(
                    u_bs=cfg.bs_position,
                    uav_positions=self.uav_positions,
                    target=target_position,
                    n_idx=agent_id,
                    W_comm=w_comm,
                    w0=w0,
                    Mx=cfg.antenna_x,
                    My=cfg.antenna_y,
                    beta0_c=cfg.beta0_c,
                    beta0_s=cfg.beta0_s,
                    K=cfg.rician_k,
                    alpha0=cfg.alpha_target,
                    alpha1=cfg.alpha_uav,
                    sigma0=cfg.awgn_power,
                    rng=self.rng,
                    include_direct_sensing_beam=False,
                )["sinr"]
                for agent_id in range(cfg.num_uavs)
            ],
            dtype=float,
        )

        uav_sensing = np.array(
            [
                uav_sensing_sinr_eq12_parts(
                    u_bs=cfg.bs_position,
                    uav=uav_position,
                    target=target_position,
                    W_comm=w_comm,
                    w0=w0,
                    Mx=cfg.antenna_x,
                    My=cfg.antenna_y,
                    beta0_s=cfg.beta0_s,
                    K=cfg.rician_k,
                    alpha0=cfg.alpha_target,
                    sigma_un=cfg.residual_noise,
                    sigma0=cfg.awgn_power,
                    rng=self.rng,
                )["sinr"]
                for uav_position in self.uav_positions
            ],
            dtype=float,
        )

        return {
            "communication_sinrs": np.maximum(communication_sinrs, EPS),
            "sensing_sinrs": np.maximum(np.concatenate(([bs_sensing], uav_sensing)), EPS),
        }

    def _reward(
        self,
        rho: float,
        mean_target_distance: float,
        goal_progress: float,
        mean_goal_distance: float,
        goal_success: bool,
        sensing_sinrs: np.ndarray,
        communication_sinrs: np.ndarray,
        communication_ok: np.ndarray,
        collision: bool,
        boundary_violations: int,
        propulsion_energy_proxy: float,
        spectral_efficiency: float,
    ) -> Dict[str, float]:
        cfg = self.config
        comm_margin = communication_sinrs / max(self.stage.gamma_min, EPS) - 1.0
        comm_shortfall = np.maximum(0.0, -comm_margin)
        return {
            "goal_progress": cfg.goal_progress_weight
            * goal_progress
            / max(cfg.max_displacement, EPS),
            "goal_distance": -cfg.goal_distance_weight * mean_goal_distance / 100.0,
            "goal_bonus": cfg.goal_bonus if goal_success else 0.0,
            "tracking": -cfg.tracking_weight * mean_target_distance / 100.0,
            "pcrb": -cfg.pcrb_weight * float(_safe_log1p(rho)),
            "sensing": cfg.sensing_weight
            * float(np.tanh(np.mean(_safe_log1p(sensing_sinrs)))),
            "hbf_spectral_efficiency": cfg.spectral_efficiency_weight
            * spectral_efficiency,
            "communication": -cfg.communication_penalty * float(np.sum(comm_shortfall)),
            "energy": -cfg.energy_weight * propulsion_energy_proxy,
            "collision": -cfg.collision_penalty if collision else 0.0,
            "boundary": -cfg.boundary_penalty * float(boundary_violations),
            "comm_constraint_count": -0.25 * float(np.count_nonzero(~communication_ok)),
        }


class CurriculumScheduler:
    def __init__(self, config: HybridBeamformingConfig) -> None:
        self.config = config
        self.current_index = 0

    @property
    def stage(self) -> CurriculumStage:
        return self.config.make_stage(self.current_index)

    def maybe_advance(self, episode: int, recent_return: float | None = None) -> bool:
        if self.current_index >= self.config.curriculum_stages - 1:
            return False

        threshold = self.config.curriculum_reward_threshold
        if threshold is not None and recent_return is not None and recent_return > threshold:
            self.current_index += 1
            return True

        every = self.config.curriculum_advance_every
        if every > 0 and episode > 0 and episode % every == 0:
            self.current_index += 1
            return True

        return False


@dataclass
class CHAPPOTrainingConfig:
    episodes: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    hidden_sizes: Tuple[int, ...] = (512, 512)
    update_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 7
    device: str | None = None


@dataclass
class CHAPPOHistory:
    episode_returns: List[float] = field(default_factory=list)
    final_rhos: List[float] = field(default_factory=list)
    final_distances: List[float] = field(default_factory=list)
    final_goal_distances: List[float] = field(default_factory=list)
    communication_success_rates: List[float] = field(default_factory=list)
    goal_success_rates: List[float] = field(default_factory=list)
    curriculum_indices: List[int] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.distributions import Normal
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "C-HAPPO training requires PyTorch. Install dependencies with "
            "`python -m pip install -r requirements.txt` and rerun."
        ) from exc
    return torch, nn, Normal


def _build_torch_modules(torch_module, nn_module, normal_dist):
    class GaussianActor(nn_module.Module):
        def __init__(
            self, obs_dim: int, action_dim: int, hidden_sizes: Iterable[int]
        ) -> None:
            super().__init__()
            layers: List[Any] = []
            previous = obs_dim
            for hidden in hidden_sizes:
                layers.append(nn_module.Linear(previous, hidden))
                layers.append(nn_module.Tanh())
                previous = hidden
            layers.append(nn_module.Linear(previous, action_dim))
            self.mean = nn_module.Sequential(*layers)
            self.log_std = nn_module.Parameter(torch_module.full((action_dim,), -0.5))

        def distribution(self, obs):
            mean = self.mean(obs)
            std = torch_module.exp(self.log_std).expand_as(mean)
            return normal_dist(mean, std)

        def forward(self, obs):
            return self.mean(obs)

    class CentralValue(nn_module.Module):
        def __init__(self, state_dim: int, hidden_sizes: Iterable[int]) -> None:
            super().__init__()
            layers: List[Any] = []
            previous = state_dim
            for hidden in hidden_sizes:
                layers.append(nn_module.Linear(previous, hidden))
                layers.append(nn_module.Tanh())
                previous = hidden
            layers.append(nn_module.Linear(previous, 1))
            self.value = nn_module.Sequential(*layers)

        def forward(self, state):
            return self.value(state).squeeze(-1)

    return GaussianActor, CentralValue


class HAPPOTrainer:
    """Sequential PPO actor updates for heterogeneous agents."""

    def __init__(
        self,
        env: HybridCsiDualUAVEnv,
        training_config: CHAPPOTrainingConfig | None = None,
    ) -> None:
        self.env = env
        self.config = training_config or CHAPPOTrainingConfig()
        self.torch, self.nn, self.Normal = _require_torch()
        self.GaussianActor, self.CentralValue = _build_torch_modules(
            self.torch, self.nn, self.Normal
        )

        self.device = self.config.device or (
            "cuda" if self.torch.cuda.is_available() else "cpu"
        )
        self.torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)

        obs_dims = env.observation_dims()
        action_dims = env.action_dims()
        self.actors: MutableMapping[str, Any] = {}
        self.actor_optimizers: MutableMapping[str, Any] = {}
        for name in env.agent_names:
            actor = self.GaussianActor(
                obs_dims[name], action_dims[name], self.config.hidden_sizes
            ).to(self.device)
            self.actors[name] = actor
            self.actor_optimizers[name] = self.torch.optim.Adam(
                actor.parameters(), lr=self.config.actor_lr
            )

        self.critic = self.CentralValue(
            env.config.global_state_dim, self.config.hidden_sizes
        ).to(self.device)
        self.critic_optimizer = self.torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

    def act(
        self, observations: Mapping[str, np.ndarray], deterministic: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        actions: Dict[str, np.ndarray] = {}
        log_probs: Dict[str, float] = {}
        for name, actor in self.actors.items():
            obs_tensor = self._tensor(observations[name]).unsqueeze(0)
            with self.torch.no_grad():
                dist = actor.distribution(obs_tensor)
                action_tensor = dist.mean if deterministic else dist.sample()
                log_prob = dist.log_prob(action_tensor).sum(dim=-1)
            actions[name] = action_tensor.squeeze(0).cpu().numpy()
            log_probs[name] = float(log_prob.item())
        return actions, log_probs

    def collect_episode(self, stage: CurriculumStage) -> Dict[str, Any]:
        obs, state = self.env.reset(stage)
        storage: Dict[str, Any] = {
            "states": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "infos": [],
            "obs": {name: [] for name in self.env.agent_names},
            "actions": {name: [] for name in self.env.agent_names},
            "log_probs": {name: [] for name in self.env.agent_names},
        }

        done = False
        while not done:
            actions, log_probs = self.act(obs, deterministic=False)
            with self.torch.no_grad():
                value = self.critic(self._tensor(state).unsqueeze(0)).item()

            for name in self.env.agent_names:
                storage["obs"][name].append(obs[name])
                storage["actions"][name].append(actions[name])
                storage["log_probs"][name].append(log_probs[name])
            storage["states"].append(state)
            storage["values"].append(value)

            obs, state, reward, done, info = self.env.step(actions)
            storage["rewards"].append(reward)
            storage["dones"].append(done)
            storage["infos"].append(info)

        storage["next_value"] = 0.0
        return storage

    def update(self, rollout: Mapping[str, Any]) -> Dict[str, float]:
        advantages, returns = self._compute_gae(rollout)
        advantages_t = self._tensor(advantages)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1.0e-8
        )
        returns_t = self._tensor(returns)
        states_t = self._tensor(np.asarray(rollout["states"], dtype=np.float32))

        modifier = advantages_t.clone()
        actor_losses: List[float] = []
        update_order = list(self.env.agent_names)
        self.rng.shuffle(update_order)

        for name in update_order:
            actor = self.actors[name]
            optimizer = self.actor_optimizers[name]
            obs_t = self._tensor(np.asarray(rollout["obs"][name], dtype=np.float32))
            actions_t = self._tensor(
                np.asarray(rollout["actions"][name], dtype=np.float32)
            )
            old_log_probs_t = self._tensor(
                np.asarray(rollout["log_probs"][name], dtype=np.float32)
            )

            latest_loss = 0.0
            for _ in range(self.config.update_epochs):
                dist = actor.distribution(obs_t)
                new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = self.torch.exp(new_log_probs - old_log_probs_t)
                unclipped = ratio * modifier.detach()
                clipped = self.torch.clamp(
                    ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
                ) * modifier.detach()
                actor_loss = -self.torch.min(unclipped, clipped).mean()
                actor_loss = actor_loss - self.config.entropy_coef * entropy

                optimizer.zero_grad()
                actor_loss.backward()
                self.nn.utils.clip_grad_norm_(
                    actor.parameters(), self.config.max_grad_norm
                )
                optimizer.step()
                latest_loss = float(actor_loss.detach().cpu().item())

            with self.torch.no_grad():
                dist_after = actor.distribution(obs_t)
                log_probs_after = dist_after.log_prob(actions_t).sum(dim=-1)
                update_ratio = self.torch.exp(log_probs_after - old_log_probs_t)
                modifier = modifier * update_ratio.detach()
            actor_losses.append(latest_loss)

        latest_critic_loss = 0.0
        for _ in range(self.config.update_epochs):
            values = self.critic(states_t)
            critic_loss = self.config.value_coef * self.torch.mean(
                (values - returns_t) ** 2
            )
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.max_grad_norm
            )
            self.critic_optimizer.step()
            latest_critic_loss = float(critic_loss.detach().cpu().item())

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": latest_critic_loss,
        }

    def _compute_gae(self, rollout: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(rollout["rewards"], dtype=np.float32)
        dones = np.asarray(rollout["dones"], dtype=np.float32)
        values = np.asarray(rollout["values"], dtype=np.float32)
        next_value = float(rollout.get("next_value", 0.0))
        values_next = np.concatenate([values[1:], np.asarray([next_value], dtype=np.float32)])

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * values_next[t] * nonterminal - values[t]
            last_advantage = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * nonterminal
                * last_advantage
            )
            advantages[t] = last_advantage
        returns = advantages + values
        return advantages, returns

    def _tensor(self, values: np.ndarray) -> Any:
        return self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device)


def train_c_happo(
    env: HybridCsiDualUAVEnv,
    training_config: CHAPPOTrainingConfig | None = None,
    verbose: bool = True,
) -> Tuple[HAPPOTrainer, CHAPPOHistory]:
    cfg = training_config or CHAPPOTrainingConfig()
    trainer = HAPPOTrainer(env, cfg)
    scheduler = CurriculumScheduler(env.config)
    history = CHAPPOHistory()
    report_every = max(1, cfg.episodes // 10)

    for episode in range(cfg.episodes):
        stage = scheduler.stage
        rollout = trainer.collect_episode(stage)
        losses = trainer.update(rollout)
        infos: List[HybridStepInfo] = rollout["infos"]
        final_info = infos[-1]

        episode_return = float(np.sum(rollout["rewards"]))
        comm_success_rate = float(
            np.mean([np.mean(info.communication_ok.astype(float)) for info in infos])
        )
        history.episode_returns.append(episode_return)
        history.final_rhos.append(final_info.rho)
        goal_success_rate = float(np.mean([float(info.goal_success) for info in infos]))
        history.final_distances.append(final_info.mean_target_distance)
        history.final_goal_distances.append(final_info.mean_goal_distance)
        history.communication_success_rates.append(comm_success_rate)
        history.goal_success_rates.append(goal_success_rate)
        history.curriculum_indices.append(stage.index)
        history.actor_losses.append(losses["actor_loss"])
        history.critic_losses.append(losses["critic_loss"])

        recent = float(np.mean(history.episode_returns[-report_every:]))
        advanced = scheduler.maybe_advance(episode + 1, recent)
        if verbose and ((episode + 1) % report_every == 0 or episode == 0 or advanced):
            print(
                f"episode={episode + 1:4d}/{cfg.episodes}, "
                f"stage={stage.index}, "
                f"return={episode_return:8.3f}, "
                f"rho={final_info.rho:.4e}, "
                f"target={final_info.mean_target_distance:6.2f} m, "
                f"goal={final_info.mean_goal_distance:6.2f} m, "
                f"comm={100.0 * comm_success_rate:5.1f}%"
            )

    return trainer, history


def evaluate_trainer(
    env: HybridCsiDualUAVEnv,
    trainer: HAPPOTrainer,
    episodes: int = 3,
    stage: CurriculumStage | None = None,
) -> Dict[str, float]:
    returns: List[float] = []
    final_rhos: List[float] = []
    final_distances: List[float] = []
    comm_rates: List[float] = []
    final_goal_distances: List[float] = []
    goal_rates: List[float] = []

    for _ in range(episodes):
        obs, _ = env.reset(stage or env.config.make_stage(env.config.curriculum_stages - 1))
        done = False
        episode_return = 0.0
        infos: List[HybridStepInfo] = []
        while not done:
            actions, _ = trainer.act(obs, deterministic=True)
            obs, _, reward, done, info = env.step(actions)
            episode_return += reward
            infos.append(info)
        returns.append(episode_return)
        final_rhos.append(infos[-1].rho)
        final_distances.append(infos[-1].mean_target_distance)
        final_goal_distances.append(infos[-1].mean_goal_distance)
        comm_rates.append(float(np.mean([np.mean(i.communication_ok) for i in infos])))
        goal_rates.append(float(np.mean([float(i.goal_success) for i in infos])))

    return {
        "mean_return": float(np.mean(returns)),
        "mean_final_rho": float(np.mean(final_rhos)),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_final_goal_distance": float(np.mean(final_goal_distances)),
        "communication_success_rate": float(np.mean(comm_rates)),
        "goal_success_rate": float(np.mean(goal_rates)),
    }


@dataclass
class CHAPPORolloutTrace:
    times: np.ndarray
    uav_positions: np.ndarray
    target_positions: np.ndarray
    rewards: np.ndarray
    rhos: np.ndarray
    mean_target_distances: np.ndarray
    goal_distances: np.ndarray
    mean_goal_distances: np.ndarray
    uav_distances: np.ndarray
    communication_sinrs: np.ndarray
    sensing_sinrs: np.ndarray
    uav_speeds: np.ndarray
    powers: np.ndarray
    reward_parts: Dict[str, np.ndarray]
    curriculum_index: int
    gamma_min: float
    csi_beta: float


def _moving_average(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    window = max(1, min(window, values.size))
    averaged = np.convolve(values, np.ones(window) / window, mode="valid")
    x_values = np.arange(window, values.size + 1)
    return x_values, averaged


def rollout_c_happo_policy(
    env: HybridCsiDualUAVEnv,
    trainer: HAPPOTrainer,
    stage: CurriculumStage | None = None,
) -> CHAPPORolloutTrace:
    stage = stage or env.config.make_stage(env.config.curriculum_stages - 1)
    observations, _ = env.reset(stage)
    positions = [env.uav_positions.copy()]
    targets = [env.target_state[:3].copy()]
    rewards: List[float] = []
    rhos: List[float] = []
    mean_distances: List[float] = []
    goal_distances: List[np.ndarray] = []
    mean_goal_distances: List[float] = []
    uav_distances: List[float] = []
    communication_sinrs: List[np.ndarray] = []
    sensing_sinrs: List[np.ndarray] = []
    speeds: List[np.ndarray] = []
    powers: List[float] = []
    reward_parts: Dict[str, List[float]] = {}

    done = False
    while not done:
        actions, _ = trainer.act(observations, deterministic=True)
        observations, _, reward, done, info = env.step(actions)
        positions.append(info.uav_positions.copy())
        targets.append(info.target_state[:3].copy())
        rewards.append(float(reward))
        rhos.append(info.rho)
        mean_distances.append(info.mean_target_distance)
        goal_distances.append(info.goal_distances.copy())
        mean_goal_distances.append(info.mean_goal_distance)
        uav_distances.append(info.uav_distance)
        communication_sinrs.append(info.communication_sinrs.copy())
        sensing_sinrs.append(info.sensing_sinrs.copy())
        speeds.append(np.linalg.norm(info.uav_velocities, axis=1))
        powers.append(info.power)
        for name, value in info.reward_parts.items():
            reward_parts.setdefault(name, []).append(float(value))

    times = np.arange(len(positions), dtype=float) * env.config.slot_duration
    return CHAPPORolloutTrace(
        times=times,
        uav_positions=np.stack(positions),
        target_positions=np.stack(targets),
        rewards=np.asarray(rewards, dtype=float),
        rhos=np.asarray(rhos, dtype=float),
        mean_target_distances=np.asarray(mean_distances, dtype=float),
        goal_distances=np.stack(goal_distances),
        mean_goal_distances=np.asarray(mean_goal_distances, dtype=float),
        uav_distances=np.asarray(uav_distances, dtype=float),
        communication_sinrs=np.stack(communication_sinrs),
        sensing_sinrs=np.stack(sensing_sinrs),
        uav_speeds=np.stack(speeds),
        powers=np.asarray(powers, dtype=float),
        reward_parts={
            name: np.asarray(values, dtype=float) for name, values in reward_parts.items()
        },
        curriculum_index=stage.index,
        gamma_min=stage.gamma_min,
        csi_beta=stage.csi_beta,
    )


def _style_plot_axis(ax: Any) -> None:
    ax.grid(True, color="#d1d5db", alpha=0.65, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_c_happo_training_curves(
    history: CHAPPOHistory,
    evaluation: Mapping[str, float],
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = np.asarray(history.episode_returns, dtype=float)
    rhos = np.asarray(history.final_rhos, dtype=float)
    distances = np.asarray(history.final_distances, dtype=float)
    goal_distances = np.asarray(history.final_goal_distances, dtype=float)
    comm_rates = np.asarray(history.communication_success_rates, dtype=float)
    goal_rates = np.asarray(history.goal_success_rates, dtype=float)
    stages = np.asarray(history.curriculum_indices, dtype=float)
    actor_losses = np.asarray(history.actor_losses, dtype=float)
    critic_losses = np.asarray(history.critic_losses, dtype=float)
    episodes = np.arange(1, len(returns) + 1)
    window = max(1, min(20, max(1, len(returns) // 5)))

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), constrained_layout=True)
    fig.suptitle("C-HAPPO Hybrid Beamforming Training", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(episodes, returns, color="#2563eb", alpha=0.18, linewidth=0.8)
    x_ma, y_ma = _moving_average(returns, window)
    ax.plot(x_ma, y_ma, color="#2563eb", linewidth=2.0, label=f"{window}-episode mean")
    ax.axhline(evaluation["mean_return"], color="#6b7280", linestyle="--", linewidth=1.1, label="eval mean")
    ax.set(title="Team return", xlabel="Episode", ylabel="Return")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[0, 1]
    ax.plot(episodes, rhos, color="#16a34a", alpha=0.18, linewidth=0.8)
    x_ma, y_ma = _moving_average(rhos, window)
    ax.plot(x_ma, y_ma, color="#16a34a", linewidth=2.0)
    ax.axhline(evaluation["mean_final_rho"], color="#6b7280", linestyle="--", linewidth=1.1, label="eval mean")
    if np.all(rhos > 0.0):
        ax.set_yscale("log")
    ax.set(title="Final PCRB trace", xlabel="Episode", ylabel="rho")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[1, 0]
    x_ma, y_ma = _moving_average(distances, window)
    ax.plot(x_ma, y_ma, color="#f97316", linewidth=2.0, label="target")
    if goal_distances.size:
        x_goal, y_goal = _moving_average(goal_distances, window)
        ax.plot(x_goal, y_goal, color="#7c3aed", linewidth=2.0, label="goal")
    ax.axhline(evaluation["mean_final_distance"], color="#f97316", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axhline(evaluation["mean_final_goal_distance"], color="#7c3aed", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(title="Final target and goal distance", xlabel="Episode", ylabel="Mean distance (m)")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[1, 1]
    x_ma, y_ma = _moving_average(100.0 * comm_rates, window)
    ax.plot(x_ma, y_ma, color="#0891b2", linewidth=2.0, label="communication")
    if goal_rates.size:
        x_goal, y_goal = _moving_average(100.0 * goal_rates, window)
        ax.plot(x_goal, y_goal, color="#7c3aed", linewidth=2.0, label="goal")
    ax.axhline(100.0 * evaluation["communication_success_rate"], color="#0891b2", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axhline(100.0 * evaluation["goal_success_rate"], color="#7c3aed", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(title="Constraint success", xlabel="Episode", ylabel="Success rate (%)")
    ax.set_ylim(-2.0, 102.0)
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[2, 0]
    ax.step(episodes, stages, where="post", color="#7c3aed", linewidth=2.0)
    ax.set(title="Curriculum stage", xlabel="Episode", ylabel="Stage index")
    _style_plot_axis(ax)

    ax = axes[2, 1]
    ax.plot(episodes, actor_losses, color="#2563eb", linewidth=1.3, label="actor")
    ax.plot(episodes, critic_losses, color="#ef4444", linewidth=1.3, label="critic")
    ax.set(title="Optimization losses", xlabel="Episode", ylabel="Loss")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_c_happo_trajectory_3d(
    trace: CHAPPORolloutTrace,
    config: HybridBeamformingConfig,
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2563eb", "#f97316")
    fig = plt.figure(figsize=(10.5, 8.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("C-HAPPO Dual-UAV Detection Trajectory", fontsize=15, fontweight="bold")

    for agent_id, color in enumerate(colors):
        path = trace.uav_positions[:, agent_id]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.2, label=f"UAV {agent_id + 1}")
        ax.scatter(*path[0], color=color, marker="o", s=48, edgecolor="white", linewidth=0.8)
        ax.scatter(*path[-1], color=color, marker="^", s=78, edgecolor="white", linewidth=0.8)

    target = trace.target_positions
    ax.plot(target[:, 0], target[:, 1], target[:, 2], color="#16a34a", linestyle="--", linewidth=2.0, label="Rogue UAV")
    ax.scatter(*target[0], color="#16a34a", marker="o", s=52)
    ax.scatter(*target[-1], color="#16a34a", marker="*", s=150)
    ax.scatter(*config.bs_position, color="#7c3aed", marker="P", s=120, label="Base station")
    for agent_id, color in enumerate(colors):
        goal = config.uav_goal_positions[agent_id]
        ax.scatter(*goal, color=color, marker="X", s=110, edgecolor="black", linewidth=0.7, label=f"UAV {agent_id + 1} goal")
        ax.plot(
            [trace.uav_positions[0, agent_id, 0], goal[0]],
            [trace.uav_positions[0, agent_id, 1], goal[1]],
            [trace.uav_positions[0, agent_id, 2], goal[2]],
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.35,
        )

    final_target = target[-1]
    ax.plot(
        [config.bs_position[0], final_target[0]],
        [config.bs_position[1], final_target[1]],
        [config.bs_position[2], final_target[2]],
        color="#7c3aed",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
    )
    for agent_id, color in enumerate(colors):
        final_uav = trace.uav_positions[-1, agent_id]
        ax.plot(
            [final_uav[0], final_target[0]],
            [final_uav[1], final_target[1]],
            [final_uav[2], final_target[2]],
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )

    all_points = np.vstack([trace.uav_positions.reshape(-1, 3), target, config.bs_position[None, :]])
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    spans = np.maximum(upper - lower, 10.0)
    centers = (lower + upper) / 2.0
    margins = 0.12 * spans
    ax.set_xlim(centers[0] - spans[0] / 2 - margins[0], centers[0] + spans[0] / 2 + margins[0])
    ax.set_ylim(centers[1] - spans[1] / 2 - margins[1], centers[1] + spans[1] / 2 + margins[1])
    ax.set_zlim(max(0.0, centers[2] - spans[2] / 2 - margins[2]), centers[2] + spans[2] / 2 + margins[2])
    ax.set_box_aspect(spans)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("Altitude z (m)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.4)
    ax.view_init(elev=25, azim=-58)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_c_happo_episode_metrics(
    trace: CHAPPORolloutTrace,
    config: HybridBeamformingConfig,
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = trace.times[1:]
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), constrained_layout=True)
    fig.suptitle("C-HAPPO Evaluation Episode Diagnostics", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for agent_id, color in enumerate(("#2563eb", "#f97316")):
        ax.plot(times, trace.uav_speeds[:, agent_id], color=color, linewidth=1.8, label=f"UAV {agent_id + 1}")
    ax.axhline(config.max_uav_speed, color="#dc2626", linestyle="--", linewidth=1.1, label="speed limit")
    ax.set(title="UAV speed", xlabel="Time (s)", ylabel="Speed (m/s)")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[0, 1]
    comm_db = 10.0 * np.log10(np.maximum(trace.communication_sinrs, np.finfo(float).tiny))
    for agent_id, color in enumerate(("#2563eb", "#f97316")):
        ax.plot(times, comm_db[:, agent_id], color=color, linewidth=1.8, label=f"UAV {agent_id + 1}")
    ax.axhline(10.0 * np.log10(trace.gamma_min), color="#dc2626", linestyle="--", linewidth=1.1, label="QoS threshold")
    ax.set(title="Communication SINR", xlabel="Time (s)", ylabel="SINR (dB)")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[1, 0]
    sensing_db = 10.0 * np.log10(np.maximum(trace.sensing_sinrs, np.finfo(float).tiny))
    for node_id, label in enumerate(("BS", "UAV 1", "UAV 2")):
        ax.plot(times, sensing_db[:, node_id], linewidth=1.7, label=label)
    ax.set(title="Sensing SINR", xlabel="Time (s)", ylabel="SINR (dB)")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[1, 1]
    ax.plot(times, trace.rhos, color="#16a34a", linewidth=2.0, label="rho")
    if np.all(trace.rhos > 0.0):
        ax.set_yscale("log")
    ax.set(title="Tracking uncertainty", xlabel="Time (s)", ylabel="rho = trace(PCRB)")
    _style_plot_axis(ax)
    distance_ax = ax.twinx()
    distance_ax.plot(times, trace.mean_target_distances, color="#f97316", linewidth=1.8, label="target distance")
    distance_ax.plot(times, trace.mean_goal_distances, color="#7c3aed", linewidth=1.5, linestyle="--", label="goal distance")
    distance_ax.set_ylabel("Mean distance (m)")
    lines = ax.get_lines() + distance_ax.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False)

    ax = axes[2, 0]
    ax.plot(times, trace.powers, color="#7c3aed", linewidth=2.0, label="BS transmit power")
    ax.axhline(config.total_power, color="#dc2626", linestyle="--", linewidth=1.1, label="power budget")
    ax.set(title="Hybrid beamforming power", xlabel="Time (s)", ylabel="Power (W)")
    ax.legend(frameon=False)
    _style_plot_axis(ax)

    ax = axes[2, 1]
    ax.plot(times, trace.rewards, color="#111827", linewidth=2.0, label="total")
    for name in ("goal_progress", "goal_distance", "tracking", "pcrb", "hbf_spectral_efficiency", "communication", "energy"):
        if name in trace.reward_parts:
            ax.plot(times, trace.reward_parts[name], linewidth=1.1, alpha=0.85, label=name)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set(title="Reward decomposition", xlabel="Time (s)", ylabel="Reward")
    ax.legend(frameon=False, ncol=2)
    _style_plot_axis(ax)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_c_happo_visualizations(
    history: CHAPPOHistory,
    trace: CHAPPORolloutTrace,
    config: HybridBeamformingConfig,
    evaluation: Mapping[str, float],
    output_dir: str,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    training_path = os.path.join(output_dir, "training_curves.png")
    trajectory_path = os.path.join(output_dir, "trajectory_3d.png")
    metrics_path = os.path.join(output_dir, "episode_metrics.png")
    data_path = os.path.join(output_dir, "run_data.npz")

    plot_c_happo_training_curves(history, evaluation, training_path)
    plot_c_happo_trajectory_3d(trace, config, trajectory_path)
    plot_c_happo_episode_metrics(trace, config, metrics_path)

    np.savez_compressed(
        data_path,
        episode_returns=np.asarray(history.episode_returns),
        final_rhos=np.asarray(history.final_rhos),
        final_distances=np.asarray(history.final_distances),
        final_goal_distances=np.asarray(history.final_goal_distances),
        communication_success_rates=np.asarray(history.communication_success_rates),
        goal_success_rates=np.asarray(history.goal_success_rates),
        curriculum_indices=np.asarray(history.curriculum_indices),
        actor_losses=np.asarray(history.actor_losses),
        critic_losses=np.asarray(history.critic_losses),
        times=trace.times,
        uav_positions=trace.uav_positions,
        target_positions=trace.target_positions,
        rewards=trace.rewards,
        rhos=trace.rhos,
        mean_target_distances=trace.mean_target_distances,
        goal_distances=trace.goal_distances,
        mean_goal_distances=trace.mean_goal_distances,
        uav_distances=trace.uav_distances,
        communication_sinrs=trace.communication_sinrs,
        sensing_sinrs=trace.sensing_sinrs,
        uav_speeds=trace.uav_speeds,
        powers=trace.powers,
        uav_goal_positions=config.uav_goal_positions,
        gamma_min=np.asarray([trace.gamma_min]),
        csi_beta=np.asarray([trace.csi_beta]),
    )
    return [training_path, trajectory_path, metrics_path, data_path]

def save_history(history: CHAPPOHistory, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "c_happo_history.npz")
    np.savez(
        path,
        episode_returns=np.asarray(history.episode_returns),
        final_rhos=np.asarray(history.final_rhos),
        final_distances=np.asarray(history.final_distances),
        final_goal_distances=np.asarray(history.final_goal_distances),
        communication_success_rates=np.asarray(history.communication_success_rates),
        goal_success_rates=np.asarray(history.goal_success_rates),
        curriculum_indices=np.asarray(history.curriculum_indices),
        actor_losses=np.asarray(history.actor_losses),
        critic_losses=np.asarray(history.critic_losses),
    )
    return path


def _parse_vec3_arg(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected comma-separated x,y,z")
    return np.asarray(parts, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C-HAPPO + hybrid beamforming + imperfect CSI for dual-UAV ISAC"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--csi-beta", type=float, default=0.10)
    parser.add_argument("--uav1-goal", type=_parse_vec3_arg, default=_parse_vec3_arg("90,20,70"))
    parser.add_argument("--uav2-goal", type=_parse_vec3_arg, default=_parse_vec3_arg("90,-20,70"))
    parser.add_argument("--max-acceleration", type=float, default=5.0)
    parser.add_argument("--rf-chains", type=int, default=3)
    parser.add_argument("--hbf-rank", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--curriculum-stages", type=int, default=2)
    parser.add_argument("--advance-every", type=int, default=25)
    parser.add_argument("--output-dir", default="output/c_happo")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG visualization output")
    args = parser.parse_args()

    env_config = HybridBeamformingConfig(
        max_steps=args.steps,
        max_uav_acceleration=args.max_acceleration,
        uav_goal_positions=np.stack([args.uav1_goal, args.uav2_goal]),
        rf_chains=args.rf_chains,
        bs_action_rank=args.hbf_rank,
        imperfect_csi_beta=args.csi_beta,
        curriculum_stages=args.curriculum_stages,
        curriculum_advance_every=args.advance_every,
    )
    training_config = CHAPPOTrainingConfig(
        episodes=args.episodes,
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        hidden_sizes=(args.hidden_size, args.hidden_size),
        seed=args.seed,
    )
    env = HybridCsiDualUAVEnv(env_config, seed=args.seed)

    try:
        trainer, history = train_c_happo(
            env, training_config=training_config, verbose=not args.quiet
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    result = evaluate_trainer(
        HybridCsiDualUAVEnv(env_config, seed=args.seed + 100),
        trainer,
        episodes=args.eval_episodes,
    )
    output_path = save_history(history, args.output_dir)
    visualization_paths: List[str] = []
    if not args.no_plots:
        trace = rollout_c_happo_policy(
            HybridCsiDualUAVEnv(env_config, seed=args.seed + 200),
            trainer,
            stage=env_config.make_stage(env_config.curriculum_stages - 1),
        )
        visualization_paths = save_c_happo_visualizations(
            history=history,
            trace=trace,
            config=env_config,
            evaluation=result,
            output_dir=args.output_dir,
        )

    print("\nC-HAPPO evaluation")
    print("------------------")
    print(f"mean return               : {result['mean_return']:.3f}")
    print(f"mean final rho            : {result['mean_final_rho']:.4e}")
    print(f"mean final target distance: {result['mean_final_distance']:.2f} m")
    print(f"mean final goal distance  : {result['mean_final_goal_distance']:.2f} m")
    print(f"communication success     : {100.0 * result['communication_success_rate']:.1f}%")
    print(f"goal success              : {100.0 * result['goal_success_rate']:.1f}%")
    print(f"saved history             : {output_path}")
    if visualization_paths:
        print("saved visualizations:")
        for path in visualization_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()




