from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


EPS = 1e-9


def _vec(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def safe_norm(values: np.ndarray) -> float:
    return max(float(np.linalg.norm(values)), EPS)


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class BasicISACConfig:
    """Small one-BS/one-UAV version of the paper's pre-MADRL model."""

    slot_duration: float = 1.0
    bs_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    uav_initial: np.ndarray = field(
        default_factory=lambda: np.array([30.0, 10.0, 70.0], dtype=float)
    )
    target_initial_state: np.ndarray = field(
        default_factory=lambda: np.array([60.0, 0.0, 60.0, 1.0, 0.2, 0.0], dtype=float)
    )

    max_uav_step: float = 8.0
    total_power: float = 1.0
    gamma_min: float = 1.0

    rician_k: float = 10.0
    beta_s0: float = 1.0e8
    beta_c0: float = 1.0e6
    alpha_target: float = 1.0
    noise_power: float = 1.0e-3
    residual_bs_noise: float = 1.0e-3
    residual_uav_noise: float = 1.0e-3

    bandwidth: float = 1.0
    kappa_d: float = 1.0
    kappa_theta: float = 1.0e-3
    kappa_phi: float = 1.0e-3
    kappa_v: float = 0.2
    fusion_weight_bs: float = 0.5
    fusion_weight_uav: float = 0.5

    process_noise_std: float = 0.05
    initial_covariance_scale: float = 25.0
    stochastic_channels: bool = False

    def __post_init__(self):
        self.bs_position = _vec(self.bs_position)
        self.uav_initial = _vec(self.uav_initial)
        self.target_initial_state = _vec(self.target_initial_state)
        if self.target_initial_state.shape != (6,):
            raise ValueError("target_initial_state must be [x, y, z, vx, vy, vz].")
        weight_sum = self.fusion_weight_bs + self.fusion_weight_uav
        self.fusion_weight_bs /= weight_sum
        self.fusion_weight_uav /= weight_sum


@dataclass(frozen=True)
class LinkMetrics:
    sinr_comm: float
    sinr_sensing_bs: float
    sinr_sensing_uav: float
    comm_power: float
    sensing_power: float


@dataclass(frozen=True)
class StepInfo:
    t: int
    uav_position: np.ndarray
    true_target_state: np.ndarray
    estimated_target_state: np.ndarray
    sinr_comm: float
    sinr_sensing_bs: float
    sinr_sensing_uav: float
    crb_bs: np.ndarray
    crb_uav: np.ndarray
    fused_crb: np.ndarray
    rho: float
    average_rho: float
    constraints: Dict[str, bool]


def state_transition_matrix(slot_duration: float) -> np.ndarray:
    ident = np.eye(3)
    return np.block(
        [
            [ident, slot_duration * ident],
            [np.zeros((3, 3)), ident],
        ]
    )


def process_noise_covariance(slot_duration: float, process_noise_std: float) -> np.ndarray:
    t = slot_duration
    base = np.array([[t**3 / 3.0, t**2 / 2.0], [t**2 / 2.0, t]])
    return np.kron(base, (process_noise_std**2) * np.eye(3))


def spherical_measurement(target_state: np.ndarray) -> np.ndarray:
    """h(zeta): [range, polar angle, azimuth angle, radial velocity]."""

    x, y, z, vx, vy, vz = _vec(target_state)
    distance = safe_norm(np.array([x, y, z]))
    theta = np.arccos(np.clip(z / distance, -1.0, 1.0))
    phi = np.arctan2(y, x)
    radial_velocity = (x * vx + y * vy + z * vz) / distance
    return np.array([distance, theta, phi, radial_velocity], dtype=float)


def numerical_jacobian(fn, x: np.ndarray, step: float = 1.0e-5) -> np.ndarray:
    x = _vec(x)
    y0 = fn(x)
    jac = np.zeros((y0.size, x.size), dtype=float)
    for col in range(x.size):
        dx = np.zeros_like(x)
        dx[col] = step
        jac[:, col] = (fn(x + dx) - fn(x - dx)) / (2.0 * step)
    return jac


class BasicUAVISACEnv:
    """Minimal environment for equations before Section IV.

    Simplifications:
    - one BS, one UAV, one sensed target;
    - single-antenna BS, so steering vectors collapse to scalar 1;
    - scalar beamformers are represented by a normalized power split;
    - no inter-UAV collision constraint because N = 1.
    """

    def __init__(self, config: BasicISACConfig | None = None, seed: int = 0):
        self.config = config or BasicISACConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> Dict[str, np.ndarray | float]:
        self.t = 0
        self.uav_position = self.config.uav_initial.copy()
        self.true_target_state = self.config.target_initial_state.copy()
        self.estimated_target_state = self.config.target_initial_state.copy()
        self.covariance = (
            self.config.initial_covariance_scale * np.eye(6, dtype=float)
        )
        self.rho_history = [float(np.trace(self.covariance))]
        links = self.compute_link_metrics(self.uav_position, self.true_target_state, 0.5)
        self.last_observation = self._observation(links.sinr_comm, self.rho_history[-1])
        return self.last_observation

    def normalized_beams(self, sensing_power_fraction: float) -> Tuple[complex, complex]:
        fraction = float(np.clip(sensing_power_fraction, 0.0, 1.0))
        sensing_power = self.config.total_power * fraction
        comm_power = self.config.total_power - sensing_power
        return complex(np.sqrt(sensing_power)), complex(np.sqrt(comm_power))

    def compute_link_metrics(
        self,
        uav_position: np.ndarray,
        target_state: np.ndarray,
        sensing_power_fraction: float,
    ) -> LinkMetrics:
        w_sensing, w_comm = self.normalized_beams(sensing_power_fraction)
        target_position = _vec(target_state[:3])

        beta_c = self.communication_channel(uav_position)
        beta_s_bs = self.sensing_channel_bs(target_position)
        beta_s_uav = self.sensing_channel_uav(uav_position, target_position)

        comm_signal = abs(beta_c * w_comm) ** 2
        target_echo_interference = (
            abs(self.config.alpha_target * beta_s_uav * w_sensing) ** 2
            + abs(self.config.alpha_target * beta_s_uav * w_comm) ** 2
        )
        sinr_comm = comm_signal / (target_echo_interference + self.config.noise_power)

        sensing_signal_bs = abs(self.config.alpha_target * beta_s_bs * w_sensing) ** 2
        sensing_interference_bs = abs(
            self.config.alpha_target * beta_s_bs * w_comm
        ) ** 2
        sinr_sensing_bs = sensing_signal_bs / (
            sensing_interference_bs
            + self.config.residual_bs_noise
            + self.config.noise_power
        )

        sensing_signal_uav = abs(self.config.alpha_target * beta_s_uav * w_sensing) ** 2
        sensing_interference_uav = abs(
            self.config.alpha_target * beta_s_uav * w_comm
        ) ** 2
        sinr_sensing_uav = sensing_signal_uav / (
            sensing_interference_uav
            + self.config.residual_uav_noise
            + self.config.noise_power
        )

        return LinkMetrics(
            sinr_comm=float(sinr_comm),
            sinr_sensing_bs=float(sinr_sensing_bs),
            sinr_sensing_uav=float(sinr_sensing_uav),
            comm_power=abs(w_comm) ** 2,
            sensing_power=abs(w_sensing) ** 2,
        )

    def communication_channel(self, uav_position: np.ndarray) -> complex:
        distance = safe_norm(_vec(uav_position) - self.config.bs_position)
        return self._rician_factor() * np.sqrt(self.config.beta_c0 / (distance**2))

    def sensing_channel_bs(self, target_position: np.ndarray) -> complex:
        distance = safe_norm(_vec(target_position) - self.config.bs_position)
        return self._rician_factor() * np.sqrt(self.config.beta_s0 / (distance**4))

    def sensing_channel_uav(
        self, uav_position: np.ndarray, target_position: np.ndarray
    ) -> complex:
        d_bs_target = safe_norm(target_position - self.config.bs_position)
        d_uav_target = safe_norm(target_position - _vec(uav_position))
        return self._rician_factor() * np.sqrt(
            self.config.beta_s0 / (d_bs_target**2 * d_uav_target**2)
        )

    def crb_from_sensing_sinr(self, sinr: float) -> np.ndarray:
        sinr_sq = max(float(sinr) ** 2, EPS)
        return np.array(
            [
                self.config.kappa_d / (sinr_sq * self.config.bandwidth),
                self.config.kappa_theta / sinr_sq,
                self.config.kappa_phi / sinr_sq,
                self.config.kappa_v / sinr_sq,
            ],
            dtype=float,
        )

    def fuse_crbs(self, crb_bs: np.ndarray, crb_uav: np.ndarray) -> np.ndarray:
        precision = (
            self.config.fusion_weight_bs / np.maximum(crb_bs, EPS)
            + self.config.fusion_weight_uav / np.maximum(crb_uav, EPS)
        )
        return 1.0 / np.maximum(precision, EPS)

    def step(
        self,
        uav_delta,
        sensing_power_fraction: float = 0.5,
        add_measurement_noise: bool = False,
    ) -> Tuple[Dict[str, np.ndarray | float], StepInfo]:
        requested_delta = _vec(uav_delta)
        clipped_delta = self._clip_delta(requested_delta)
        self.uav_position = self.uav_position + clipped_delta

        f = state_transition_matrix(self.config.slot_duration)
        q = process_noise_covariance(
            self.config.slot_duration, self.config.process_noise_std
        )
        self.true_target_state = f @ self.true_target_state

        links = self.compute_link_metrics(
            self.uav_position, self.true_target_state, sensing_power_fraction
        )
        crb_bs = self.crb_from_sensing_sinr(links.sinr_sensing_bs)
        crb_uav = self.crb_from_sensing_sinr(links.sinr_sensing_uav)
        fused_crb = self.fuse_crbs(crb_bs, crb_uav)

        predicted_state = f @ self.estimated_target_state
        predicted_cov = f @ self.covariance @ f.T + q
        h_pred = spherical_measurement(predicted_state)
        h_true = spherical_measurement(self.true_target_state)
        measurement_cov = np.diag(fused_crb)
        measurement = h_true.copy()
        if add_measurement_noise:
            measurement += self.rng.multivariate_normal(np.zeros(4), measurement_cov)

        h_jac = numerical_jacobian(spherical_measurement, predicted_state)
        innovation = measurement - h_pred
        innovation[1] = wrap_to_pi(innovation[1])
        innovation[2] = wrap_to_pi(innovation[2])

        innovation_cov = h_jac @ predicted_cov @ h_jac.T + measurement_cov
        kalman_gain = predicted_cov @ h_jac.T @ np.linalg.pinv(innovation_cov)
        self.estimated_target_state = predicted_state + kalman_gain @ innovation
        ident = np.eye(6)
        self.covariance = (ident - kalman_gain @ h_jac) @ predicted_cov
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        rho = float(np.trace(self.covariance))
        self.rho_history.append(rho)
        self.t += 1

        constraints = {
            "communication_sinr": links.sinr_comm >= self.config.gamma_min,
            "uav_speed": safe_norm(clipped_delta) <= self.config.max_uav_step + 1.0e-9,
            "bs_total_power": abs(
                links.comm_power + links.sensing_power - self.config.total_power
            )
            <= 1.0e-9,
        }

        info = StepInfo(
            t=self.t,
            uav_position=self.uav_position.copy(),
            true_target_state=self.true_target_state.copy(),
            estimated_target_state=self.estimated_target_state.copy(),
            sinr_comm=links.sinr_comm,
            sinr_sensing_bs=links.sinr_sensing_bs,
            sinr_sensing_uav=links.sinr_sensing_uav,
            crb_bs=crb_bs,
            crb_uav=crb_uav,
            fused_crb=fused_crb,
            rho=rho,
            average_rho=float(np.mean(self.rho_history[1:])),
            constraints=constraints,
        )
        self.last_observation = self._observation(links.sinr_comm, rho)
        return self.last_observation, info

    def full_state(self) -> Dict[str, np.ndarray | float]:
        return {
            "sinr_comm": self.last_observation["sinr_comm"],
            "uav_position": self.uav_position.copy(),
            "rho": self.rho_history[-1],
            "target_state": self.true_target_state.copy(),
        }

    def _observation(self, sinr_comm: float, rho: float) -> Dict[str, np.ndarray | float]:
        return {
            "sinr_comm": float(sinr_comm),
            "uav_position": self.uav_position.copy(),
            "rho": float(rho),
            "target_state_estimate": self.estimated_target_state.copy(),
        }

    def _clip_delta(self, delta: np.ndarray) -> np.ndarray:
        norm = safe_norm(delta)
        if norm <= self.config.max_uav_step:
            return delta
        return delta / norm * self.config.max_uav_step

    def _rician_factor(self) -> complex:
        k = self.config.rician_k
        los = np.sqrt(k / (k + 1.0))
        if not self.config.stochastic_channels:
            return complex(los)
        scatter = (self.rng.normal() + 1j * self.rng.normal()) / np.sqrt(2.0)
        return los + np.sqrt(1.0 / (k + 1.0)) * scatter
