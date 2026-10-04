"""Sensing-aware receding-horizon baselines using only public belief state."""

import numpy as np

from .annealing import SimulatedAnnealingController
from .crb import crb_from_sensing_sinr
from .mpc import MPCController
from .nominal_links import nominal_link_metrics
from .pcrb import process_noise_covariance, state_transition_matrix
from .planning import rollout_cost
from .tracking import measurement_covariance, measurement_jacobian


class BeliefTrajectoryCost:
    """Average dimensionless belief/link cost over the configured planning horizon."""

    def __init__(self, config, estimate, covariance, sensing_weight, communication_weight,
                 uncertainty_ref=1.0):
        self.config = config
        self.estimate = np.asarray(estimate, dtype=float).copy()
        self.covariance = np.asarray(covariance, dtype=float).copy()
        self.sensing_weight = float(sensing_weight)
        self.communication_weight = float(communication_weight)
        self.uncertainty_ref = float(uncertainty_ref)
        if self.uncertainty_ref <= 0:
            raise ValueError("uncertainty_ref must be positive")

    def __call__(self, positions, velocities, active):
        batch, horizon = positions.shape[:2]
        result = np.zeros(batch)
        transition = state_transition_matrix(self.config.slot_duration)
        process = process_noise_covariance(
            self.config.slot_duration, self.config.process_noise_intensity)
        for candidate in range(batch):
            mean, covariance = self.estimate.copy(), self.covariance.copy()
            for step in range(horizon):
                live = active[candidate, step]
                if not np.any(live):
                    break
                mean = transition @ mean
                covariance = transition @ covariance @ transition.T + process
                comm, sensing = nominal_link_metrics(
                    positions[candidate, step], mean[:3], active[candidate, step], self.config)
                covariance = self._measurement_update(
                    mean, covariance, positions[candidate, step], velocities[candidate, step],
                    active[candidate, step], comm, sensing)
                uncertainty = np.log1p(
                    np.trace(covariance[:3, :3]) / self.uncertainty_ref)
                shortfall = np.logaddexp(
                    0.0, (self.config.gamma_min - comm[live]) / self.config.gamma_min)
                communication_cost = float(np.mean(shortfall))
                result[candidate] += (
                    self.sensing_weight * uncertainty
                    + self.communication_weight * communication_cost)
        return result / horizon

    def _measurement_update(self, mean, covariance, positions, velocities, active,
                            communication, sensing):
        cfg = self.config
        if not cfg.collect_measurements:
            return covariance
        upload = np.r_[True, active & (communication >= cfg.gamma_min)]
        if not cfg.tracking_use_uav_measurements:
            upload[1:] = False
        sensor_positions = np.vstack((cfg.bs_position, positions))
        sensor_velocities = np.vstack((np.zeros(3), velocities))
        identity = np.eye(6)
        for source in np.flatnonzero(upload):
            crb = crb_from_sensing_sinr(
                sensing[source], cfg.bandwidth, cfg.kappa_d, cfg.kappa_theta,
                cfg.kappa_phi, cfg.kappa_v)
            noise = measurement_covariance(crb, cfg)
            jacobian = measurement_jacobian(
                mean, cfg.bs_position, sensor_positions[source],
                sensor_velocities[source], source > 0)
            innovation = jacobian @ covariance @ jacobian.T + noise
            gain = np.linalg.solve(innovation, jacobian @ covariance).T
            correction = identity - gain @ jacobian
            covariance = correction @ covariance @ correction.T + gain @ noise @ gain.T
            covariance = (covariance + covariance.T) / 2.0
        return covariance


class _SensingAwareMixin:
    def _init_sensing(self, sensing_weight, communication_weight, uncertainty_ref):
        self.sensing_weight = float(sensing_weight)
        self.communication_weight = float(communication_weight)
        self.uncertainty_ref = float(uncertainty_ref)
        if self.sensing_weight < 0 or self.communication_weight < 0:
            raise ValueError("sensing weights must be non-negative")
        if self.uncertainty_ref <= 0:
            raise ValueError("uncertainty_ref must be positive")
        self._belief = None
        self.settings.update(
            sensing_weight=self.sensing_weight,
            communication_weight=self.communication_weight,
            uncertainty_ref=self.uncertainty_ref,
            trajectory_cost_units="horizon_mean_dimensionless_added_to_native_navigation_cost",
            link_forecast="ratio_of_expected_powers",
            target_forecast="constant_velocity")

    def set_belief(self, estimated_target_state, tracking_covariance):
        estimate = np.asarray(estimated_target_state, dtype=float)
        covariance = np.asarray(tracking_covariance, dtype=float)
        if estimate.shape != (6,) or covariance.shape != (6, 6):
            raise ValueError("belief must contain state (6,) and covariance (6,6)")
        self._belief = estimate.copy(), covariance.copy()

    def _belief_cost(self):
        if self._belief is None:
            raise RuntimeError("set_belief must be called before sensing-aware planning")
        return BeliefTrajectoryCost(
            self.config, *self._belief, self.sensing_weight, self.communication_weight,
            self.uncertainty_ref)


class SensingAwareSAController(_SensingAwareMixin, SimulatedAnnealingController):
    DEFAULT_SETTINGS = {
        **SimulatedAnnealingController.DEFAULT_SETTINGS,
        "sensing_weight": .2, "communication_weight": .5, "uncertainty_ref": 1.0,
    }

    def __init__(self, config, sensing_weight=.2, communication_weight=.5,
                 uncertainty_ref=1.0, **kwargs):
        super().__init__(config, **kwargs)
        self._init_sensing(sensing_weight, communication_weight, uncertainty_ref)
        self.settings.update(
            objective="navigation_effort_clearance_expected_belief",
            forecast_steps_per_action=(self.iterations + 5) * self.horizon)

    def _rollout_cost(self, sequences, position, velocity, goals, active):
        if self.sensing_weight == 0 and self.communication_weight == 0:
            return super()._rollout_cost(sequences, position, velocity, goals, active)
        return rollout_cost(sequences, position, velocity, goals, active,
                            self.config, self.weights, self._belief_cost())


class SensingAwareMPCController(_SensingAwareMixin, MPCController):
    DEFAULT_SETTINGS = {
        **MPCController.DEFAULT_SETTINGS,
        "sensing_weight": .2, "communication_weight": .5, "uncertainty_ref": 1.0,
    }

    def __init__(self, config, sensing_weight=.2, communication_weight=.5,
                 uncertainty_ref=1.0, **kwargs):
        super().__init__(config, **kwargs)
        self._init_sensing(sensing_weight, communication_weight, uncertainty_ref)
        self.settings.update(
            objective="navigation_effort_clearance_expected_belief",
            max_forecast_steps_per_action=49 * self.horizon)

    def _trajectory_cost_enabled(self):
        return self.sensing_weight != 0 or self.communication_weight != 0

    def _trajectory_cost(self, positions, velocities, active):
        return self._belief_cost()(positions, velocities, active)
