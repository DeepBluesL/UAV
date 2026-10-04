"""按传感器几何生成量测并以CV-EKF递推目标状态。"""

import numpy as np

from .pcrb import process_noise_covariance, state_transition_matrix


ANGLE_INDICES = (1, 2)


def wrap_angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def geometric_measurement(state, bs_position, sensor_position, sensor_velocity, bistatic):
    """量测为传播路径长、方位、俯仰和路径长变化率。

    BS单基地使用2*BS-target往返路径，双基地使用BS-target-UAV路径；既有
    SINR→CRB的range项仅作该路径长度的未校准噪声代理。
    """
    state = np.asarray(state, dtype=float)
    position, velocity = state[:3], state[3:]
    bs = np.asarray(bs_position, dtype=float)
    sensor = np.asarray(sensor_position, dtype=float)
    sensor_velocity = np.asarray(sensor_velocity, dtype=float)
    from_sensor = position - sensor
    sensor_distance = max(float(np.linalg.norm(from_sensor)), 1e-12)
    sensor_los = from_sensor / sensor_distance
    azimuth = np.arctan2(from_sensor[1], from_sensor[0])
    elevation = np.arctan2(from_sensor[2], np.hypot(from_sensor[0], from_sensor[1]))
    if bistatic:
        from_bs = position - bs
        bs_distance = max(float(np.linalg.norm(from_bs)), 1e-12)
        distance = bs_distance + sensor_distance
        rate = velocity @ (from_bs / bs_distance) + (velocity - sensor_velocity) @ sensor_los
    else:
        distance = 2.0 * sensor_distance
        rate = 2.0 * (velocity - sensor_velocity) @ sensor_los
    return np.array([distance, azimuth, elevation, rate], dtype=float)


def measurement_jacobian(state, bs_position, sensor_position, sensor_velocity, bistatic,
                         eps=1e-5):
    """单/双基地量测解析Jacobian；eps仅保留接口兼容。"""
    del eps
    state = np.asarray(state, dtype=float)
    position, velocity = state[:3], state[3:]
    bs = np.asarray(bs_position, dtype=float)
    sensor = np.asarray(sensor_position, dtype=float)
    sensor_velocity = np.asarray(sensor_velocity, dtype=float)
    delta = position - sensor
    distance = max(float(np.linalg.norm(delta)), 1e-12)
    los = delta / distance
    relative_velocity = velocity - sensor_velocity
    jacobian = np.zeros((4, 6))
    horizontal2 = max(float(delta[0] ** 2 + delta[1] ** 2), 1e-12)
    horizontal = np.sqrt(horizontal2)
    jacobian[1, :3] = [-delta[1] / horizontal2, delta[0] / horizontal2, 0.0]
    jacobian[2, :3] = [-delta[0] * delta[2] / (distance ** 2 * horizontal),
                       -delta[1] * delta[2] / (distance ** 2 * horizontal),
                       horizontal / distance ** 2]
    rate_position = (relative_velocity - los * (relative_velocity @ los)) / distance
    if bistatic:
        bs_delta = position - bs
        bs_distance = max(float(np.linalg.norm(bs_delta)), 1e-12)
        bs_los = bs_delta / bs_distance
        jacobian[0, :3] = bs_los + los
        jacobian[3, :3] = ((velocity - bs_los * (velocity @ bs_los)) / bs_distance
                           + rate_position)
        jacobian[3, 3:] = bs_los + los
    else:
        jacobian[0, :3] = 2.0 * los
        jacobian[3, :3] = 2.0 * rate_position
        jacobian[3, 3:] = 2.0 * los
    return jacobian


def measurement_covariance(crb, config):
    """CRB仅作方差代理；原顺序[d,极角,方位,率]需换成[d,方位,俯仰,率]。"""
    floors = np.array([
        config.measurement_range_std_floor,
        config.measurement_angle_std_floor,
        config.measurement_angle_std_floor,
        config.measurement_range_rate_std_floor,
    ]) ** 2
    reordered = np.asarray(crb, dtype=float)[[0, 2, 1, 3]]
    return np.diag(np.maximum(reordered, floors))


class TrackingEKF:
    """六维CV-EKF；过程协方差与既有PCRB模型使用同一口径。"""

    def __init__(self, config, rng):
        self.config, self.rng = config, rng
        self.state = None
        self.covariance = None
        self.prior_state = None
        self.prior_covariance = None
        self.measurement_counts = np.zeros(3, dtype=int)
        # Offline diagnostics only; these copies never enter the filter recursion.
        self.last_residuals = np.full((3, 4), np.nan)
        self.last_innovations = np.full((3, 4, 4), np.nan)

    def initialize(self, truth):
        std = np.array([self.config.tracking_initial_position_std] * 3
                       + [self.config.tracking_initial_velocity_std] * 3)
        self.state = np.asarray(truth, dtype=float) + self.rng.normal(size=6) * std
        self.covariance = np.diag(std ** 2)
        self.prior_state = self.state.copy()
        self.prior_covariance = self.covariance.copy()
        self.measurement_counts[:] = 0
        self.last_residuals[:] = np.nan
        self.last_innovations[:] = np.nan
        return self.state.copy(), self.covariance.copy()

    def predict(self):
        cfg = self.config
        transition = state_transition_matrix(cfg.slot_duration)
        process = process_noise_covariance(cfg.slot_duration, cfg.process_noise_intensity)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.prior_state = self.state.copy()
        self.prior_covariance = self.covariance.copy()
        return self.prior_state.copy(), self.prior_covariance.copy()

    def update(self, truth, positions, velocities, source_crbs, source_mask, standard_noise):
        """顺序融合有效上传；量测仅在此处由真值生成。"""
        source_mask = np.asarray(source_mask, dtype=bool)
        bs = self.config.bs_position
        sensor_positions = np.vstack((bs, positions))
        sensor_velocities = np.vstack((np.zeros(3), velocities))
        identity = np.eye(6)
        self.last_residuals[:] = np.nan
        self.last_innovations[:] = np.nan
        for source in np.flatnonzero(source_mask):
            bistatic = source > 0
            sensor_position = sensor_positions[source]
            sensor_velocity = sensor_velocities[source]
            covariance = measurement_covariance(source_crbs[source], self.config)
            truth_measurement = geometric_measurement(
                truth, bs, sensor_position, sensor_velocity, bistatic)
            measured = truth_measurement + np.sqrt(np.diag(covariance)) * standard_noise[source]
            measured[1] = wrap_angle(measured[1])
            measured[2] = wrap_angle(measured[2])
            predicted = geometric_measurement(
                self.state, bs, sensor_position, sensor_velocity, bistatic)
            jacobian = measurement_jacobian(
                self.state, bs, sensor_position, sensor_velocity, bistatic)
            residual = measured - predicted
            residual[1] = wrap_angle(residual[1])
            residual[2] = wrap_angle(residual[2])
            innovation = jacobian @ self.covariance @ jacobian.T + covariance
            self.last_residuals[source] = residual
            self.last_innovations[source] = innovation
            cross_covariance = self.covariance @ jacobian.T
            gain = np.linalg.solve(innovation, cross_covariance.T).T
            self.state = self.state + gain @ residual
            correction = identity - gain @ jacobian
            self.covariance = (correction @ self.covariance @ correction.T
                               + gain @ covariance @ gain.T)
            self.covariance = (self.covariance + self.covariance.T) / 2.0
            self.measurement_counts[source] += 1
        return self.state.copy(), self.covariance.copy()
