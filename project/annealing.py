"""有限预算的在线模拟退火导航基线，不是完整 ISAC 或全局最优规划。"""

import numpy as np

from .baselines import observation_state
from .planning import physical_to_raw, rollout_cost


class SimulatedAnnealingController:
    """用 Metropolis 接受准则优化短时域联合加速度序列。"""

    DEFAULT_SETTINGS = {
        "horizon": 4, "iterations": 16, "initial_temperature": 1.0,
        "final_temperature": .08, "navigation_weight": 1.0,
        "effort_weight": .03, "clearance_weight": .12,
        "boundary_weight": .2, "seed": 0,
    }

    def __init__(self, config, horizon=4, iterations=16, initial_temperature=1.0,
                 final_temperature=.08, navigation_weight=1.0, effort_weight=.03,
                 clearance_weight=.12, boundary_weight=.2, seed=0):
        self.config = config
        self.horizon, self.iterations = int(horizon), int(iterations)
        if self.horizon < 1 or self.iterations < 1:
            raise ValueError("horizon and iterations must be positive")
        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self.weights = dict(navigation=float(navigation_weight), effort=float(effort_weight),
                            clearance=float(clearance_weight), boundary=float(boundary_weight),
                            collision=1.e4)
        self.seed = int(seed)
        self.settings = dict(
            horizon=self.horizon, iterations=self.iterations,
            initial_temperature=self.initial_temperature,
            final_temperature=self.final_temperature,
            navigation_weight=self.weights["navigation"], effort_weight=self.weights["effort"],
            clearance_weight=self.weights["clearance"], boundary_weight=self.weights["boundary"],
            seed=self.seed, model_evaluations_per_action=self.iterations + 5,
            objective="navigation_effort_clearance")
        self.reset(self.seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            self.settings["seed"] = self.seed
        self.rng = np.random.default_rng(self.seed)
        self._warm = None

    def _rollout_cost(self, sequences, position, velocity, goals, active):
        return rollout_cost(sequences, position, velocity, goals, active,
                            self.config, self.weights)

    def _references(self, goal, velocity, position, active):
        cfg = self.config
        distance = np.linalg.norm(goal, axis=1, keepdims=True)
        direction = goal / np.maximum(distance, 1e-12)
        desired_speed = np.minimum(cfg.max_uav_speed, distance / (self.horizon * cfg.slot_duration))
        track = (direction * desired_speed - velocity) / cfg.slot_duration
        track[~active] = 0.0
        zero = np.zeros((self.horizon, 2, 3))
        tracking = np.broadcast_to(track, zero.shape).copy()
        separation = position[0] - position[1]
        side = np.cross(separation, np.array([0., 0., 1.]))
        if np.linalg.norm(side) < 1e-9:
            side = np.array([0., 1., 0.])
        side /= np.linalg.norm(side)
        offset = np.vstack((side, -side)) * (.6 * cfg.max_uav_acceleration)
        left, right = tracking.copy(), tracking.copy()
        left[:2] += offset
        right[:2] -= offset
        warm = tracking if self._warm is None else self._warm.copy()
        return np.stack((zero, tracking, left, right, warm))

    def act(self, obs, active, deterministic=True):
        del deterministic  # reset(seed) 控制在线搜索的复现性。
        active = np.asarray(active, dtype=bool)
        position, velocity, goal_delta, teammate_position, teammate_velocity = observation_state(
            obs, self.config)
        position = np.vstack(((position[0] + teammate_position[1]) / 2,
                              (position[1] + teammate_position[0]) / 2))
        velocity = np.vstack(((velocity[0] + teammate_velocity[1]) / 2,
                              (velocity[1] + teammate_velocity[0]) / 2))
        goals = position + goal_delta
        references = self._references(goal_delta, velocity, position, active)
        costs = self._rollout_cost(references, position, velocity, goals, active)
        current = references[int(np.argmin(costs))].copy()
        current_cost = float(np.min(costs))
        best, best_cost = current.copy(), current_cost
        temperatures = np.geomspace(self.initial_temperature, self.final_temperature,
                                    self.iterations)
        for temperature in temperatures:
            proposal = current.copy()
            step = self.rng.integers(self.horizon)
            agent = self.rng.integers(2)
            proposal[step, agent] += self.rng.normal(
                0.0, temperature * self.config.max_uav_acceleration, 3)
            cost = float(self._rollout_cost(
                proposal, position, velocity, goals, active)[0])
            delta = cost - current_cost
            if delta <= 0 or self.rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                current, current_cost = proposal, cost
            if current_cost < best_cost:
                best, best_cost = current.copy(), current_cost
        self._warm = np.concatenate((best[1:], best[-1:]), axis=0)
        raw = physical_to_raw(best[0], active, self.config)
        return raw, np.zeros(2, dtype=np.float32)
