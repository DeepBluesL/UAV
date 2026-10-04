"""Independent-RNG scenario randomization used only during training."""

from dataclasses import replace

import numpy as np

from .domain_randomization import domain_randomize
from .physics import ISACPhysics


class TrainingScenarioSampler:
    """Sample continuous tasks without consulting benchmark scenarios or seeds."""

    def __init__(self, base, distribution="fixed", seed=7):
        allowed = {"fixed", "randomized", "domain_randomized", "curriculum"}
        if distribution not in allowed:
            raise ValueError(f"training_distribution must be one of {sorted(allowed)}")
        self.base = base
        self.distribution = distribution
        # Separate stream: scenario draws never consume sensor/channel randomness.
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53434E]))
        self.current_kind = "fixed"
        self.interactions = 0

    @property
    def phase(self):
        if self.distribution != "curriculum":
            return self.distribution
        if self.interactions < 50_000:
            return "easy"
        if self.interactions < 150_000:
            return "geometry"
        return "domain"

    def advance(self, steps):
        """Add environment or behavior-cloning interactions to curriculum progress."""
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        self.interactions += int(steps)

    def sample(self):
        if self.distribution == "fixed":
            self.current_kind = "fixed"
            return replace(self.base)
        if self.phase == "easy":
            self.current_kind = "curriculum_easy"
            return replace(self.base)
        sampled = self._sample_randomized()
        if self.distribution == "randomized" or self.phase == "geometry":
            if self.distribution == "curriculum":
                self.current_kind = "curriculum_geometry_" + self.current_kind
            return sampled
        geometry_kind = self.current_kind
        sampled = domain_randomize(sampled, self.rng)
        self.current_kind = (("curriculum_domain_" if self.distribution == "curriculum"
                              else "domain_") + geometry_kind)
        return sampled

    def _sample_randomized(self):
        rng, base = self.rng, self.base
        draw = rng.random()
        if draw < .25:
            self.current_kind = "nominal"
            return replace(base)
        if draw < .50:
            self.current_kind = "crossing"
            x0, x1 = rng.uniform(20, 40), rng.uniform(105, 140)
            half = rng.uniform(18, 35)
            z = rng.uniform(55, 85)
            starts = np.array([[x0, -half, z], [x0, half, z]])
            goals = np.array([[x1, half, z], [x1, -half, z]])
        else:
            self.current_kind = "continuous"
            starts = base.uav_initial + rng.uniform(
                [-10, -12, -10], [10, 12, 10], size=(2, 3))
            # Sample across the flight volume, including goals behind the starts.
            reverse_first = bool(rng.integers(0, 2))
            goals = np.vstack([
                self._distant_goal(start, reverse=(index == 0) == reverse_first)
                for index, start in enumerate(starts)
            ])
            required = base.min_uav_distance + 2 * base.goal_tolerance
            while np.linalg.norm(goals[0] - goals[1]) < required:
                goals[1] = self._distant_goal(starts[1], reverse=reverse_first)
        margin = 1.e-6
        starts = np.clip(starts, base.world_low + margin, base.world_high - margin)
        goals = np.clip(goals, base.world_low + margin, base.world_high - margin)
        if np.linalg.norm(starts[0] - starts[1]) < base.min_uav_distance:
            direction = 1 if starts[0, 1] <= 0 else -1
            starts[1, 1] = np.clip(
                starts[0, 1] + direction * (base.min_uav_distance + 1),
                base.world_low[1] + margin, base.world_high[1] - margin)
        distances = np.linalg.norm(goals - starts, axis=1)
        arrival_lower = int(np.max(np.ceil(
            np.maximum(0., distances - base.goal_tolerance)
            / (base.max_uav_speed * base.slot_duration))))
        # Geometric slack only; collision avoidance can still make a task harder.
        deadline_low = min(110, max(35, arrival_lower + 10))
        return replace(
            base, uav_initial=starts, uav_initial_velocities=np.zeros((2, 3)),
            uav_goal_positions=goals, max_steps=int(rng.integers(deadline_low, 111)),
            imperfect_csi_beta=float(rng.uniform(.05, .31)),
        )

    def _distant_goal(self, start, reverse):
        for _ in range(100):
            goal = self.rng.uniform(self.base.world_low, self.base.world_high)
            low, high = self.base.world_low[0], self.base.world_high[0]
            if reverse:
                goal[0] = self.rng.uniform(low, max(low + 1.e-6, start[0] - 10))
            else:
                goal[0] = self.rng.uniform(min(high - 1.e-6, start[0] + 10), high)
            if np.linalg.norm(goal - start) >= 25:
                return goal
        raise RuntimeError("Could not sample a nontrivial training goal")

    def reset_env(self, env):
        """Install a new task while preserving the environment's RNG stream."""
        env.config = self.sample()
        env.physics = ISACPhysics(env.config, env.rng)
        return env.reset()
