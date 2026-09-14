"""按完整团队回合统计；未完成的采样片段不冒充一个失败回合。"""

import numpy as np


class EpisodeRecorder:
    def __init__(self, initial_info):
        self.last_info = initial_info
        self.total_reward = 0.
        self.steps = 0
        self.rho_sum = 0.
        self.comm_ok = np.zeros(2)
        self.comm_count = np.zeros(2)
        self.safety_interventions = 0
        self.collisions = 0
        self.boundary_requests = 0
        self.reward_parts = {}

    def add(self, reward, info):
        self.last_info = info
        self.steps += 1
        self.total_reward += reward
        self.rho_sum += info["rho_pos"]
        active = info["active_before"]
        self.comm_count += active
        self.comm_ok += active & info["communication_ok"]
        self.safety_interventions += int(info["safety_intervention"])
        self.collisions += int(info["collision"])
        self.boundary_requests += int(info["boundary_requests"].sum())
        for name, value in info["reward_parts"].items():
            self.reward_parts[name] = self.reward_parts.get(name, 0.) + value

    def summary(self):
        info = self.last_info
        row = dict(
            episode_return=self.total_reward, length=self.steps,
            team_success=int(info["team_success"]),
            termination_reason=info["termination_reason"],
            rho_pos_mean=self.rho_sum / self.steps if self.steps else info["rho_pos"],
            rho_pos_final=info["rho_pos"],
            safety_interventions=self.safety_interventions, collisions=self.collisions,
            boundary_requests=self.boundary_requests,
        )
        for i in range(2):
            row.update({
                f"arrived_{i}": int(info["arrived"][i]),
                f"arrival_time_{i}": float(info["arrival_times"][i]) if info["arrived"][i] else None,
                f"path_length_{i}": float(info["path_lengths"][i]),
                f"energy_proxy_{i}": float(info["energy_proxies"][i]),
                f"active_steps_{i}": int(info["active_steps"][i]),
                f"goal_distance_{i}": float(info["goal_distances"][i]),
                f"communication_rate_{i}": (
                    float(self.comm_ok[i] / self.comm_count[i]) if self.comm_count[i] else None),
            })
        row.update({f"reward_{key}": value for key, value in self.reward_parts.items()})
        return row