"""冻结策略的回合评估，以及可复现的无学习导航对照。"""

import numpy as np

from .control import goal_raw_action
from .env import DualUAVEnv
from .metrics import EpisodeRecorder


class GoalController:
    """仅供环境验收的加减速导航规则；不是训练后的 PPO 策略。"""

    def __init__(self, config):
        self.config = config

    def act(self, obs, active, deterministic=True):
        return goal_raw_action(obs, active, self.config), np.zeros(2, dtype=np.float32)


def rollout_policy(policy, env, seed):
    obs, _, info = env.reset(seed=seed)
    recorder = EpisodeRecorder(info)
    positions = [info["positions"]]
    # 真值仅保存到诊断轨迹；策略 act() 始终只接收 obs 与活动标记。
    targets = [info["target_state"][:3]]
    estimates = [info["estimated_target_state"][:3]]
    rhos = [info["rho_pos"]]
    covariances = [info["tracking_covariance"]]
    active = [info["active"]]
    priors = [info.get("prior_estimated_target_state", info["estimated_target_state"])[:3]]
    measurements = [info.get("measurement_source_mask", np.zeros(3, dtype=bool))]
    rewards, source_masks, communication, reward_parts = [], [], [], {}
    while not env.terminated:
        if hasattr(policy, "set_belief"):
            policy.set_belief(
                info["estimated_target_state"], info["tracking_covariance"])
        # 执行时不调用集中 Critic，只让活动 Actor 前向计算。
        actions, _ = policy.act(obs, env.active, deterministic=True)
        obs, _, reward, terminated, truncated, info = env.step(actions)
        recorder.add(reward, info)
        positions.append(info["positions"])
        targets.append(info["target_state"][:3])
        estimates.append(info["estimated_target_state"][:3])
        rhos.append(info["rho_pos"])
        covariances.append(info["tracking_covariance"])
        active.append(info["active"])
        priors.append(info.get("prior_estimated_target_state", info["estimated_target_state"])[:3])
        measurements.append(info.get("measurement_source_mask", np.zeros(3, dtype=bool)))
        rewards.append(reward)
        source_masks.append(info["sensing_source_mask"])
        communication.append(info["communication_sinrs"])
        for key, value in info["reward_parts"].items():
            reward_parts.setdefault(key, []).append(value)
        if terminated or truncated:
            break
    trace = {
        "positions": np.asarray(positions), "target_positions": np.asarray(targets),
        "estimated_target_positions": np.asarray(estimates),
        "prior_target_positions": np.asarray(priors),
        "measurement_source_mask": np.asarray(measurements, dtype=bool),
        "uncertainty_kind": np.asarray(info.get("uncertainty_kind", "proxy_pcrb")),
        "bs_position": env.config.bs_position.copy(),
        "rho_pos": np.asarray(rhos), "active": np.asarray(active),
        "tracking_covariances": np.asarray(covariances),
        "rewards": np.asarray(rewards),
        "times": np.arange(len(positions)) * env.config.slot_duration,
        "source_mask": np.asarray(source_masks, dtype=bool).reshape(-1, 3),
        "communication_sinrs": np.asarray(communication).reshape(-1, 2),
        "goals": env.config.uav_goal_positions.copy(),
        **{f"reward_{key}": np.asarray(value) for key, value in reward_parts.items()},
    }
    return recorder.summary(), trace


def evaluate(policy, environment, reward, seeds):
    """同一场景使用明确的种子列表；多场景可逐个传入对应 EnvConfig。"""
    if hasattr(policy, "eval"):
        policy.eval()
    rows, first_trace = [], None
    env = DualUAVEnv(environment, reward)
    for seed in seeds:
        row, trace = rollout_policy(policy, env, int(seed))
        row["seed"] = int(seed)
        rows.append(row)
        if first_trace is None:
            first_trace = trace
    if not rows:
        raise ValueError("Evaluation requires at least one seed")

    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "episodes": len(rows), "team_success_rate": mean("team_success"),
        "mean_return": mean("episode_return"), "mean_rho_pos": mean("rho_pos_mean"),
        "mean_safety_interventions": mean("safety_interventions"),
        "mean_collisions": mean("collisions"), "mean_boundary_requests": mean("boundary_requests"),
    }
    for i in range(2):
        summary.update({
            f"uav_success_rate_{i}": mean(f"arrived_{i}"),
            f"mean_arrival_time_{i}": mean(f"arrival_time_{i}"),
            f"mean_path_length_{i}": mean(f"path_length_{i}"),
            f"mean_energy_proxy_{i}": mean(f"energy_proxy_{i}"),
            f"mean_communication_rate_{i}": mean(f"communication_rate_{i}"),
        })
    return summary, rows, first_trace
