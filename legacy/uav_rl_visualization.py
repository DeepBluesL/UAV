"""双 UAV 强化学习结果可视化。

该模块只负责把训练历史和一个贪心回合画成 PNG；环境和算法仍在
``dual_uav_rl.py`` 中。默认使用无窗口 Agg 后端，因此在服务器或命令行也能运行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


UAV_COLORS = ("#2563eb", "#f97316")
TARGET_COLOR = "#16a34a"
BS_COLOR = "#7c3aed"
GRID_COLOR = "#d1d5db"


def _moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """返回移动平均的横坐标和数值，数据较短时自动缩小窗口。"""

    values = np.asarray(values, dtype=float)
    window = max(1, min(window, len(values)))
    averaged = np.convolve(values, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(values) + 1)
    return x, averaged


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, alpha=0.6, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_training_curves(
    history,
    random_result: Dict[str, float],
    learned_result: Dict[str, float],
    output_path: Path,
) -> None:
    """绘制训练回报、PCRB、跟踪距离与约束/能耗曲线。"""

    returns = np.asarray(history.episode_returns, dtype=float)
    rhos = np.asarray(history.final_rhos, dtype=float)
    distances = np.asarray(history.final_distances, dtype=float)
    comm_rates = np.asarray(history.communication_success_rates, dtype=float)
    energy = np.asarray(history.episode_energy_proxies, dtype=float)
    episodes = np.arange(1, len(returns) + 1)
    window = max(5, min(25, len(returns) // 10))

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    fig.suptitle("Dual-UAV IQL Training Curves", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(episodes, returns, color=UAV_COLORS[0], alpha=0.18, linewidth=0.8)
    x_ma, y_ma = _moving_average(returns, window)
    ax.plot(x_ma, y_ma, color=UAV_COLORS[0], linewidth=2.0, label=f"{window}-episode mean")
    ax.axhline(
        random_result["mean_return"], color="#6b7280", linestyle="--", linewidth=1.2,
        label="random baseline",
    )
    ax.set(title="Team return", xlabel="Episode", ylabel="Return")
    ax.legend(frameon=False)
    _style_axis(ax)

    ax = axes[0, 1]
    ax.plot(episodes, rhos, color=TARGET_COLOR, alpha=0.16, linewidth=0.8)
    x_ma, y_ma = _moving_average(rhos, window)
    ax.plot(x_ma, y_ma, color=TARGET_COLOR, linewidth=2.0)
    ax.axhline(
        learned_result["mean_final_rho"], color="#374151", linestyle=":", linewidth=1.2,
        label="greedy evaluation mean",
    )
    ax.set_yscale("log")
    ax.set(title="Final PCRB trace", xlabel="Episode", ylabel="rho = trace(PCRB)")
    ax.legend(frameon=False)
    _style_axis(ax)

    ax = axes[1, 0]
    ax.plot(episodes, distances, color=UAV_COLORS[1], alpha=0.16, linewidth=0.8)
    x_ma, y_ma = _moving_average(distances, window)
    ax.plot(x_ma, y_ma, color=UAV_COLORS[1], linewidth=2.0)
    ax.axhline(
        learned_result["mean_final_distance"], color="#374151", linestyle=":",
        linewidth=1.2, label="greedy evaluation mean",
    )
    ax.set(title="Final tracking distance", xlabel="Episode", ylabel="Mean distance (m)")
    ax.legend(frameon=False)
    _style_axis(ax)

    ax = axes[1, 1]
    x_ma, comm_ma = _moving_average(100.0 * comm_rates, window)
    ax.plot(x_ma, comm_ma, color=UAV_COLORS[0], linewidth=2.0, label="Comm. success")
    ax.set(title="Communication and motion cost", xlabel="Episode", ylabel="Success rate (%)")
    ax.set_ylim(-2.0, 102.0)
    _style_axis(ax)
    energy_ax = ax.twinx()
    x_ma, energy_ma = _moving_average(energy, window)
    energy_ax.plot(x_ma, energy_ma, color=UAV_COLORS[1], linewidth=1.8, label="Energy proxy")
    energy_ax.set_ylabel("Episode energy proxy")
    lines = ax.get_lines() + energy_ax.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="lower right")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_3d(trace, config, output_path: Path) -> None:
    """按论文概念图的角色关系绘制 BS、两架 UAV 与移动目标的三维轨迹。"""

    fig = plt.figure(figsize=(10.5, 8.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("Learned Dual-UAV ISAC Trajectory", fontsize=15, fontweight="bold")

    for agent_id, color in enumerate(UAV_COLORS):
        path = trace.uav_positions[:, agent_id]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.2, label=f"UAV {agent_id + 1}")
        ax.scatter(*path[0], color=color, marker="o", s=48, edgecolor="white", linewidth=0.8)
        ax.scatter(*path[-1], color=color, marker="^", s=78, edgecolor="white", linewidth=0.8)

    target = trace.target_positions
    ax.plot(
        target[:, 0], target[:, 1], target[:, 2], color=TARGET_COLOR,
        linestyle="--", linewidth=2.0, label="Moving target",
    )
    ax.scatter(*target[0], color=TARGET_COLOR, marker="o", s=52)
    ax.scatter(*target[-1], color=TARGET_COLOR, marker="*", s=150)
    ax.scatter(*config.bs_position, color=BS_COLOR, marker="P", s=110, label="Base station")

    # 最终时隙画出感知链路，让图与论文的 ISAC 场景关系一致。
    for agent_id, color in enumerate(UAV_COLORS):
        final_uav = trace.uav_positions[-1, agent_id]
        ax.plot(
            [final_uav[0], target[-1, 0]],
            [final_uav[1], target[-1, 1]],
            [final_uav[2], target[-1, 2]],
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )

    all_points = np.vstack(
        [trace.uav_positions.reshape(-1, 3), target, config.bs_position[None, :]]
    )
    spans = np.maximum(np.ptp(all_points, axis=0), 10.0)
    centers = np.mean([np.min(all_points, axis=0), np.max(all_points, axis=0)], axis=0)
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


def plot_episode_metrics(trace, config, output_path: Path) -> None:
    """绘制速度、通信 SINR、PCRB/距离和奖励分解。"""

    times = trace.times[1:]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    fig.suptitle("Greedy Policy - One Episode Diagnostics", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for agent_id, color in enumerate(UAV_COLORS):
        ax.step(times, trace.uav_speeds[:, agent_id], where="post", color=color, linewidth=1.8, label=f"UAV {agent_id + 1}")
    ax.axhline(config.max_uav_speed, color="#dc2626", linestyle="--", linewidth=1.2, label="speed limit")
    ax.set(title="UAV speed", xlabel="Time (s)", ylabel="Speed (m/s)")
    ax.legend(frameon=False)
    _style_axis(ax)

    ax = axes[0, 1]
    comm_db = 10.0 * np.log10(np.maximum(trace.communication_sinrs, np.finfo(float).tiny))
    for agent_id, color in enumerate(UAV_COLORS):
        ax.plot(times, comm_db[:, agent_id], color=color, linewidth=1.8, label=f"UAV {agent_id + 1}")
    gamma_db = 10.0 * np.log10(config.gamma_min)
    ax.axhline(gamma_db, color="#dc2626", linestyle="--", linewidth=1.2, label="QoS threshold")
    ax.set(title="Communication quality", xlabel="Time (s)", ylabel="Communication SINR (dB)")
    ax.legend(frameon=False)
    _style_axis(ax)

    ax = axes[1, 0]
    ax.plot(times, trace.rhos, color=TARGET_COLOR, linewidth=2.0, label="PCRB trace")
    ax.set_yscale("log")
    ax.set(title="Tracking uncertainty and geometry", xlabel="Time (s)", ylabel="rho = trace(PCRB)")
    _style_axis(ax)
    distance_ax = ax.twinx()
    distance_ax.plot(times, trace.mean_target_distances, color=UAV_COLORS[1], linewidth=1.8, label="Mean target distance")
    distance_ax.set_ylabel("Mean target distance (m)")
    lines = ax.get_lines() + distance_ax.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False)

    ax = axes[1, 1]
    ax.plot(times, trace.rewards, color="#111827", linewidth=2.0, label="Total reward")
    for name in ("tracking", "pcrb", "sensing", "communication", "energy"):
        if name in trace.reward_parts:
            ax.plot(times, trace.reward_parts[name], linewidth=1.1, alpha=0.8, label=name)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set(title="Reward decomposition", xlabel="Time (s)", ylabel="Reward")
    ax.legend(frameon=False, ncol=2)
    _style_axis(ax)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_run_artifacts(
    history,
    trace,
    config,
    random_result: Dict[str, float],
    learned_result: Dict[str, float],
    output_dir: str | Path,
) -> List[Path]:
    """保存三张 PNG 和可复用的 NumPy 数据文件。"""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training_curves.png"
    trajectory_path = output_dir / "trajectory_3d.png"
    metrics_path = output_dir / "episode_metrics.png"
    data_path = output_dir / "run_data.npz"

    plot_training_curves(history, random_result, learned_result, training_path)
    plot_trajectory_3d(trace, config, trajectory_path)
    plot_episode_metrics(trace, config, metrics_path)

    np.savez_compressed(
        data_path,
        episode_returns=np.asarray(history.episode_returns),
        final_rhos=np.asarray(history.final_rhos),
        final_distances=np.asarray(history.final_distances),
        communication_success_rates=np.asarray(history.communication_success_rates),
        episode_energy_proxies=np.asarray(history.episode_energy_proxies),
        times=trace.times,
        uav_positions=trace.uav_positions,
        target_positions=trace.target_positions,
        actions=trace.actions,
        rewards=trace.rewards,
        rhos=trace.rhos,
        communication_sinrs=trace.communication_sinrs,
        sensing_sinrs=trace.sensing_sinrs,
        uav_speeds=trace.uav_speeds,
    )
    return [training_path, trajectory_path, metrics_path, data_path]
