"""轨迹图：目标真值只用于离线诊断，不进入策略观测。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(output, trace):
    positions, goals, active = trace["positions"], trace["goals"], trace["active"]
    colors = ("#2563eb", "#dc2626")
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    # 从侧面观察黑飞航迹，避免沿默认视线投影成一个点。
    ax.view_init(elev=24, azim=-125)
    for i, color in enumerate(colors):
        ax.plot(*positions[:, i].T, color=color, linewidth=2, label=f"UAV{i + 1}")
        ax.scatter(*positions[0, i], color=color, marker="o", s=42)
        ax.scatter(*goals[i], color=color, marker="X", s=90)
        exits = np.flatnonzero(~active[:, i])
        if len(exits):
            ax.scatter(*positions[exits[0], i], color=color, marker="s", s=48)

    bs = np.asarray(trace["bs_position"])
    ax.scatter(*bs, color="#f59e0b", edgecolor="black", marker="^", s=120, label="BS")
    truth = trace["target_positions"]
    ax.plot(*truth.T, color="black", linewidth=2,
            label="Unauthorized UAV truth (diagnostic only)")
    ax.scatter(*truth[0], color="black", marker="o", s=42)
    ax.scatter(*truth[-1], color="black", marker="X", s=70)
    estimate = trace.get("estimated_target_positions")
    if estimate is not None and len(estimate):
        ax.plot(*estimate.T, color="#10b981", linestyle="--", linewidth=1.6,
                label="Unauthorized UAV estimate")
    ax.set(title="Two sensing UAVs, BS and unauthorized UAV", xlabel="x (m)",
           ylabel="y (m)", zlabel="z (m)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(Path(output) / "trajectory.png", dpi=160)
    plt.close(fig)
