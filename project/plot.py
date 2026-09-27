"""训练日志与单回合轨迹图；使用非交互后端，命令行运行不会弹窗。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .plot_trajectory import plot_trajectory


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_results(output, epochs, episodes, trace):
    output = Path(output)
    colors = ("#2563eb", "#dc2626")
    if epochs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        steps = [row["environment_steps"] for row in epochs]
        axes[0, 0].plot(steps, [r["mean_step_reward"] for r in epochs])
        axes[0, 0].set(title="Mean reward per environment step", ylabel="Reward")
        axes[0, 1].plot(steps, [r["value_loss"] for r in epochs])
        axes[0, 1].set(title="Central critic loss", ylabel="MSE")
        for i, color in enumerate(colors):
            axes[1, 0].plot(steps, [r[f"kl_{i}"] for r in epochs], color=color, label=f"UAV {i + 1}")
            axes[1, 1].plot(steps, [r[f"entropy_{i}"] for r in epochs], color=color, label=f"UAV {i + 1}")
        axes[1, 0].set(title="Policy KL (active samples)", ylabel="KL")
        axes[1, 1].set(title="Joint Gaussian entropy (active samples)", ylabel="Entropy")
        for ax in axes.flat:
            ax.set_xlabel("Environment interactions")
            ax.grid(alpha=.25)
        axes[1, 0].legend()
        axes[1, 1].legend()
        _save(fig, output / "training.png")

    if episodes:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        numbers = np.arange(1, len(episodes) + 1)
        axes[0].plot(numbers, [r["episode_return"] for r in episodes])
        successes = np.array([r["team_success"] for r in episodes])
        axes[1].plot(numbers, np.cumsum(successes) / numbers)
        axes[0].set(title="Completed team episodes", xlabel="Episode", ylabel="Return")
        axes[1].set(title="Cumulative team success", xlabel="Episode", ylabel="Success rate", ylim=(-.05, 1.05))
        for ax in axes:
            ax.grid(alpha=.25)
        _save(fig, output / "episodes.png")

    positions, goals = trace["positions"], trace["goals"]
    plot_trajectory(output, trace)

    times = trace["times"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, color in enumerate(colors):
        distances = np.linalg.norm(positions[:, i] - goals[i], axis=1)
        axes[0, 0].plot(times, distances, color=color, label=f"UAV {i + 1}")
        comm = trace["communication_sinrs"][:, i].copy()
        comm[~trace["source_mask"][:, i + 1]] = np.nan
        axes[1, 0].plot(times[1:], comm, color=color, label=f"UAV {i + 1}")
    axes[0, 0].set(title="Distance to own goal", ylabel="m")
    axes[0, 1].semilogy(times, trace["rho_pos"], color="#7c3aed")
    kind = str(trace.get("uncertainty_kind", "proxy_pcrb"))
    title = "EKF position covariance" if kind == "ekf_covariance" else "Position PCRB proxy"
    axes[0, 1].set(title=title, ylabel="trace(C_pp), m²")
    axes[1, 0].set(title="Communication while active", ylabel="SINR (linear)")
    axes[1, 1].plot(times[1:], trace["rewards"], color="#111827", label="Total")
    for name in ("progress", "arrival", "completion", "sensing", "timeout"):
        key = f"reward_{name}"
        if key in trace:
            axes[1, 1].plot(times[1:], trace[key], linewidth=1, label=name)
    axes[1, 1].set(title="Reward components", ylabel="Reward")
    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    axes[1, 1].legend(fontsize=8)
    _save(fig, output / "evaluation_metrics.png")

    if "prior_target_positions" in trace and str(trace.get("uncertainty_kind")) == "ekf_covariance":
        from .plot_tracking import plot_tracking
        plot_tracking(trace, output)
