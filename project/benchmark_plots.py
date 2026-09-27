"""Compact comparison plots for single-checkpoint benchmark evaluations."""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PANELS = (
    ("success_rate", "success_ci_low", "success_ci_high", "Team success rate", "Rate"),
    ("restricted_team_time_mean", "restricted_team_time_ci_low",
     "restricted_team_time_ci_high", "Restricted team time", "Time (s)"),
    ("episode_return_mean", "episode_return_ci_low", "episode_return_ci_high",
     "Episode return", "Return"),
    ("rho_prefix_mean_mean", "rho_prefix_mean_ci_low", "rho_prefix_mean_ci_high",
     "Prefix position uncertainty", "Mean covariance trace (m²)"),
    ("boundary_requests_mean", "boundary_requests_ci_low", "boundary_requests_ci_high",
     "Boundary requests", "Count / episode"),
    ("tracking_prefix_rmse_mean", "tracking_prefix_rmse_ci_low", "tracking_prefix_rmse_ci_high",
     "Prefix tracking RMSE", "3D position RMSE (m)"),
    ("total_energy_proxy_mean", "total_energy_proxy_ci_low", "total_energy_proxy_ci_high",
     "Energy proxy", "Dimensionless"),
    ("policy_ms_per_step_mean", "policy_ms_per_step_ci_low", "policy_ms_per_step_ci_high",
     "Decision time", "Milliseconds / step"),
    ("safety_interventions_mean", "safety_interventions_ci_low",
     "safety_interventions_ci_high", "Safety interventions", "Count / episode"),
)


def _safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return name or "scenario"


def plot_benchmark(output, summaries):
    """Write one 3x3 mean/95%-CI comparison figure per scenario."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({row["scenario"] for row in summaries})
    paths = []
    for scenario in scenarios:
        rows = sorted((row for row in summaries if row["scenario"] == scenario),
                      key=lambda row: row["method"])
        methods = [row["method"] for row in rows]
        fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
        for ax, (mean_key, low_key, high_key, title, ylabel) in zip(axes.flat, PANELS):
            plotted = []
            for index, row in enumerate(rows):
                mean, low, high = row.get(mean_key), row.get(low_key), row.get(high_key)
                if mean is None:
                    continue
                lower = 0 if low is None else max(0., float(mean) - float(low))
                upper = 0 if high is None else max(0., float(high) - float(mean))
                ax.errorbar(index, mean, yerr=np.array([[lower], [upper]]), fmt="o",
                            capsize=4, color="#2563eb")
                plotted.append(index)
            ax.set(title=title, ylabel=ylabel, xticks=range(len(methods)), xticklabels=methods)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.25)
            if mean_key == "success_rate":
                ax.set_ylim(0, 1.05)
            if not plotted:
                ax.text(.5, .5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(
            f"Benchmark comparison: {scenario}\n"
            "95% evaluation uncertainty for fixed policies; not training-seed uncertainty",
            fontsize=12,
        )
        path = output / f"comparison_{_safe_name(scenario)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
