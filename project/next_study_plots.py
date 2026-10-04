"""Plots for the final multi-budget study; never selects checkpoints by test results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .next_study_io import load_manifest, value, variant_name, variant_specs


def _save(fig, path, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _primary_rows(rows, epoch):
    return [row for row in rows if row["kind"] == "rule" or row.get("epoch") == epoch]


def _heatmap(output, rows, manifest):
    rows = _primary_rows(rows, int(manifest.get("primary_epoch", 100)))
    methods = sorted({row["method"] for row in rows})
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    lookup = {(row["method"], row["scenario"]): row for row in rows}
    specs = (("success_rate", "Success", 0., 1., "YlGn"),
             ("restricted_team_time", "Restricted time (s)", None, None, "YlOrRd"),
             ("tracking_prefix_rmse", "Prefix RMSE (m)", None, None, "PuBu"),
             ("communication_prefix_rate", "Prefix communication rate", 0., 1., "YlGn"))
    fig, axes = plt.subplots(2, 2, figsize=(max(12, .8 * len(scenarios)),
                                          max(7, .45 * len(methods) + 4)))
    for ax, (metric, title, low, high, cmap) in zip(axes.ravel(), specs):
        data = np.full((len(methods), len(scenarios)), np.nan)
        for i, method in enumerate(methods):
            for j, scenario in enumerate(scenarios):
                row = lookup.get((method, scenario))
                if row is not None and value(row, metric) is not None:
                    data[i, j] = value(row, metric)
        image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=low, vmax=high)
        for i, j in np.ndindex(data.shape):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2g}", ha="center", va="center", fontsize=7)
        ax.set(xticks=range(len(scenarios)), xticklabels=scenarios,
               yticks=range(len(methods)), yticklabels=methods, title=title)
        ax.tick_params(axis="x", rotation=50, labelsize=7)
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle("Primary checkpoint; RL means across training seeds, rules across evaluation episodes")
    _save(fig, output / "primary_comparison_heatmap.png")


def _tradeoff(output, rows, manifest):
    primary = _primary_rows(rows, int(manifest.get("primary_epoch", 100)))
    scenarios = [scenario for scenario in ("nominal", "crossing")
                 if any(row["scenario"] == scenario for row in primary)]
    methods = sorted({row["method"] for row in primary})
    colors = {method: plt.get_cmap("tab20")(index % 20)
              for index, method in enumerate(methods)}
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5), squeeze=False)
    for ax, scenario in zip(axes.ravel(), scenarios):
        selected = [item for item in primary if item["scenario"] == scenario]
        for row in selected:
            x, y = value(row, "restricted_team_time"), value(row, "tracking_prefix_rmse")
            if x is None or y is None:
                continue
            success = value(row, "success_rate")
            collisions = value(row, "collisions") or 0.
            risky = success is None or success < 1. or collisions > 0.
            ax.scatter(x, y, marker="x" if risky else "o", s=65,
                       color=colors[row["method"]], linewidths=1.8)
        ax.set(title=scenario, xlabel="Restricted team time (s)",
               ylabel="Fixed-prefix tracking RMSE (m)")
        ax.grid(alpha=.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[method], marker="s", linestyle="None",
                      label=method) for method in methods]
    handles += [Line2D([], [], color="black", marker="o", linestyle="None", label="success, no collision"),
                Line2D([], [], color="black", marker="x", linestyle="None", label="failure or collision")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(.5, .01))
    fig.suptitle("Navigation–tracking tradeoff\nCrosses have failures or collisions; no Pareto claim",
                 fontsize=10, y=.995)
    _save(fig, output / "navigation_tracking_tradeoff.png", rect=(0, .22, 1, .96))


def _budgets(output, rows):
    learned = [row for row in rows if row["kind"] == "checkpoint"]
    scenarios = [name for name in ("nominal", "crossing")
                 if any(row["scenario"] == name for row in learned)]
    fig, axes = plt.subplots(2, max(1, len(scenarios)), figsize=(6 * max(1, len(scenarios)), 7),
                             squeeze=False)
    for column, scenario in enumerate(scenarios):
        for method in sorted({row["method"] for row in learned}):
            selected = sorted((row for row in learned
                               if row["method"] == method and row["scenario"] == scenario),
                              key=lambda row: row["epoch"])
            if not selected:
                continue
            epochs = [row["epoch"] for row in selected]
            axes[0, column].plot(epochs, [value(row, "success_rate") for row in selected],
                                 marker="o", label=method)
            axes[1, column].plot(epochs, [value(row, "tracking_prefix_rmse") for row in selected],
                                 marker="o", label=method)
        axes[0, column].set(title=f"{scenario}: success", ylabel="Rate", ylim=(-.05, 1.05))
        axes[1, column].set(title=f"{scenario}: prefix RMSE", ylabel="m", xlabel="Planned epoch budget")
        for ax in axes[:, column]: ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle("Budget checkpoints fixed in advance\nRaw rewards omitted across reward configurations",
                 fontsize=10, y=.995)
    _save(fig, output / "budget_learning_curves.png", rect=(0, 0, 1, .96))


def _load_trace(output, method, scenario, seed, epoch, is_rule):
    base = (output / "eval" / "rules" if is_rule else
            output / "eval" / f"{method}_seed{seed}_epoch{epoch}")
    policy = method if is_rule else "mappo"
    path = base / "trajectories" / scenario / policy / "trajectory.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _trajectories(output, rows, manifest):
    primary, seed = int(manifest.get("primary_epoch", 100)), int(manifest["training_seeds"][0])
    requested = manifest.get("trajectory_methods")
    if requested is None:
        variants = [variant_name(spec) for spec in variant_specs(manifest)]
        requested = ["goal", "sensing_sa", *variants]
    methods = list(dict.fromkeys(requested))
    scenarios = [name for name in ("nominal", "crossing")
                 if any(row["scenario"] == name for row in rows)]
    fig = plt.figure(figsize=(4 * len(methods), 4 * len(scenarios)))
    plotted = 0
    for row_index, scenario in enumerate(scenarios):
        for column, method in enumerate(methods):
            is_rule = any(row["kind"] == "rule" and row["method"] == method for row in rows)
            trace = _load_trace(output, method, scenario, seed, primary, is_rule)
            if trace is None:
                continue
            ax = fig.add_subplot(len(scenarios), len(methods),
                                 row_index * len(methods) + column + 1, projection="3d")
            for index, color in enumerate(("#2563eb", "#dc2626")):
                ax.plot(*trace["positions"][:, index].T, color=color, label=f"UAV {index + 1}")
                ax.scatter(*trace["goals"][index], color=color, marker="X", label=f"goal {index + 1}")
            ax.plot(*trace["target_positions"].T, color="black", label="target truth (offline)")
            ax.plot(*trace["estimated_target_positions"].T, color="#10b981", linestyle="--",
                    label="estimate")
            ax.scatter(*trace["bs_position"], color="#f59e0b", marker="^", label="BS")
            ax.set(title=f"{scenario} / {method}", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
            plotted += 1
    if plotted:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96),
                   ncol=len(labels), fontsize=7)
    fig.suptitle(f"Preselected train seed {seed}; stored first evaluation seed {manifest['evaluation_seeds'][0]}",
                 y=.995)
    _save(fig, output / "trajectory_comparison.png", rect=(0, 0, 1, .91))


def plot_next_study(output):
    output, manifest = Path(output), load_manifest(output)
    import json
    summary = json.loads((output / "study_summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    _heatmap(output, rows, manifest); _tradeoff(output, rows, manifest)
    _budgets(output, rows); _trajectories(output, rows, manifest)
    return [output / name for name in ("primary_comparison_heatmap.png",
            "navigation_tracking_tradeoff.png", "budget_learning_curves.png",
            "trajectory_comparison.png")]
