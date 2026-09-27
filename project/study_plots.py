"""study 的策略热图、跨训练种子学习曲线和诊断轨迹比较。"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = ("nominal", "shifted_goals", "crossing", "reverse_goals", "wide_bounds",
             "fast_speed", "wide_fast", "noisy_csi", "tight_deadline",
             "prediction_only", "bs_only")


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _save(fig, path, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _heatmaps(output, rows, manifest):
    methods = sorted({r["method"] for r in rows})
    configured = manifest.get("scenarios", [])
    if isinstance(configured, dict):
        configured = list(configured)
    present = {r["scenario"] for r in rows}
    scenarios = [s for s in configured if s in present]
    scenarios += [s for s in SCENARIOS if s in present and s not in scenarios]
    scenarios += sorted(present - set(scenarios))
    lookup = {(r["method"], r["scenario"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, .55 * len(methods) + 2)))
    specs = (("success_rate_mean", "Success rate", 0., 1., "YlGn"),
             ("restricted_team_time_mean", "Restricted team time (s)", None, None, "YlOrRd"))
    for ax, (key, title, low, high, cmap) in zip(axes, specs):
        data = np.full((len(methods), len(scenarios)), np.nan)
        for i, method in enumerate(methods):
            for j, scenario in enumerate(scenarios):
                row = lookup.get((method, scenario), {})
                value = row.get(key)
                if value is None and key == "success_rate_mean":
                    value = row.get("success_rate")
                data[i, j] = np.nan if value is None else float(value)
        image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=low, vmax=high)
        for i, j in np.ndindex(data.shape):
            if np.isfinite(data[i, j]):
                red, green, blue, _ = image.cmap(image.norm(data[i, j]))
                luminance = .2126 * red + .7152 * green + .0722 * blue
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if luminance < .48 else "black")
        ax.set(xticks=range(len(scenarios)), xticklabels=scenarios,
               yticks=range(len(methods)), yticklabels=methods, title=title)
        ax.tick_params(axis="x", rotation=45)
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle("RL: mean across training seeds; rules: mean across evaluation episodes")
    _save(fig, output / "policy_comparison.png")


def _learning(output, manifest):
    variants = [v.get("name", v) if isinstance(v, dict) else v for v in manifest["variants"]]
    seeds = manifest.get("training_seeds", manifest.get("seeds", []))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = ("#2563eb", "#059669", "#dc2626")
    for color, variant in zip(colors, variants):
        reward_runs, success_runs = [], []
        for seed in seeds:
            base = output / "train" / f"{variant}_seed{seed}"
            training = _csv(base / "training.csv")
            reward_runs.append({int(float(r["epoch"])): float(r["mean_step_reward"]) for r in training})
            grouped = defaultdict(list)
            for row in _csv(base / "episodes.csv"):
                grouped[int(float(row["epoch"]))].append(float(row["team_success"]))
            success_runs.append({epoch: np.mean(values) for epoch, values in grouped.items()})
        for ax, runs, title in ((axes[0], reward_runs, "Mean step reward"),
                                (axes[1], success_runs, "Completed-episode success by epoch")):
            epochs = sorted(set.union(*(set(run) for run in runs)))
            values = np.asarray([[run.get(e, np.nan) for e in epochs] for run in runs])
            valid = np.isfinite(values).any(axis=0)
            epochs, values = np.asarray(epochs)[valid], values[:, valid]
            ax.plot(epochs, np.nanmedian(values, axis=0), color=color, label=variant)
            ax.fill_between(epochs, np.nanmin(values, axis=0), np.nanmax(values, axis=0),
                            color=color, alpha=.16)
            ax.set(title=title, xlabel="Epoch")
    axes[0].set_ylabel("Reward / step")
    axes[1].set(ylabel="Success rate", ylim=(-.05, 1.05))
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Median and min–max across training seeds; rewards across distributions are not directly comparable")
    _save(fig, output / "learning_curves.png")


def _trajectories(output, manifest):
    seed = manifest.get("training_seeds", manifest.get("seeds"))[0]
    variants = [v.get("name", v) if isinstance(v, dict) else v for v in manifest["variants"]]
    if "pure_fixed_cpu" in variants and "pure_fixed" in variants:
        variants.remove("pure_fixed")
    methods = ("goal", "sa", *variants)
    scenarios = ("nominal", "crossing", "shifted_goals")
    columns = len(methods)
    fig = plt.figure(figsize=(3.6 * columns, 10))
    for row, scenario in enumerate(scenarios):
        traces, limits = {}, []
        for method in methods:
            base = (output / "eval" / "rules" if method in ("goal", "sa")
                    else output / "eval" / f"{method}_seed{seed}")
            path = base / "trajectories" / scenario / (method if method in ("goal", "sa") else "mappo") / "trajectory.npz"
            with np.load(path) as data:
                traces[method] = {key: data[key] for key in data.files}
            trace = traces[method]
            limits.extend((trace["positions"].reshape(-1, 3), trace["goals"],
                           trace["target_positions"], trace["bs_position"][None]))
            if "estimated_target_positions" in trace:
                limits.append(trace["estimated_target_positions"])
        points = np.concatenate(limits)
        low, high = points.min(0), points.max(0)
        pad = np.maximum((high - low) * .06, 1.)
        for col, method in enumerate(methods):
            ax = fig.add_subplot(3, columns, row * columns + col + 1, projection="3d")
            trace = traces[method]
            positions, goals = trace["positions"], trace["goals"]
            for i, color in enumerate(("#2563eb", "#dc2626")):
                ax.plot(*positions[:, i].T, color=color, label=f"UAV {i + 1}" if row == col == 0 else None)
                ax.scatter(*goals[i], color=color, marker="X", label="goals" if row == col == i == 0 else None)
            ax.plot(*trace["target_positions"].T, color="black", linewidth=1.2,
                    label="target truth (diagnostic only)" if row == col == 0 else None)
            if "estimated_target_positions" in trace:
                ax.plot(*trace["estimated_target_positions"].T, color="#10b981", linestyle="--",
                        linewidth=1, label="EKF estimate" if row == col == 0 else None)
            ax.scatter(*trace["bs_position"], marker="^", color="#f59e0b",
                       label="BS" if row == col == 0 else None)
            ax.set_title(f"{scenario} / {method}", fontsize=9)
            ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                   xlim=(low[0]-pad[0], high[0]+pad[0]),
                   ylim=(low[1]-pad[1], high[1]+pad[1]),
                   zlim=(low[2]-pad[2], high[2]+pad[2]))
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.suptitle("First training seed and first evaluation seed; truth is offline diagnosis only",
                 y=.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .965),
               ncol=len(labels), fontsize=8)
    _save(fig, output / "trajectory_comparison.png", rect=(0, 0, 1, .925))


def make_study_plots(output, summary=None):
    output = Path(output)
    manifest = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
    if summary is None:
        summary = json.loads((output / "study_summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"] if isinstance(summary, dict) else summary
    _heatmaps(output, rows, manifest)
    _learning(output, manifest)
    _trajectories(output, manifest)
    return [output / name for name in ("policy_comparison.png", "learning_curves.png",
                                        "trajectory_comparison.png")]
