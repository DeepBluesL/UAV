"""Offline EKF consistency diagnostics; filter behavior and noise are unchanged."""

import argparse
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np

from .artifacts import write_csv, write_json
from .env import DualUAVEnv
from .evaluate import GoalController
from .sensing_diagnostics import scenario_config

CHI2_POSITION_95 = 7.8147279
CHI2_MEASUREMENT_95 = 9.487729
SOURCES = ("bs", "uav0", "uav1")
WINDOWS = ((1, 3, "steps_1_3"), (4, 10, "steps_4_10"), (1, 10, "steps_1_10"))


def run_episode(config, scenario, seed):
    env = DualUAVEnv(config, seed=seed)
    policy = GoalController(config)
    obs, _, info = env.reset(seed=seed)
    rows = []
    while not info["terminated"] and info["step"] < 10:
        action, _ = policy.act(obs, info["active"])
        obs, _, _, _, _, info = env.step(action)
        error = info["estimated_target_state"][:3] - info["target_state"][:3]
        covariance = info["tracking_covariance"][:3, :3]
        nees = _quadratic(error, covariance)
        tracker = env.physics.tracker
        for source in range(3):
            residual = tracker.last_residuals[source]
            innovation = tracker.last_innovations[source]
            delivered = bool(np.isfinite(residual).all())
            rows.append({
                "scenario": scenario, "seed": seed, "step": info["step"],
                "source": SOURCES[source], "delivered": int(delivered),
                "position_error": float(np.linalg.norm(error)), "position_nees": nees,
                "position_covered_95": int(nees <= CHI2_POSITION_95),
                "nis": _quadratic(residual, innovation) if delivered else np.nan,
            })
    return rows


def _quadratic(vector, covariance):
    try:
        return float(vector @ np.linalg.solve(covariance, vector))
    except np.linalg.LinAlgError:
        return float(vector @ np.linalg.pinv(covariance) @ vector)


def summarize(rows):
    summaries = []
    for scenario in sorted({row["scenario"] for row in rows}):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for low, high, window in WINDOWS:
            selected = [row for row in scenario_rows if low <= row["step"] <= high]
            # Position fields repeat across sources; use BS rows once per step.
            position = [row for row in selected if row["source"] == "bs"]
            summaries.append(_summary_row(scenario, window, "position", position, "position_nees",
                                          CHI2_POSITION_95, "position_covered_95"))
            for source in SOURCES:
                delivered = [row for row in selected
                             if row["source"] == source and row["delivered"]]
                summaries.append(_summary_row(scenario, window, source, delivered, "nis",
                                              CHI2_MEASUREMENT_95))
    return summaries


def _summary_row(scenario, window, source, rows, field, threshold, coverage_field=None):
    values = np.asarray([row[field] for row in rows], dtype=float)
    covered = ([row[coverage_field] for row in rows] if coverage_field
               else [value <= threshold for value in values])
    return {"scenario": scenario, "window": window, "source": source, "samples": len(rows),
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "median": float(np.median(values)) if len(values) else np.nan,
            "maximum": float(np.max(values)) if len(values) else np.nan,
            "chi2_95_threshold": threshold,
            "empirical_coverage_95": float(np.mean(covered)) if len(covered) else np.nan}


def run(args):
    output = args.output.resolve()
    project = Path(__file__).resolve().parent
    if not output.is_relative_to(project):
        raise ValueError("Keep diagnostic output inside project")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scenario in args.scenarios:
        config = scenario_config(scenario)
        for seed in args.seeds:
            rows.extend(run_episode(config, scenario, seed))
    summaries = summarize(rows)
    write_csv(output / "steps.csv", rows)
    write_csv(output / "summary.csv", summaries)
    sources = [project / name for name in ("calibration_diagnostics.py", "tracking.py",
                                            "physics.py", "env.py", "config.py")]
    root = project.parent
    write_json(output / "manifest.json", {
        "status": "complete", "evaluation_seeds": args.seeds, "scenarios": args.scenarios,
        "controller": "GoalController", "steps": 10,
        "thresholds": {"position_nees_chi2_3_95": CHI2_POSITION_95,
                       "source_nis_chi2_4_95": CHI2_MEASUREMENT_95},
        "interpretation": "Coverage is descriptive; correlated steps are not independent tests. No filter parameters or floors were tuned.",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                        capture_output=True, text=True).stdout.strip(),
        "source_tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                                 capture_output=True, text=True).stdout.strip()),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    })
    print(f"Filter calibration diagnostic complete: {output}")
    return rows, summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Offline EKF NEES/NIS development diagnostic.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=("nominal", "crossing"),
                        default=["nominal", "crossing"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
