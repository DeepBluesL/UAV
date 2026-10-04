"""STEP 1 sensing-controllability probes.

Run: python -m project.sensing_diagnostics --output project/output/sensing_probe --seeds 101 102 103
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from .artifacts import write_csv, write_json
from .config import EnvConfig
from .diagnostic_paths import PATH_NAMES, WaypointController
from .env import DualUAVEnv

COMPONENTS = ("range", "azimuth", "elevation", "range_rate")
CRB_TO_EKF_ORDER = (0, 2, 1, 3)
SOURCES = ("bs", "uav0", "uav1")


def scenario_config(name):
    base = EnvConfig(sensing_mode="ekf")
    if name == "nominal":
        return base
    if name == "crossing":
        return replace(base, uav_initial=np.array([[30., -25., 70.], [30., 25., 70.]]),
                       uav_goal_positions=np.array([[130., 25., 70.], [130., -25., 70.]]))
    raise ValueError(f"unknown scenario: {name}")


def variance_floors(config):
    return np.array([config.measurement_range_std_floor ** 2,
                     config.measurement_angle_std_floor ** 2,
                     config.measurement_angle_std_floor ** 2,
                     config.measurement_range_rate_std_floor ** 2])


def run_episode(config, scenario, path_name, seed, prefix_steps=10):
    env = DualUAVEnv(config, seed=seed)
    policy = WaypointController(config, path_name)
    obs, _, info = env.reset(seed=seed)
    policy.reset(seed)
    steps, delivered = [], []
    safety_interventions = boundary_requests = collisions = 0
    while not info["terminated"]:
        action, _ = policy.act(obs, info["active"], deterministic=True)
        obs, _, _, _, _, info = env.step(action)
        safety_interventions += int(info["safety_intervention"])
        boundary_requests += int(np.count_nonzero(info["boundary_requests"]))
        collisions += int(info["collision"])
        active = np.asarray(info["active_before"], dtype=bool)
        comm_ok = np.asarray(info["communication_ok"], dtype=bool)
        position_error = info["estimated_target_state"][:3] - info["target_state"][:3]
        position_covariance = np.asarray(info["tracking_covariance"][:3, :3])
        try:
            position_nees = float(position_error @ np.linalg.solve(position_covariance,
                                                                    position_error))
        except np.linalg.LinAlgError:
            position_nees = float(position_error @ np.linalg.pinv(position_covariance)
                                  @ position_error)
        steps.append({
            "scenario": scenario, "path": path_name, "seed": seed, "step": info["step"],
            "tracking_error": info["tracking_position_error"], "rho_pos": info["rho_pos"],
            "position_nees": position_nees,
            "active_uavs": int(active.sum()), "communication_ok": int((comm_ok & active).sum()),
            "minimum_separation": info["minimum_separation"],
            "safety_intervention": int(info["safety_intervention"]),
            "collision": int(info["collision"]),
            "measurement_count": info["measurement_count"],
        })
        crbs = np.asarray(info["source_crbs"])
        for source in np.flatnonzero(info["measurement_source_mask"]):
            ekf_crb = crbs[source, list(CRB_TO_EKF_ORDER)]
            for component, raw, floor in zip(COMPONENTS, ekf_crb, variance_floors(config)):
                delivered.append((int(source), component, float(raw), float(floor)))
    prefix = steps[:prefix_steps]
    full_prefix = len(prefix) == prefix_steps
    opportunities = sum(row["active_uavs"] for row in prefix)
    episode = {
        "scenario": scenario, "path": path_name, "seed": seed,
        "team_success": int(info["team_success"]), "termination_reason": info["termination_reason"],
        "episode_steps": info["step"], "episode_time": info["step"] * config.slot_duration,
        "arrival_time_0": info["arrival_times"][0], "arrival_time_1": info["arrival_times"][1],
        "path_length_0": info["path_lengths"][0], "path_length_1": info["path_lengths"][1],
        "safety_interventions": safety_interventions, "boundary_requests": boundary_requests,
        "collisions": collisions, "minimum_separation": min(r["minimum_separation"] for r in steps),
        "rmse_first10": (float(np.sqrt(np.mean([r["tracking_error"] ** 2 for r in prefix])))
                           if full_prefix else np.nan),
        "rho_mean_first10": (float(np.mean([r["rho_pos"] for r in prefix]))
                              if full_prefix else np.nan),
        "position_nees_mean_first10": (float(np.mean([r["position_nees"] for r in prefix]))
                                        if full_prefix else np.nan),
        "communication_fraction_first10": (sum(r["communication_ok"] for r in prefix) / opportunities
                                             if full_prefix and opportunities else np.nan),
        "bs_updates": int(info["measurement_counts"][0]),
        "uav0_updates": int(info["measurement_counts"][1]),
        "uav1_updates": int(info["measurement_counts"][2]),
    }
    return episode, steps, delivered


def aggregate_sources(records):
    buckets = defaultdict(list)
    for scenario, path_name, source, component, raw, floor in records:
        buckets[(scenario, path_name, source, component)].append((raw, floor))
    rows = []
    for (scenario, path_name, source, component), values in sorted(buckets.items()):
        raw = np.asarray([v[0] for v in values]); floor = np.asarray([v[1] for v in values])
        ratio = raw / floor
        rows.append({"scenario": scenario, "path": path_name, "source": SOURCES[source],
                     "component": component, "delivered_samples": len(values),
                     "floor_variance": floor[0], "floor_hit_fraction": float(np.mean(raw <= floor)),
                     "raw_crb_mean": float(np.mean(raw)), "raw_to_floor_mean": float(np.mean(ratio)),
                     "raw_to_floor_median": float(np.median(ratio)),
                     "raw_to_floor_min": float(np.min(ratio)), "raw_to_floor_max": float(np.max(ratio))})
    return rows


def run_diagnostics(args):
    started = perf_counter()
    output = args.output.resolve()
    project_dir = Path(__file__).resolve().parent
    if not output.is_relative_to(project_dir):
        raise ValueError("Keep diagnostic output inside project")
    output.mkdir(parents=True, exist_ok=False)
    episodes, step_rows, source_records = [], [], []
    for scenario in args.scenarios:
        config = scenario_config(scenario)
        for path_name in args.paths:
            for seed in args.seeds:
                episode, steps, delivered = run_episode(config, scenario, path_name, seed)
                episodes.append(episode); step_rows.extend(steps)
                source_records.extend((scenario, path_name, *row) for row in delivered)
    source_rows = aggregate_sources(source_records)
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "steps.csv", step_rows)
    write_csv(output / "source_crb.csv", source_rows)
    if not args.no_plots:
        from .diagnostic_plots import plot_diagnostics
        plot_diagnostics(output, episodes, source_rows)
    root = project_dir.parent
    source_files = [project_dir / name for name in
                    ("sensing_diagnostics.py", "diagnostic_paths.py", "diagnostic_plots.py",
                     "env.py", "physics.py", "tracking.py", "control.py", "config.py")]
    manifest = {
        "status": "complete", "evaluation_seeds": args.seeds, "scenarios": args.scenarios,
        "paths": args.paths, "prefix_steps": 10, "runtime_seconds": perf_counter() - started,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform()},
        "scenario_configs": {name: asdict(scenario_config(name)) for name in args.scenarios},
        "controller_settings": {"switch_radius": 8.0, "lateral_offset_m": 12.0,
                                "altitude_offset_m": 10.0,
                                "action_rule": "project.control.goal_raw_action"},
        "nees_note": "Position NEES is an offline covariance-scale diagnostic; correlated time steps are not treated as independent chi-square samples.",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                        capture_output=True, text=True).stdout.strip(),
        "source_tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                                 capture_output=True, text=True).stdout.strip()),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "rng_note": "Tracker source noise is aligned by seed. Channel draws may diverge after path-dependent exits, so this is not a fully paired causal counterfactual.",
    }
    write_json(output / "manifest.json", manifest)
    print(f"Sensing diagnostics complete: {len(episodes)} episodes; output={output}")
    return episodes, step_rows, source_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose whether feasible flight paths control sensing quality.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--scenarios", nargs="+", choices=("nominal", "crossing"), default=["nominal", "crossing"])
    parser.add_argument("--paths", nargs="+", choices=PATH_NAMES, default=list(PATH_NAMES))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.paths)) != len(args.paths):
        parser.error("seeds and paths must not contain duplicates")
    return args


if __name__ == "__main__":
    run_diagnostics(parse_args())
