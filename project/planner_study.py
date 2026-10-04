"""Reproducible development runner for navigation and belief-aware planners."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import zipfile

import numpy as np

from .annealing import SimulatedAnnealingController
from .artifacts import write_csv, write_json
from .benchmark import evaluate_episode
from .benchmark_metrics import paired_comparisons, summarize
from .config import EnvConfig, RewardConfig
from .env import DualUAVEnv
from .evaluate import GoalController
from .mpc import MPCController
from .sensing_planning import SensingAwareMPCController, SensingAwareSAController

PROJECT = Path(__file__).resolve().parent


def controller(name, settings, config, uncertainty_ref):
    family = settings["family"]
    if family == "goal":
        return GoalController(config)
    sensing = float(settings.get("sensing_weight", 0.0))
    communication = float(settings.get("communication_weight", 0.0))
    kwargs = dict(sensing_weight=sensing, communication_weight=communication,
                  uncertainty_ref=uncertainty_ref)
    search = dict(settings.get("search", {}))
    if family == "sa":
        return (SimulatedAnnealingController(config, **search) if sensing == communication == 0
                else SensingAwareSAController(config, **kwargs, **search))
    if family == "mpc":
        return (MPCController(config, **search) if sensing == communication == 0
                else SensingAwareMPCController(config, **kwargs, **search))
    raise ValueError(f"unknown planner family for {name}: {family}")


def configurations(names):
    raw = json.loads((PROJECT / "closed_loop_config.json").read_text(encoding="utf-8"))
    base = EnvConfig(**raw["environment"])
    reward = RewardConfig(**raw["reward"])
    result = {}
    for name in names:
        if name == "nominal":
            result[name] = base
        elif name == "crossing":
            result[name] = replace(
                base, uav_initial=np.array([[30., -25., 70.], [30., 25., 70.]]),
                uav_goal_positions=np.array([[130., 25., 70.], [130., -25., 70.]]))
        else:
            raise ValueError(f"unknown scenario: {name}")
    return result, reward


def snapshot(output):
    archive = output / "source_snapshot.zip"
    excluded = {"output", "experiments", "__pycache__"}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(PROJECT.rglob("*")):
            relative = path.relative_to(PROJECT)
            if path.is_file() and not excluded.intersection(relative.parts):
                if path.suffix in {".py", ".json", ".md"}:
                    bundle.write(path, Path("project") / relative)
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def family_pairs(rows, grid):
    result = []
    for family, reference in grid.get("family_references", {}).items():
        methods = {name for name, settings in grid["methods"].items()
                   if settings.get("comparison_group", settings["family"]) == family}
        selected = [row for row in rows if row["method"] in methods]
        result.extend(paired_comparisons(selected, reference))
    return result


def run(args):
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        raise ValueError("Keep output inside project")
    output.mkdir(parents=True, exist_ok=False)
    (output / "grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
    snapshot_hash = snapshot(output)
    scenarios, reward = configurations(grid["scenarios"])
    root = PROJECT.parent
    command = ("python -m project.planner_study --grid " + str(args.grid)
               + " --output " + str(args.output) + " --seeds "
               + " ".join(map(str, args.seeds)))
    manifest = {
        "status": "running", "grid": grid, "seeds": args.seeds,
        "environment": {name: asdict(cfg) for name, cfg in scenarios.items()},
        "reward": asdict(reward), "reproduce": command,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_snapshot": "source_snapshot.zip", "source_snapshot_sha256": snapshot_hash,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                        capture_output=True, text=True).stdout.strip(),
        "completed_episodes": 0,
    }
    manifest["controller_settings"] = {
        scenario: {method: getattr(controller(method, settings, config,
                                               float(grid["uncertainty_ref"])), "settings", {})
                   for method, settings in grid["methods"].items()}
        for scenario, config in scenarios.items()}
    write_json(output / "manifest.json", manifest)
    rows = []
    started = perf_counter()
    for scenario, config in scenarios.items():
        for method, settings in grid["methods"].items():
            policy = controller(method, settings, config, float(grid["uncertainty_ref"]))
            env = DualUAVEnv(config, reward)
            for seed in args.seeds:
                row, trace = evaluate_episode(policy, env, seed, int(grid["prefix_steps"]))
                row.update(scenario=scenario, method=method, seed=seed,
                           prefix_steps=int(grid["prefix_steps"]))
                rows.append(row)
                trace_dir = output / "trajectories" / scenario / method / str(seed)
                trace_dir.mkdir(parents=True)
                np.savez_compressed(trace_dir / "trajectory.npz", **trace)
                write_csv(output / "episodes.csv", rows)
                manifest["completed_episodes"] = len(rows)
                manifest["wall_seconds"] = perf_counter() - started
                write_json(output / "manifest.json", manifest)
                print(f"{scenario}/{method}/{seed}: success={row['team_success']}", flush=True)
    summaries = summarize(rows)
    paired = family_pairs(rows, grid)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired.csv", paired)
    manifest.update(status="complete", wall_seconds=perf_counter() - started)
    write_json(output / "manifest.json", manifest)
    print(f"Planner study complete: {len(rows)} episodes; output={output}")
    return rows, summaries, paired


def parse_args():
    parser = argparse.ArgumentParser(description="Run a named sensing-planner grid.")
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error("use distinct nonnegative seeds")
    return args


if __name__ == "__main__":
    run(parse_args())
