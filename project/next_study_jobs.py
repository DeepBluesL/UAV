"""Prepare frozen study inputs and run independent training subprocesses."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import numpy as np
import torch

from .artifacts import write_json

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "project"


def prepare_study(protocol, output, jobs):
    """Capture code/configuration before the first training process starts."""
    if protocol["status"] != "frozen":
        raise ValueError("Freeze development choices before starting formal training")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    base = json.loads((ROOT / protocol["base_config"]).read_text(encoding="utf-8"))
    suite = json.loads((ROOT / protocol["suite"]).read_text(encoding="utf-8"))
    write_json(output / "configs" / "base.json", base)
    write_json(output / "configs" / "suite.json", suite)
    sources = sorted(PROJECT.glob("*.py")) + sorted(PROJECT.glob("*.json"))
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(ROOT))
        for source in sorted((PROJECT / "tests").glob("test_*.py")):
            archive.write(source, source.relative_to(ROOT))
    manifest = {**protocol, "status": "training", "parallel_jobs": jobs,
                "frozen_scenarios": list(suite["scenarios"]),
                "rule_methods": protocol.get("rule_methods", ["goal", "mpc", "sa", "sensing_mpc", "sensing_sa"]),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": torch.__version__, "platform": platform.platform()},
                "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True).stdout.strip()}
    manifest["source_tree_dirty"] = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    write_json(output / "study_manifest.json", manifest)
    tasks = []
    for variant, changes in protocol["variants"].items():
        for seed in protocol["training_seeds"]:
            name = f"{variant}_seed{seed}"
            settings = deepcopy(base)
            settings.setdefault("environment", {}).update(observation_version=changes["observation_version"])
            settings.setdefault("reward", {}).update(
                sensing=changes["sensing"], rho_ref=1.0,
                sensing_penalty=changes.get("sensing_penalty", "legacy"))
            settings.setdefault("ppo", {}).update(
                epochs=changes["max_epochs"], steps_per_epoch=protocol["steps_per_epoch"],
                seed=seed, control_mode=changes["control_mode"],
                training_distribution=changes["training_distribution"],
                initialization=changes.get("initialization", "random"),
                demonstration_steps=changes.get("demonstration_steps", 0),
                checkpoint_epochs=[epoch for epoch in protocol["budget_epochs"]
                                   if epoch <= changes["max_epochs"]],
                device="cuda", rollout_device="cpu")
            config_path = output / "configs" / f"{name}.json"
            write_json(config_path, settings)
            tasks.append((name, config_path, output / "train" / name))
    # Run the primary navigation references first; list order is reproducible.
    return manifest, tasks


def run_training(task, validation_seeds):
    name, config_path, directory = task
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "project.train", "--config", str(config_path),
               "--output", str(directory), "--no-plots", "--eval-seeds", *map(str, validation_seeds)]
    child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with (directory / "console.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=stream,
                                   stderr=subprocess.STDOUT)
        write_json(directory / "process.json", {"pid": process.pid, "command": command})
        code = process.wait()
    write_json(directory / "process.json", {"pid": process.pid, "returncode": code, "command": command})
    if code:
        raise RuntimeError(f"{name} failed; inspect {directory / 'console.log'}")
    print(f"Training complete: {name}", flush=True)
