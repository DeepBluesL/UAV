"""Evaluate prespecified budget snapshots without selecting on test results."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import write_json
from .next_study_jobs import ROOT


def evaluation_jobs(output, manifest):
    tasks = []
    for variant, changes in manifest["variants"].items():
        for seed in manifest["training_seeds"]:
            for epoch in manifest["budget_epochs"]:
                if epoch <= changes["max_epochs"]:
                    checkpoint = output / "train" / f"{variant}_seed{seed}" / f"policy_epoch{epoch}.pt"
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    tasks.append((f"{variant}_seed{seed}_epoch{epoch}", checkpoint, ["mappo"]))
    tasks.append(("rules", None, manifest.get(
        "rule_methods", ["goal", "mpc", "sa", "sensing_mpc", "sensing_sa"])))
    return tasks


def _run_evaluation(output, manifest, task):
    name, checkpoint, methods = task
    directory = output / "eval" / name
    if directory.exists():
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected_count = len(manifest["evaluation_seeds"]) * len(saved["scenarios"]) * len(methods)
        valid = (saved["status"] == "complete" and saved["total_episodes"] == expected_count
                 and saved["evaluation_seeds"] == manifest["evaluation_seeds"]
                 and saved["methods"] == methods)
        if checkpoint is not None:
            valid = valid and saved["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        suite = json.loads((output / "configs" / "suite.json").read_text(encoding="utf-8"))
        valid = valid and saved["suite"] == suite
        if not valid:
            raise ValueError(f"Incomplete or mismatched evaluation: {directory}; inspect its process/log")
        print(f"Verified completed evaluation: {name}", flush=True)
        return
    command = [sys.executable, "-B", "-m", "project.benchmark",
               "--suite", str(output / "configs" / "suite.json"), "--methods", *methods,
               "--reference", methods[0], "--device", "cpu", "--output", str(directory),
               "--eval-seeds", *map(str, manifest["evaluation_seeds"]), "--no-plots"]
    if checkpoint is None:
        command += ["--config", str(output / "configs" / "base.json")]
    else:
        command += ["--checkpoint", str(checkpoint)]
    logs = output / "eval_logs"
    logs.mkdir(exist_ok=True)
    child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with (logs / f"{name}.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=stream,
                                   stderr=subprocess.STDOUT)
        write_json(logs / f"{name}.process.json", {"pid": process.pid, "command": command})
        code = process.wait()
    write_json(logs / f"{name}.process.json", {"pid": process.pid, "returncode": code, "command": command})
    if code:
        raise RuntimeError(f"Evaluation failed: {name}; see {logs / (name + '.log')}")
    print(f"Evaluation complete: {name}", flush=True)


def evaluate_next_study(output, jobs=3):
    output = Path(output)
    manifest_path = output / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = evaluation_jobs(output, manifest)
    manifest["status"] = "evaluating"
    write_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_evaluation, output, manifest, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    manifest.update(status="evaluated", evaluated_snapshots=len(tasks) - 1)
    write_json(manifest_path, manifest)
