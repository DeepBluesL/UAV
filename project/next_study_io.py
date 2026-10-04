"""Shared parsing helpers for the final multi-budget study artifacts."""

import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "restricted_team_time", "safety_interventions", "collisions", "boundary_requests",
    "tracking_prefix_rmse", "rho_prefix_mean", "communication_prefix_rate",
    "tracking_prefix_late_rmse", "position_nees_prefix_mean", "episode_return",
)
TRACKING_METRICS = METRICS[4:-1]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in tuple(row.items()):
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def load_manifest(output):
    path = Path(output) / "study_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"evaluated", "complete"}:
        raise RuntimeError(f"Study is not evaluated (status={manifest.get('status')!r})")
    return manifest


def variant_name(value):
    return value.get("name") if isinstance(value, dict) else value


def variant_specs(manifest):
    variants = manifest["variants"]
    if isinstance(variants, dict):
        return [dict(value, name=name) if isinstance(value, dict) else {"name": name}
                for name, value in variants.items()]
    return variants


def variant_epochs(value, manifest):
    if isinstance(value, dict):
        for key in ("evaluation_epochs", "budget_epochs", "epochs"):
            if key in value:
                epochs = value[key]
                return [int(epochs)] if np.isscalar(epochs) else [int(x) for x in epochs]
        if "epoch" in value:
            return [int(value["epoch"])]
    epochs = [int(x) for x in manifest.get(
        "budget_epochs", [manifest.get("primary_epoch", 100)])]
    if isinstance(value, dict) and "max_epochs" in value:
        epochs = [epoch for epoch in epochs if epoch <= int(value["max_epochs"])]
    return epochs


def finite_mean(rows, key):
    values = [float(row[key]) for row in rows
              if row.get(key) is not None and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def value(row, metric):
    for key in (metric + "_mean", metric):
        result = row.get(key)
        if result is not None:
            return float(result)
    return None


def require_eval_seeds(rows, requested, label):
    requested = {int(x) for x in requested}
    scenarios = {row["scenario"] for row in rows}
    methods = {row["method"] for row in rows}
    if not scenarios or not methods:
        raise ValueError(f"No evaluation rows for {label}")
    for scenario in scenarios:
        for method in methods:
            present = {int(row["seed"]) for row in rows
                       if row["scenario"] == scenario and row["method"] == method}
            if present != requested:
                raise ValueError(
                    f"Incomplete evaluation seeds for {label}/{scenario}/{method}: "
                    f"expected {sorted(requested)}, got {sorted(present)}")


def require_study_groups(rows, manifest, label, methods=None, suite_path=None):
    configured = manifest.get("scenarios")
    suite = manifest.get("suite")
    if configured is None and suite_path is not None and Path(suite_path).exists():
        configured = json.loads(Path(suite_path).read_text(encoding="utf-8")).get("scenarios")
    if configured is None and isinstance(suite, dict):
        configured = suite.get("scenarios")
    if configured:
        expected = set(configured if not isinstance(configured, dict) else configured.keys())
        present = {row["scenario"] for row in rows}
        if present != expected:
            raise ValueError(f"Incomplete scenarios for {label}: expected {sorted(expected)}, got {sorted(present)}")
    if methods:
        present = {row["method"] for row in rows}
        if present != set(methods):
            raise ValueError(f"Incomplete methods for {label}: expected {sorted(methods)}, got {sorted(present)}")
