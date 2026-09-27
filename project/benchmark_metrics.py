"""Episode enrichment and uncertainty summaries for benchmark evaluation."""

from collections import defaultdict
from math import sqrt

import numpy as np


METRICS = (
    "episode_return", "restricted_team_time", "success_time",
    "total_path_length", "total_energy_proxy", "goal_distance_mean",
    "communication_rate", "rho_pos_mean", "rho_prefix_mean",
    "safety_interventions", "collisions", "boundary_requests",
    "policy_ms_per_step",
)
PAIRED_METRICS = ("episode_return", "restricted_team_time", "rho_prefix_mean")


def _number(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def enrich_episode(row, trace, config, prefix_steps=10):
    """Return a copy of an EpisodeRecorder row with comparison metrics."""
    if prefix_steps < 1:
        raise ValueError("prefix_steps must be positive")
    result = dict(row)
    success = bool(row["team_success"])
    elapsed = float(row["length"]) * float(config.slot_duration)
    result["restricted_team_time"] = (
        elapsed if success else float(config.max_steps) * float(config.slot_duration))
    result["success_time"] = elapsed if success else None
    result["total_path_length"] = float(row["path_length_0"] + row["path_length_1"])
    result["total_energy_proxy"] = float(row["energy_proxy_0"] + row["energy_proxy_1"])
    result["goal_distance_mean"] = float(
        (row["goal_distance_0"] + row["goal_distance_1"]) / 2)
    counts = np.asarray([row["active_steps_0"], row["active_steps_1"]], dtype=float)
    rates = [row["communication_rate_0"], row["communication_rate_1"]]
    valid = [(count, _number(rate)) for count, rate in zip(counts, rates)
             if count > 0 and _number(rate) is not None]
    result["communication_rate"] = (
        float(sum(count * rate for count, rate in valid) / sum(count for count, _ in valid))
        if valid else None)
    rho = np.asarray(trace["rho_pos"], dtype=float).reshape(-1)
    result["rho_prefix_mean"] = (
        float(np.mean(rho[1:prefix_steps + 1])) if rho.size - 1 >= prefix_steps else None)
    return result


def wilson_interval(successes, episodes, z=1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if episodes < 1:
        return None, None
    p = successes / episodes
    scale = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / scale
    radius = z * sqrt((p * (1 - p) + z * z / (4 * episodes)) / episodes) / scale
    return center - radius, center + radius


def _bootstrap_mean(values, rng, samples=2000):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        value = float(values[0])
        return value, value
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return float(low), float(high)


def _metric_summary(result, metric, rows, rng):
    values = [_number(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    result[f"{metric}_n"] = len(values)
    if not values:
        for suffix in ("mean", "std", "ci_low", "ci_high"):
            result[f"{metric}_{suffix}"] = None
        return
    result[f"{metric}_mean"] = float(np.mean(values))
    result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) >= 2 else None
    low, high = _bootstrap_mean(values, rng)
    result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = low, high


def summarize(rows):
    """Group episode rows by scenario and method and summarize uncertainty."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["method"])].append(row)
    summaries = []
    for group_index, ((scenario, method), group) in enumerate(sorted(groups.items())):
        episodes = len(group)
        successes = sum(bool(row["team_success"]) for row in group)
        low, high = wilson_interval(successes, episodes)
        result = {
            "scenario": scenario, "method": method, "episodes": episodes,
            "successes": successes, "success_rate": successes / episodes,
            "success_ci_low": low, "success_ci_high": high,
        }
        rng = np.random.default_rng(group_index)
        for metric in METRICS:
            _metric_summary(result, metric, group, rng)
        summaries.append(result)
    return summaries


def paired_comparisons(rows, reference="goal"):
    """Compare matched seeds; equal seeds need not share physical noise sequences."""
    indexed = {}
    for row in rows:
        key = (row["scenario"], row["method"], row["seed"])
        if key in indexed:
            raise ValueError(f"Duplicate scenario/method/seed row: {key}")
        indexed[key] = row
    scenarios = sorted({row["scenario"] for row in rows})
    results = []
    for scenario in scenarios:
        methods = sorted({row["method"] for row in rows if row["scenario"] == scenario})
        reference_rows = {seed: row for (s, m, seed), row in indexed.items()
                          if s == scenario and m == reference}
        for method in methods:
            if method == reference:
                continue
            method_rows = {seed: row for (s, m, seed), row in indexed.items()
                           if s == scenario and m == method}
            result = {"scenario": scenario, "method": method, "reference": reference}
            rng = np.random.default_rng(len(results))
            for metric in PAIRED_METRICS:
                differences = []
                for seed in sorted(method_rows.keys() & reference_rows.keys()):
                    left, right = _number(method_rows[seed].get(metric)), _number(reference_rows[seed].get(metric))
                    if left is not None and right is not None:
                        differences.append(left - right)
                result[f"{metric}_n"] = len(differences)
                result[f"{metric}_mean"] = float(np.mean(differences)) if differences else None
                low, high = _bootstrap_mean(differences, rng) if differences else (None, None)
                result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = low, high
            results.append(result)
    return results
