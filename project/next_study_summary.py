"""Hierarchical final-study aggregation and bilingual reporting."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .artifacts import write_csv, write_json
from .benchmark_metrics import summarize
from .next_study_io import (METRICS, finite_mean, load_manifest, read_csv,
                            require_eval_seeds, require_study_groups, value,
                            variant_epochs, variant_name,
                            variant_specs)


def _seed_row(variant, seed, epoch, scenario, rows, spec, primary,
              prefix_steps=None, demo_interactions=None):
    uses_demo = (bool(spec.get("demo", spec.get("use_demo", False)))
                 or bool(spec.get("demonstration_steps", 0))
                 or str(spec.get("initialization", "")).endswith("_bc")) if isinstance(spec, dict) else False
    result = {
        "kind": "checkpoint", "method": variant, "train_seed": seed,
        "epoch": epoch, "budget_epoch": epoch, "primary": epoch == primary,
        "demo": uses_demo,
        "demo_interactions": demo_interactions, "prefix_steps": prefix_steps,
        "scenario": scenario, "evaluation_episodes": len(rows),
        "success_rate": float(np.mean([bool(row["team_success"]) for row in rows])),
    }
    result.update({metric: finite_mean(rows, metric) for metric in METRICS})
    return result


def _across_training_seeds(seed_rows, expected_seeds):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["method"], row["epoch"], row["scenario"])].append(row)
    output = []
    for (method, epoch, scenario), rows in sorted(grouped.items()):
        present = {int(row["train_seed"]) for row in rows}
        if present != set(expected_seeds):
            raise ValueError(f"Incomplete training seeds for {method}/epoch{epoch}/{scenario}")
        result = {key: rows[0][key] for key in
                  ("kind", "method", "epoch", "budget_epoch", "primary", "demo",
                   "demo_interactions", "prefix_steps")}
        result.update(scenario=scenario, training_seeds=len(rows),
                      evaluation_episodes=sum(row["evaluation_episodes"] for row in rows),
                      uncertainty_unit="training_seed")
        for metric in ("success_rate", *METRICS):
            values = [row[metric] for row in rows if row.get(metric) is not None]
            result[metric + "_mean"] = float(np.mean(values)) if values else None
            result[metric + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None)
        output.append(result)
    return output


def _rule_rows(rows, primary, prefix_steps=None):
    output = []
    for row in summarize(rows):
        prefix = (prefix_steps.get(row["scenario"])
                  if isinstance(prefix_steps, dict) else prefix_steps)
        converted = {"kind": "rule", "epoch": None, "budget_epoch": None,
                     "primary": True, "demo": False, "training_seeds": 0,
                     "prefix_steps": prefix,
                     "uncertainty_unit": "evaluation_episode", **row}
        converted["success_rate_mean"] = row["success_rate"]
        converted["success_rate_std"] = None
        output.append(converted)
    return output


def _paired_ablations(rows, manifest):
    configured = manifest.get("paired_ablations", [])
    if not configured:
        names = {row["method"] for row in rows}
        candidates = (("residual_nav_v1", "residual_nav"),
                      ("residual_nav", "residual_sense_low"),
                      ("residual_nav", "residual_sense_high"),
                      ("residual_nav", "residual_sense_legacy"))
        configured = [pair for pair in candidates if set(pair) <= names]
    lookup = {(row["method"], row["train_seed"], row["epoch"], row["scenario"]): row
              for row in rows}
    output = []
    for pair in configured:
        left, right = ((pair["left"], pair["right"])
                       if isinstance(pair, dict) else pair)
        for epoch in sorted({row["epoch"] for row in rows}):
            for scenario in sorted({row["scenario"] for row in rows}):
                differences = defaultdict(list)
                for seed in manifest["training_seeds"]:
                    a = lookup.get((left, int(seed), epoch, scenario))
                    b = lookup.get((right, int(seed), epoch, scenario))
                    if a is None or b is None:
                        continue
                    for metric in ("success_rate", *METRICS[:-1]):
                        if a.get(metric) is not None and b.get(metric) is not None:
                            differences[metric].append(b[metric] - a[metric])
                if differences:
                    item = {"left": left, "right": right, "epoch": epoch,
                            "scenario": scenario}
                    for metric, values in differences.items():
                        item[metric + "_difference_n"] = len(values)
                        item[metric + "_difference_mean"] = float(np.mean(values))
                        item[metric + "_difference_std"] = (
                            float(np.std(values, ddof=1)) if len(values) > 1 else None)
                    output.append(item)
    return output


def _report(output, manifest, rows, ablations, english=False):
    primary = int(manifest.get("primary_epoch", 100))
    title = "Final study report" if english else "最终研究报告"
    lines = [f"# {title}", "", ("Each checkpoint is averaged within its evaluation episodes, then equally across training seeds. Rule uncertainty describes evaluation noise."
             if english else "每个 checkpoint 先在评估回合内平均，再对训练种子等权平均；规则方法的不确定性仅表示评估噪声。"),
             ("Checkpoints were fixed by the planned budgets; no test result selected a checkpoint."
              if english else "Checkpoint 由预定预算固定，未按测试结果选择最高 checkpoint。"), "",
             f"Primary epoch: {primary}; budgets: {manifest.get('budget_epochs', [])}; training seeds: {manifest['training_seeds']}.", "",
             "## Primary comparison", "",
             "| scenario | method | success | restricted time | prefix RMSE | rho | comm | late RMSE | NEES | safety | collisions |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["kind"] == "checkpoint" and row["epoch"] != primary:
            continue
        def show(metric):
            mean = value(row, metric)
            if mean is None:
                return "—"
            std = row.get(metric + "_std") if row["kind"] == "checkpoint" else None
            return f"{mean:.4g} ± {std:.3g}" if std is not None else f"{mean:.4g}"
        lines.append(f"| {row['scenario']} | {row['method']} | {show('success_rate')} | "
                     f"{show('restricted_team_time')} | {show('tracking_prefix_rmse')} | "
                     f"{show('rho_prefix_mean')} | {show('communication_prefix_rate')} | "
                     f"{show('tracking_prefix_late_rmse')} | {show('position_nees_prefix_mean')} | "
                     f"{show('safety_interventions')} | {show('collisions')} |")
    lines += ["", "## Same-budget paired ablations", "",
              "Differences are right minus left, paired by training seed. Raw returns are omitted because reward configurations differ.", ""]
    for row in ablations:
        lines.append(f"- {row['left']} → {row['right']}, epoch {row['epoch']}, {row['scenario']}: "
                     f"Δsuccess={row.get('success_rate_difference_mean', float('nan')):.4g}, "
                     f"Δtime={row.get('restricted_team_time_difference_mean', float('nan')):.4g}, "
                     f"n={row.get('restricted_team_time_difference_n', 0)}")
    lines += ["", "## Figures", "", "- primary_comparison_heatmap.png", "- navigation_tracking_tradeoff.png",
              "- budget_learning_curves.png", "- trajectory_comparison.png", "",
              ("Navigation outcomes (success, restricted time, safety) are reported separately from fixed-prefix tracking, communication, late RMSE, and NEES. Points with failures or collisions are not called Pareto-efficient."
               if english else "导航结果（成功率、限制时间、安全）与固定前缀 RMSE、rho、通信、后段 RMSE、NEES 分开报告；存在失败或碰撞的点不称为 Pareto 有效。"), "",
              ("Raw returns are not interpreted across reward configurations."
               if english else "不同奖励配置之间不把原始回报差异解释为性能提升。")]
    name = "REPORT.en.md" if english else "REPORT.md"
    (output / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_next_study(output):
    output, manifest = Path(output), load_manifest(output)
    frozen_suite = output / "configs" / "suite.json"
    seeds, eval_seeds = [int(x) for x in manifest["training_seeds"]], manifest["evaluation_seeds"]
    primary, seed_rows = int(manifest.get("primary_epoch", 100)), []
    for spec in variant_specs(manifest):
        variant = variant_name(spec)
        for seed in seeds:
            initialization = output / "train" / f"{variant}_seed{seed}" / "initialization.json"
            init = json.loads(initialization.read_text(encoding="utf-8")) if initialization.exists() else {}
            demo_interactions = next((init[key] for key in
                                      ("demonstration_steps", "demo_interactions",
                                       "bc_interactions", "demo_transitions")
                                      if key in init), None)
            for epoch in variant_epochs(spec, manifest):
                path = output / "eval" / f"{variant}_seed{seed}_epoch{epoch}" / "episodes.csv"
                episodes = read_csv(path); require_eval_seeds(episodes, eval_seeds, path.parent.name)
                require_study_groups(
                    episodes, manifest, path.parent.name, ["mappo"], frozen_suite)
                metadata = path.parent / "manifest.json"
                meta = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
                groups = defaultdict(list)
                for row in episodes: groups[row["scenario"]].append(row)
                prefixes = meta.get("scenario_prefix_steps", {})
                seed_rows += [_seed_row(variant, seed, epoch, scenario, group, spec, primary,
                                        prefixes.get(scenario, meta.get("prefix_steps")),
                                        demo_interactions)
                              for scenario, group in sorted(groups.items())]
    summary = _across_training_seeds(seed_rows, seeds)
    rules = read_csv(output / "eval" / "rules" / "episodes.csv")
    require_eval_seeds(rules, eval_seeds, "rules")
    require_study_groups(
        rules, manifest, "rules", manifest.get("rule_methods"), frozen_suite)
    rules_manifest = output / "eval" / "rules" / "manifest.json"
    rules_meta = json.loads(rules_manifest.read_text(encoding="utf-8")) if rules_manifest.exists() else {}
    summary += _rule_rows(rules, primary,
                          rules_meta.get("scenario_prefix_steps", rules_meta.get("prefix_steps")))
    ablations = _paired_ablations(seed_rows, manifest)
    write_csv(output / "seed_summary.csv", seed_rows); write_csv(output / "study_summary.csv", summary)
    payload = {"manifest": manifest, "rows": summary, "paired_ablations": ablations}
    write_json(output / "study_summary.json", payload)
    _report(output, manifest, summary, ablations); _report(output, manifest, summary, ablations, True)
    return {"seed_rows": seed_rows, **payload}
