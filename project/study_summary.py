"""多训练种子 study 的分层汇总与简洁中文报告。"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import write_csv, write_json
from .benchmark_metrics import METRICS, summarize


SUMMARY_METRICS = ("success_rate", *METRICS)


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
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


def _manifest_values(manifest, key, default):
    value = manifest.get(key, default)
    if isinstance(value, dict):
        value = list(value)
    return [item.get("name", item) if isinstance(item, dict) else item for item in value]


def _seed_row(variant, seed, scenario, rows):
    result = {"kind": "checkpoint", "method": variant, "train_seed": seed,
              "scenario": scenario, "evaluation_episodes": len(rows)}
    result["success_rate"] = float(np.mean([bool(r["team_success"]) for r in rows]))
    for metric in METRICS:
        values = [float(r[metric]) for r in rows
                  if r.get(metric) is not None and np.isfinite(float(r[metric]))]
        result[metric] = float(np.mean(values)) if values else None
    return result


def _across_seeds(seed_rows):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["scenario"], row["method"])].append(row)
    output = []
    for (scenario, method), rows in sorted(grouped.items()):
        result = {"kind": "checkpoint", "scenario": scenario, "method": method,
                  "training_seeds": len(rows),
                  "evaluation_episodes": sum(r["evaluation_episodes"] for r in rows)}
        for metric in SUMMARY_METRICS:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            result[f"{metric}_mean"] = float(np.mean(values)) if values else None
            result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result[f"{metric}_min"] = float(np.min(values)) if values else None
            result[f"{metric}_max"] = float(np.max(values)) if values else None
        output.append(result)
    return output


def _report(output, manifest, summary):
    variants = _manifest_values(manifest, "variants", [])
    seeds = _manifest_values(manifest, "training_seeds", manifest.get("seeds", []))
    lines = ["# MAPPO study 汇总", "",
             f"训练变体：{', '.join(map(str, variants))}；独立训练种子：{', '.join(map(str, seeds))}。",
             "每个 checkpoint 先平均其 evaluation episodes，再跨训练种子统计均值、样本标准差和范围；规则方法的区间只表示评估回合不确定性。",
             "这些结果不构成全局最优或训练收敛证明。", "",
             "## 主要图表", "",
             "- [策略比较](policy_comparison.png)",
             "- [训练曲线](learning_curves.png)",
             "- [轨迹比较](trajectory_comparison.png)", "",
             "## 主要任务指标", "",
             "学习方法写作“跨训练种子均值 ± 训练种子样本标准差”；规则方法括号为 evaluation episodes 的 95% 区间，两类区间含义不同。", "",
             "Goal 是独立目标导航，MPC／SA 使用集中式运动模型规划，RL 在执行时使用各自局部 Actor；它们的先验、协调方式和决策耗时不同，应结合安全干预与 `policy_ms_per_step` 解读。", "",
             "回报是进展、距离、时间、能耗、通信、感知和终止项的加权和，只宜在同一场景与同一奖励配置内比较；尤其不能用跨速度场景的回报差异直接声称效率提升。", "",
             "| 场景 | 方法 | 成功率 | 限制时间 (s) | 回报 |", "|---|---|---:|---:|---:|"]
    for row in summary:
        if row["kind"] == "checkpoint":
            def learned(metric):
                mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
                return "—" if mean is None else f"{mean:.4g} ± {std:.3g}" if std is not None else f"{mean:.4g}"
            success, elapsed, reward = (learned("success_rate"),
                                        learned("restricted_team_time"), learned("episode_return"))
        else:
            def rule(metric):
                mean = row.get(f"{metric}_mean", row.get(metric))
                low, high = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
                return "—" if mean is None else f"{mean:.4g} ({low:.3g}, {high:.3g})"
            success = f"{row['success_rate']:.4g} ({row['success_ci_low']:.3g}, {row['success_ci_high']:.3g})"
            elapsed, reward = rule("restricted_team_time"), rule("episode_return")
        lines.append(f"| {row['scenario']} | {row['method']} | {success} | {elapsed} | {reward} |")
    lines += ["", "## 前缀跟踪误差与不确定性", "",
              "`tracking_prefix_rmse` 是 EKF 估计相对诊断真值的 RMSE；`rho_prefix_mean` 是 EKF 后验位置协方差迹（m²），两者不可互换。", "",
              "| 场景 | 方法 | Prefix RMSE | EKF position covariance trace (m²) |", "|---|---:|---:|---:|"]
    for row in summary:
        if row["scenario"] not in ("nominal", "shifted_goals"):
            continue
        def show(key):
            value = row.get(key)
            return "—" if value is None else f"{value:.5g}"
        lines.append(f"| {row['scenario']} | {row['method']} | "
                     f"{show('tracking_prefix_rmse_mean')} | {show('rho_prefix_mean_mean')} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_study(output):
    """读取完成的 study 目录并写分层统计；缺少正式输入时明确报错。"""
    output = Path(output)
    manifest_path = output / "study_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Study is not complete (status={manifest.get('status')!r}); refuse formal summary")
    variants = _manifest_values(manifest, "variants", [])
    seeds = [int(x) for x in _manifest_values(
        manifest, "training_seeds", manifest.get("seeds", []))]
    seed_rows = []
    for variant in variants:
        for seed in seeds:
            path = output / "eval" / f"{variant}_seed{seed}" / "episodes.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            groups = defaultdict(list)
            for row in _read_csv(path):
                groups[row["scenario"]].append(row)
            seed_rows.extend(_seed_row(variant, seed, scenario, rows)
                             for scenario, rows in sorted(groups.items()))
    summary = _across_seeds(seed_rows)
    rules_path = output / "eval" / "rules" / "episodes.csv"
    if rules_path.exists():
        for row in summarize(_read_csv(rules_path)):
            summary.append({"kind": "rule", **row})
    write_csv(output / "seed_summary.csv", seed_rows)
    write_csv(output / "study_summary.csv", summary)
    write_json(output / "study_summary.json", {"manifest": manifest, "rows": summary})
    _report(output, manifest, summary)
    return {"seed_rows": seed_rows, "rows": summary, "manifest": manifest}
