"""把统一评估结果写成可阅读的 Markdown；不自动宣称算法总体优越。"""

from pathlib import Path


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def interval(row, metric, digits=2):
    mean = row.get(metric + "_mean")
    if mean is None:
        return "—"
    low, high = row[metric + "_ci_low"], row[metric + "_ci_high"]
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def write_report(output, manifest, summaries, paired):
    checkpoint = manifest["checkpoint"]
    seeds = manifest["evaluation_seeds"]
    prefix = manifest["prefix_steps"]
    ekf = all(cfg.get("sensing_mode", "proxy") == "ekf" for cfg in manifest["scenarios"].values())
    uncertainty = "EKF 位置协方差迹" if ekf else "位置 PCRB 代理"
    lines = [
        "# 规则基线与冻结 MAPPO 对比", "",
        f"共 {manifest['total_episodes']} 回合；每个场景、每种方法使用同一组 {len(seeds)} 个评估种子。",
        "完整种子、场景、参数、源码哈希与运行环境见 `manifest.json`。", "",
    ]
    if checkpoint:
        lines += [
            f"MAPPO 训练种子为 {checkpoint['training_seed']}，训练预算为 "
            f"{checkpoint['training_environment_steps']} 个联合环境步。",
            f"Checkpoint SHA-256：`{checkpoint['sha256']}`。", "",
        ]
    lines += [
        "## 统计口径", "",
        "- 成功率区间使用 95% Wilson；连续均值区间使用 2000 次评估回合 bootstrap。",
        "- 区间只反映给定策略下的评估波动；单个训练种子不能证明训练稳定性。",
        "- 同种子用于配对；活动源退出时 RNG 消耗可不同，不是严格共同随机数。",
        "- 限制完成时间：成功用实际时间，失败用该场景期限；越低越好。",
        "- 成功耗时均值只统计成功回合，样本数见 summary.csv，不能单独评价失败多的方法。",
        f"- 前缀不确定性为第 1 至 {prefix} 步的{uncertainty}均值，排除初始先验；不足前缀长度记缺失。",
        "- tracking_prefix_rmse 是同一前缀的三维位置误差 RMSE（m），真值仅用于离线评分。",
        "- 全回合不确定性、跟踪误差、能耗和路程受回合长度影响，优先比较同前缀指标。",
        "- 通信满足率按两机活动步数加权；能耗为无量纲代理，不是焦耳。",
        "- 碰撞数与安全干预数分别记录，零碰撞可能来自环境拒绝危险动作。",
        "- 决策耗时仅计预热后的 policy.act，不含环境物理计算；这是单进程参考计时。",
        "- 规则方法无需训练；MPC/SA 联合使用两机观测规划，属于集中导航参考。",
        "- 场景均在评估前指定；是否分布外需要结合该模型的训练分布判断。随机场景训练已覆盖部分交叉、目标和CSI变化。", "",
    ]
    for scenario in manifest["scenarios"]:
        rows = [row for row in summaries if row["scenario"] == scenario]
        lines += [f"## {scenario}", "", manifest["suite"]["scenarios"][scenario]["description"], "",
                  "| 方法 | 成功/回合 | 成功率 95% CI | 回报均值 [95% CI] | 限制完成时间/s [95% CI] |",
                  "| --- | --- | --- | --- | --- |"]
        for row in rows:
            success = f"{row['success_rate']:.1%} [{row['success_ci_low']:.1%}, {row['success_ci_high']:.1%}]"
            lines.append(f"| {row['method']} | {row['successes']}/{row['episodes']} | {success} | "
                         f"{interval(row, 'episode_return')} | {interval(row, 'restricted_team_time')} |")
        lines += ["", "| 方法 | 前缀不确定性/m² | 前缀样本数 | 路程/m | 能耗代理 | 通信满足率 | 越界请求 | 安全干预 | 碰撞 | 决策/ms |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            lines.append(f"| {row['method']} | {number(row['rho_prefix_mean_mean'], 6)} | "
                         f"{row['rho_prefix_mean_n']} | {number(row['total_path_length_mean'])} | "
                         f"{number(row['total_energy_proxy_mean'])} | {number(row['communication_rate_mean'], 4)} | "
                         f"{number(row['boundary_requests_mean'])} | {number(row['safety_interventions_mean'])} | "
                         f"{number(row['collisions_mean'])} | {number(row['policy_ms_per_step_mean'], 3)} |")
        lines.append("")
        if (Path(output) / f"comparison_{scenario}.png").exists():
            lines += [f"![{scenario} comparison](comparison_{scenario}.png)", ""]
    lines += [f"## 与 {manifest['statistics']['paired_reference']} 的配对差值", "",
              f"差值方向为 method − {manifest['statistics']['paired_reference']}；回报越大越好，时间与不确定性越小越好。",
              "以下区间是探索性比较，没有多重比较校正。", "",
              "| 场景 | 方法 | 回报差 [95% CI] | 时间差/s [95% CI] | 前缀不确定性差/m² [95% CI] |",
              "| --- | --- | --- | --- | --- |"]
    for row in paired:
        lines.append(f"| {row['scenario']} | {row['method']} | {interval(row, 'episode_return')} | "
                     f"{interval(row, 'restricted_team_time')} | {interval(row, 'rho_prefix_mean', 6)} |")
    lines += ["", "## 结论边界", "",
              ("本次使用仿真量测与 EKF 融合；协方差属于滤波器置信度，真实误差另行统计，尚非实物传感器验证。" if ekf else
               "目标估计仍采用 truth + noise 代理，没有实际测量与滤波闭环；PCRB 不能作为实测跟踪误差。"),
              "本表评价的是给定场景下的任务执行与仿真指标，未比较 IPPO、MADDPG 等重新训练的学习方法，",
              "也未使用多个训练种子；不能据此宣称 MAPPO 的总体最优性或真实系统的感知收益。", ""]
    (Path(output) / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
