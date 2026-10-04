# 可复核实验快照

本目录保存适合随代码共享的逐回合CSV、统计汇总、配置、报告和结果图。训练权重及调试产物仍在忽略的 `project/output/` 下；新闭环归档保留用于重绘的首种子轨迹。

- [2026-09-27：六种方法、五场景、3000回合](benchmark_20260927/REPORT.md)
- [主要结论与解读](../docs/BASELINE_RESULTS.md)
- [运行方法与统计口径](../docs/BASELINES.md)

这是冻结结果快照。新实验请先写入新的 `project/output/` 子目录，确认完成后再单独归档；不要覆盖已有记录。

## SA 与边界／速度受控实验

`limits_sa_20260927/` 保存旧 proxy 冻结模型的 900 回合实验。6 场景 × 3 方法 × 50 个评估种子；不与新 EKF 结果混合。分析见 [LIMITS_SA_RESULTS.md](../docs/LIMITS_SA_RESULTS.md)。

## EKF、随机场景与残差 RL

[closed_loop_20260927](closed_loop_20260927/README.md) 保存主比较 3960 回合及独立辅助比较 990 回合，含 12 次训练日志、逐回合数据和首种子轨迹；不含权重。实测解读见 [CLOSED_LOOP_RESULTS.md](../docs/CLOSED_LOOP_RESULTS.md)。

## 感知可控性、滤波一致性与规划开发实验

[isac_development_20261004](isac_development_20261004/README.md) 保留140条路径回合、20次十步滤波诊断、14次规划试跑、260次开发比较与70次独立验证，包括失败和未复现的改善。不含最终测试集或训练权重。
