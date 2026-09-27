# Closed-loop study / 闭环实验归档（2026-09-27）

[中文实测解读](../../docs/CLOSED_LOOP_RESULTS.md) · [English results](../../docs/CLOSED_LOOP_RESULTS.en.md) · [完整统计](REPORT.md)

## Scope / 范围

主比较为 `pure_fixed_cpu`、`pure_randomized`、`residual_randomized`，各三个训练种子（7、17、27），每模型 204800 个联合环境步，CPU 采样、CUDA 更新、CPU 评估。11 个场景，每模型每场景 30 个评估种子。主比较包括 2970 个学习策略回合与 990 个规则策略回合，共 3960 回合。

`auxiliary/` 单独保存先前已启动的 `pure_fixed` GPU 采样模型的训练日志与 990 回合评估，不进入主汇总。该目录不用于选择主比较模型。旧 proxy 模型的 900 回合实验在相邻 `limits_sa_20260927/`，不可混池。

The primary analysis matches rollout/update/evaluation devices across all nine learned policies. The three earlier GPU-rollout runs are kept under `auxiliary/`, excluded from the primary summary. Both successful and failed outcomes are retained.

## Files / 文件

- `study_manifest.json`：主比较完成状态、预算、种子和各阶段源码哈希；每个 `eval/*/manifest.json` 包含实际评估配置及 checkpoint 哈希。
- `configs/`：实际执行配置。`configs/base.json` 是最初快照，各模型的配置及 `train/*/config.json` 才是其设备设置的依据。
- `train/`：主比较的完整 training/episode CSV、训练与验证汇总、配置和控制台日志；学习曲线可由这些文件重建。
- `eval/`：主比较与规则的逐回合 CSV、汇总、元数据、报告、各场景第一评估种子的 `trajectory.npz`。
- `study_summary.csv`、`seed_summary.csv`：跨训练种子统计和逐 checkpoint 统计。
- `tracking_ablation_statistics.json`：Goal 路径上的跟踪消融及配对 bootstrap 参数。
- `policy_comparison.png`、`learning_curves.png`、`trajectory_comparison.png`、`tracking_ablation.png`：四张主图。
- `source_initial.zip` / `source_split_device.zip`：辅助 GPU 训练与主比较训练的源码快照；`source_final.zip` 是归档时的报告／绘图代码、测试与新配置。
- `execution_note.json`：设备拆分和评估进程重启记录。所有训练完整结束，没有停止或丢弃未完成模型；读取配置旧类失败后，仅在新进程补完剩余评估。
- `auxiliary_original_manifest.json`：最初调度计划留档，不用于重算主比较。
- `runtime.json` / `artifact_sha256.json`：运行库版本与归档文件校验值。

权重未上传，仍保存在本地 `project/output/closed_loop_study_20260927/train/`。因此此归档可以重汇总和重绘，不能单独重新运行模型评估。每个原始评估报告内的逐场景 PNG 链接对应本地原始输出；为减少重复文件，归档只保留四张主图和用于重绘的 NPZ。

Policy weights are intentionally omitted. The archive supports re-aggregation and plotting from saved data, not checkpoint inference. Source ZIP files use repository-relative paths. Historical manifests retain the original local file paths as provenance.

## Replot / 重绘

从仓库根目录，在具备项目依赖的环境中执行以下单行 Windows CMD 命令；仅重写派生汇总和图表，不重新训练或评估：

```bat
python -m project.study --stage summarize --output project/experiments/closed_loop_20260927
```

归档校验清单对应首次保存的文件；不同 Matplotlib 版本重新绘图后，PNG 哈希可能改变。新实验应使用新的 `project/output/` 目录，方法和命令见 [中文指南](../../docs/CLOSED_LOOP_STUDY.md) / [English guide](../../docs/CLOSED_LOOP_STUDY.en.md)。
