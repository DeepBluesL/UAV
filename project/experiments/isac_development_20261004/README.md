# ISAC 顺序研究：开发阶段证据归档（2026-10-04）

本目录逐字节归档五项已经完成的开发／验证实验，不含模型权重（`.pt`），也不含最终测试数据。各目录保留原始 CSV、NPZ 轨迹、图、报告、运行 manifest、源码哈希；规划实验还保留运行前生成的 `source_snapshot.zip`。

## 内容与范围

| 目录 | 范围 | 回合／记录 |
|---|---|---:|
| `sensing_diagnostics_step1_dev_4101_4110` | 10 个开发种子 × 2 场景 × 7 条可行路径 | 140 回合 |
| `filter_calibration_dev_4101_4110` | Goal、2 场景、10 个开发种子、每回合前 10 步；逐源 NIS 与位置 NEES | 20 回合，600 条逐源记录 |
| `sensing_planning_pilot_4101` | 单一开发种子、2 场景、7 个规划设置 | 14 回合 |
| `sensing_planning_dev_4101_4110` | 10 个开发种子、2 场景、13 个设置 | 260 回合 |
| `sensing_planning_validation_4201_4205` | 5 个预留验证种子、2 场景、7 个冻结设置 | 70 回合 |

开发种子为 4101–4110，验证种子为 4201–4205。两者用途分离；本归档没有使用预注册最终测试种子 5101–5130。失败回合、任务退化和不显著结果均保留。

## 单行复现命令

命令必须使用新的输出目录，因为运行器拒绝覆盖已有结果。

```powershell
D:\anaconda3\envs\uav\python.exe -B -m project.sensing_diagnostics --output project/output/sensing_diagnostics_step1_dev_4101_4110_REPRO --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
D:\anaconda3\envs\uav\python.exe -B -m project.calibration_diagnostics --output project/output/filter_calibration_dev_4101_4110_REPRO --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
D:\anaconda3\envs\uav\python.exe -B -m project.planner_study --grid project/planner_study_pilot.json --output project/output/sensing_planning_pilot_4101_REPRO --seeds 4101
D:\anaconda3\envs\uav\python.exe -B -m project.planner_study --grid project/planner_study_dev_grid.json --output project/output/sensing_planning_dev_4101_4110_REPRO --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
D:\anaconda3\envs\uav\python.exe -B -m project.planner_study --grid project/planner_study_validation_grid.json --output project/output/sensing_planning_validation_4201_4205_REPRO --seeds 4201 4202 4203 4204 4205
```

## 复现边界

各 manifest 记录基准 Git commit、dirty-tree 状态、运行配置和环境。STEP1 与滤波诊断保存所依赖源码的逐文件 SHA-256；三个规划目录保存运行前完整源码 ZIP 及其 SHA-256。由于实验运行时工作树包含尚未提交的并行开发改动，单独 checkout manifest 中的 commit 不足以精确复现；应优先使用归档哈希核对对应源码，规划实验应以各自 `source_snapshot.zip` 为权威运行快照。

`archive_manifest.json` 对本目录除自身外的所有文件给出相对路径、字节数与 SHA-256，用于验证复制和后续传输没有改变字节。
