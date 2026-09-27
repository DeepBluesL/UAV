# `project`：双 UAV 协同导航与感知

这是仓库中的自包含 MAPPO 实现。安装、训练、评估、公式口径和输出说明见 [完整使用指南](docs/GUIDE.md)；仓库整体结构见 [根 README](../README.md)。

所有命令均从仓库根目录执行，并按 Windows CMD 单行书写：

```bat
python -m pip install -r project/requirements.txt
python -m project.train --help
python -B -m unittest discover -s project/tests -v
```

## 文档

- [使用指南](docs/GUIDE.md)：训练、评估、配置、输出与实现口径
- [物理公式迁移](docs/PHYSICS_MIGRATION.md)：旧公式到包内模块的对应关系
- [验证记录](docs/VERIFICATION.md)：迁移测试、历史硬件与实验结果
- [Spinning Up 许可证](LICENSE-spinningup.txt)

## 模块与修改入口

| 分组 | 文件 | 主要修改内容 |
| --- | --- | --- |
| 配置与入口 | `config.py`, `train.py`, `evaluate.py` | 场景、超参数、训练与评估流程 |
| 环境与观测 | `env.py`, `observations.py`, `rewards.py` | 状态推进、观测、奖励和终止条件 |
| 物理模型 | `physics.py`, `channels.py`, `communication.py`, `sensing.py` | 信道、通信与感知计算 |
| 估计与界 | `measurements.py`, `crb.py`, `pcrb.py` | 测量模型、CRB 与 PCRB |
| MAPPO | `core.py`, `ppo.py` | Actor/Critic、Buffer、GAE 与更新 |
| 结果与绘图 | `artifacts.py`, `metrics.py`, `plot.py`, `plot_trajectory.py`, `replot.py` | 保存、统计和可视化 |
| 验证 | `tests/`, `formula_demo.py` | 单元/集成测试与公式演示 |

示例配置位于 [example_config.json](example_config.json)，默认输出目录为 `project/output/`。
