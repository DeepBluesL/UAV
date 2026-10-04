# UAV：双无人机导航与协同感知

**简体中文** | [English](README.en.md)

基于 MAPPO 的双 UAV 通感一体化仿真。两个独立 Actor 控制友机运动，一个集中式 Critic 学习团队回报；基站采用确定性波束规则。两架 UAV 需要在期限内安全到达各自终点，并在航程中协同感知一个运动目标。单机到达后停止动作及感知上传。

`ekf` 模式实现“带噪量测 → 融合滤波 → 目标估计 → 后续决策”。目标真值用于仿真与离线评分，不直接输入 Actor、Critic 或规划器。量测噪声仍是未经过实物标定的模型。

## 当前内容与实验状态

- 已支持纯 RL、Goal 导航基础上的残差 RL，以及 Goal、PD、人工势场、MPC、模拟退火等规则对照。
- 新研究包含感知型 SA/MPC、v2 观测、奖励消融、课程学习和 Goal 行为克隆初始化；代码与开发验证证据已合入 `main`。
- **截至 2026-10-05，24 个模型的正式研究尚未完成最终汇总与归档。** 已完成结果见[历史闭环实验](project/docs/CLOSED_LOOP_RESULTS.md)和[本轮开发验证归档](project/experiments/isac_development_20261004/README.md)；它们与本轮最终测试分开解释。

## 项目结构与配置

```text
UAV/
├── project/          当前实现：环境、物理模型、滤波、PPO 与实验入口
│   ├── docs/         使用指南、实验协议与结果说明
│   ├── experiments/  已发布的实验数据与图表
│   ├── tests/        单元与集成测试
│   └── output/       本地模型、日志与结果；Git 忽略
├── legacy/           历史代码，仅作参考
├── requirements.txt  项目依赖入口
└── README.en.md      英文说明
```

| 配置 | 用途 |
|---|---|
| [closed_loop_config.json](project/closed_loop_config.json) | 单次 EKF 闭环训练，CPU 采样推理、CUDA 批量更新 |
| [next_study_protocol.json](project/next_study_protocol.json) | 完整研究：24 个模型、3 个训练种子、48 个预定快照评估 |
| [example_config.json](project/example_config.json) | 历史 `proxy` 模式，用于复核旧实验 |

## 安装

以下均为 **Windows CMD 单行命令**。所有 Python 命令从仓库根目录执行；已有仓库和环境时跳过创建步骤。

```cmd
git clone https://github.com/DeepBluesL/UAV.git
cd UAV
conda create -n uav python=3.12 -y
conda activate uav
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

上例使用 CUDA 13.0 的 PyTorch wheel 源；其他显卡和驱动组合按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择构建。项目环境使用 NumPy 在 CPU 上执行物理仿真，GPU 负责批量网络更新，因此 GPU 利用率不会一直很高。

## 单次训练、评估与画图

先用短训练检查流程；这不代表策略已收敛：

```cmd
python -m project.train --config project/closed_loop_config.json --control-mode residual --epochs 2 --steps-per-epoch 128 --seed 7 --eval-seed 101 102 --output project/output/quick_check
```

运行闭环残差 RL；将 `--control-mode residual` 改为 `--control-mode pure` 即为纯 RL：

```cmd
python -m project.train --config project/closed_loop_config.json --control-mode residual --seed 7 --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7
```

评估读取模型保存的配置。下面使用 CPU 做小批量推理，训练仍按配置使用 CUDA 更新：

```cmd
python -m project.train --mode evaluate --checkpoint project/output/run_seed7/policy.pt --device cpu --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7_eval
python -m project.replot --output project/output/run_seed7
```

结果包括 `policy.pt`、CSV/JSON 日志和 PNG 图表；轨迹图显示基站、友机、各自终点，以及目标的真实和估计轨迹。每次运行使用新输出目录：单次训练可能覆盖同名文件，批量研究拒绝覆盖已有目录。目前没有保存优化器状态，不支持原样断点续训。若全部使用 CPU，训练命令添加 `--device cpu --rollout-device cpu`。

## 对照与完整研究

比较默认六种规则方法：随机、Goal、PD、人工势场、导航 MPC 和 SA：

```cmd
python -m project.benchmark --config project/closed_loop_config.json --seed-start 2001 --episodes 30 --output project/output/comparison_rules
```

运行完整的新研究，包括训练、固定预算评估和汇总；它明显长于一次训练：

```cmd
python -m project.next_study --config project/next_study_protocol.json --output project/output/my_next_study --stage all --jobs 3
```

`--jobs 3` 表示三个独立进程。主比较采用相同的 204,800 次联合环境交互预算，行为克隆示范也计入预算；部分变体另保存 300/500 epoch 快照。全部正式评估计划包含 15 个场景、30 个评估种子，共 23,850 回合。分阶段命令、种子用途和参数修改见[新研究指南](project/docs/NEXT_STUDY_GUIDE.md)；历史九模型研究见[闭环实验指南](project/docs/CLOSED_LOOP_STUDY.md)。

先看成功率、安全干预、碰撞和完成时间，再比较相同时间窗口的跟踪 RMSE、协方差与通信。协方差小不等于实际误差小，不同奖励配置的总回报也不能直接比较。当前正式方案中，纯／残差 RL 的同奖励比较限于导航；非零感知奖励消融在残差 RL 内进行。

## 从哪里修改

| 内容 | 文件 |
|---|---|
| 场景、超参数与实验组 | `project/config.py`、对应 JSON 配置 |
| 运动、退出、观测与奖励 | `project/env.py`、`observations.py`、`observation_specs.py`、`rewards.py` |
| 信道、通信与感知物理模型 | `project/physics.py`、`channels.py`、`communication.py`、`sensing.py`、`crb.py`、`pcrb.py` |
| 仿真量测与融合滤波 | `project/tracking.py` |
| 网络、PPO 与动作映射 | `project/core.py`、`ppo.py`、`rollout.py`、`control.py` |
| 随机场景、课程与模仿初始化 | `project/training_scenarios.py`、`domain_randomization.py`、`behavior_cloning.py` |
| 感知规划与结果汇总 | `project/sensing_planning.py`、`nominal_links.py`、`next_study_summary.py`、`next_study_plots.py` |

```cmd
python -B -m unittest discover -s project/tests -v
```

更多说明：[模块索引](project/README.md) · [使用指南](project/docs/GUIDE.md) · [物理公式迁移](project/docs/PHYSICS_MIGRATION.md) · [对照实验指南](project/docs/BASELINES.md) · [实验归档](project/experiments/README.md)。

PPO 参考 OpenAI Spinning Up；相关第三方代码的许可见 [LICENSE-spinningup.txt](project/LICENSE-spinningup.txt)，该许可不代表仓库全部代码的授权声明。
