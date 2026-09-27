# UAV：双无人机导航与协同感知

**简体中文** | [English](README.en.md)

基于 MAPPO 的双 UAV 通感一体化仿真：两个独立 Actor 控制无人机运动，一个集中 Critic 学习团队回报。任务是在期限内到达各自终点，同时兼顾感知、通信与安全。

## 项目结构

```text
UAV/
├── project/                 当前主实现，环境、物理公式和 PPO 均在包内
│   ├── train.py             训练、评估与导航对照入口
│   ├── example_config.json  实验配置（默认使用 CUDA）
│   ├── tests/               单元与集成测试
│   ├── docs/                详细使用说明、公式迁移和验证记录
│   └── output/              本地模型、日志与图表，不上传 Git
├── legacy/                  旧环境、旧训练器与 2uav 参考代码
├── requirements.txt         安装入口，引用 project/requirements.txt
├── README.md                中文说明
└── README.en.md             英文说明
```

## 安装

下面命令均适用于 **Windows CMD**，每条命令占一行。已有仓库和 `uav` 环境时，直接进入仓库根目录并激活环境即可。

```bat
git clone https://github.com/DeepBluesL/UAV.git
cd UAV
conda create -n uav python=3.12 -y
conda activate uav
```

先安装 GPU 版 PyTorch，再安装项目依赖。下面使用 CUDA 13.0 wheel 源；其他显卡或驱动组合请按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择构建。

```bat
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 训练与评估

所有命令从仓库根目录（`project` 的上一级）执行。正式训练：

```bat
python -m project.train --config project/example_config.json --seed 7 --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7
```

先检查流程可运行一次短训练；它不能用来判断收敛：

```bat
python -m project.train --config project/example_config.json --epochs 2 --steps-per-epoch 128 --seed 7 --eval-seed 101 102 --output project/output/quick_check
```

评估训练结束后保存的模型：

```bat
python -m project.train --mode evaluate --checkpoint project/output/run_seed7/policy.pt --device cuda --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7_eval
```

重新绘图与运行测试：

```bat
python -m project.replot --output project/output/run_seed7
python -B -m unittest discover -s project/tests -v
```

示例配置使用 GPU，CPU 运行时在训练或评估命令中加 `--device cpu`。神经网络可用 GPU；NumPy 物理环境仍在 CPU 串行执行。`--eval-seed` 与 `--eval-seeds` 等价。

结果保存在指定的 `project/output/` 子目录，包括模型 `policy.pt`、训练/评估日志和 PNG 图表。轨迹图展示基站、两架 UAV，以及黑飞的真实/估计轨迹。**每次实验使用不同输出目录**，同名文件会被覆盖；当前不支持断点续训。

## 从哪里修改

| 内容 | 位置 |
| --- | --- |
| 场景、奖励权重、学习率和训练预算 | [example_config.json](project/example_config.json)，完整字段见 [config.py](project/config.py) |
| 运动、到达退出和观测 | `project/env.py`、`project/observations.py` |
| 奖励表达式 | [rewards.py](project/rewards.py) |
| 网络、GAE 和 PPO 更新 | `project/core.py`、`project/ppo.py` |
| 信道、SINR、CRB/PCRB | `project/channels.py`、`communication.py`、`sensing.py`、`measurements.py`、`crb.py`、`pcrb.py` |
| 训练统计与绘图 | `project/train.py`、`metrics.py`、`plot.py`、`plot_trajectory.py` |

详细说明见 [使用指南](project/docs/GUIDE.md)、[物理公式迁移](project/docs/PHYSICS_MIGRATION.md) 和 [验证记录](project/docs/VERIFICATION.md)。旧代码的用途和运行方式见 [legacy/README.md](legacy/README.md)。

黑飞真值不直接输入 Actor/Critic 观测；当前目标估计采用加噪代理，尚未实现实际测量与滤波闭环，PCRB 也不等于实测跟踪误差。

PPO 参考 OpenAI Spinning Up，第三方许可保留在 [LICENSE-spinningup.txt](project/LICENSE-spinningup.txt)。该许可用于相关第三方代码，不代表仓库全部代码的授权声明。
