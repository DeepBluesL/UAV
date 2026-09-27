# 双 UAV 导航与协同感知 MAPPO 使用指南

> 此文档保留原代理模式的说明与历史结果。新增仿真量测、EKF、SA、纯／残差 RL 和随机训练的接口见 [闭环实验指南](CLOSED_LOOP_STUDY.md)。


本文所有命令都从仓库根目录执行，并按 Windows CMD 单行书写。

两个独立的局部高斯 Actor 控制两架 UAV 的三维运动，一个集中 Critic 学习团队回报。
任务成功条件是两架 UAV 在期限内安全到达各自终点，途中兼顾感知质量和通信质量。

环境、物理公式、奖励、PPO、训练和绘图全部位于 `project` 内，不依赖 `legacy/test.py`。
本文命令与当前 `train.py` 的接口一致；**先完成运行准备，再执行需要的命令**。

## 1. 运行准备

先进入仓库根目录并创建或激活自己的 Python 环境。需要 GPU 时，先按[仓库安装说明](../../README.md#安装)安装匹配本机驱动的 PyTorch，再安装项目其余依赖：

~~~bat
cd /d "E:\Code\UAV"
python -m pip install -r project/requirements.txt
python -c "import sys, torch; print(sys.executable); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
~~~

- 每条命令占一行；示例中的 `E:\Code\UAV` 请替换为自己的仓库目录。
- 统一用 `python -m project.<模块>` 从仓库根目录运行。
- GPU 用户应先安装 GPU 版 PyTorch，再执行 `project/requirements.txt` 的安装命令。
- 本机迁移时的硬件、版本与验证结果保留在 [VERIFICATION.md](VERIFICATION.md)。

评估种子的正式参数名为 `--eval-seeds`，也支持 `--eval-seed` 别名。查看全部参数：

~~~bat
python -m project.train --help
~~~

## 2. 先跑一次短训练

下面运行 2 个采样和更新批次，每批 128 个联合环境步，使用两层各 32 单元的网络。
训练完成后自动保存模型，使用种子 101、102 评估，并生成图表。

~~~bat
python -m project.train --device cuda --epochs 2 --steps-per-epoch 128 --hidden-size 32 --seed 7 --eval-seeds 101 102 --output project/output/quick_check
~~~

这 256 步只用于检查运行流程，不能用于判断策略是否收敛。
终端每个 epoch 输出一次进度；训练、评估全部完成后会打印成功率与输出目录。

## 3. 按配置文件训练

使用 [example_config.json](../example_config.json) 启动训练：

~~~bat
python -m project.train --config project/example_config.json --seed 7 --eval-seeds 1001 1002 1003 1004 1005 --output project/output/run_seed7
~~~

当前示例配置是 `epochs=100`、`steps_per_epoch=2048`，合计 **204800 个联合环境步**。
网络尺寸采用 `config.py` 默认的 `[128, 128]`；示例配置的 `device="cuda"` 将网络放到 GPU 上。
一次环境步同时执行两架 UAV 的动作；总交互步数不再乘以 UAV 数量。

需要修改训练预算时，可直接用命令行覆盖配置。下面仍是从头开始的一次训练：

~~~bat
python -m project.train --config project/example_config.json --epochs 200 --steps-per-epoch 2048 --seed 7 --output project/output/run_seed7_200epochs
~~~

训练过程中，每个 epoch 更新 `training.csv` 和完整回合的 `episodes.csv`。
`policy.pt` 在全部 epoch 完成后保存；自动评估和绘图也在训练结束后执行。
当前没有中途自动存档、最佳模型筛选或周期性评估。

### 多个训练种子

以下命令依次启动三次独立训练，每次使用相同的评估种子，结果分目录保存：

~~~bat
for %S in (7 8 9) do python -m project.train --config project/example_config.json --seed %S --eval-seeds 1001 1002 1003 1004 1005 --output "project/output/compare_seed%S"
~~~

上面是在 CMD 窗口中直接输入的写法。如果保存到 `.bat` 或 `.cmd` 文件，
将循环变量的所有 `%S` 改成 `%%S`。

## 4. 评估已训练的模型

加载第 2 节产生的模型，冻结策略，只执行评估：

~~~bat
python -m project.train --mode evaluate --checkpoint project/output/quick_check/policy.pt --eval-seeds 101 102 --output project/output/quick_check_eval
~~~

评估第 3 节的模型时，将 checkpoint 换成 `project/output/run_seed7/policy.pt`，
同时为本次评估指定新的输出目录。

- 评估使用 Actor 的确定性动作，不更新网络，也不调用 Critic 选择动作。
- 场景、奖励和网络结构从 checkpoint 读取，不能同时传入 `--config`。
- 评估不能传入 `--epochs`、`--steps-per-epoch`、`--hidden-size` 或 `--seed`；随机种子用 `--eval-seeds` 指定。
- 如需检查不同任务期限，可加 `--max-steps`，但这会改变评估条件；与训练条件比较时应保持一致。
- 默认在 CPU 上加载；`--device` 可显式指定运行设备。

### 当前能否断点续训？

**目前不支持。** `policy.pt` 保存网络权重和配置，没有保存优化器、采样器、环境及随机数状态。
`--checkpoint` 用于 `--mode evaluate`；再次执行训练命令会重新初始化网络。
复用原输出目录也不会恢复训练，反而会覆盖同名结果文件。

## 5. 运行无学习导航对照

使用相同场景和评估种子，检查导航规则能否完成任务：

~~~bat
python -m project.train --mode baseline --config project/example_config.json --eval-seeds 1001 1002 1003 1004 1005 --output project/output/navigation_baseline
~~~

该模式使用 `GoalController`，不训练 PPO，也不生成 `policy.pt`。
它用于核对场景可达性及指标口径；比较自定义场景时，baseline 应使用与训练相同的配置和任务期限。

更多规则基线、人工势场和短时域 MPC 的统一对比见 [BASELINES.md](BASELINES.md)，实际结果见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。

## 6. 修改场景、奖励和 PPO 参数

**生效顺序：`config.py` 默认值 → JSON 中提供的字段 → 命令行覆盖。**
JSON 可只写需要修改的字段，其他字段沿用默认值。
例如，复制一份实验配置后再编辑，便于保留原示例：

~~~bat
copy "project\example_config.json" "project\my_config.json"
~~~

配置分三段；下面的 JSON 可以直接作为 `project/my_config.json` 的内容：

~~~json
{
  "environment": {
    "max_steps": 100,
    "uav_goal_positions": [[150, 70, 90], [120, -30, 30]],
    "imperfect_csi_beta": 0.1
  },
  "reward": {
    "sensing": 0.2,
    "rho_ref": 0.01
  },
  "ppo": {
    "epochs": 100,
    "steps_per_epoch": 2048,
    "hidden_sizes": [128, 128],
    "pi_lr": 0.0003,
    "vf_lr": 0.001,
    "train_pi_iters": 4,
    "train_v_iters": 4,
    "seed": 7,
    "device": "cuda"
  }
}
~~~

~~~bat
python -m project.train --config project/my_config.json --output project/output/custom_run
~~~

| 想调整的内容 | 修改位置 |
| --- | --- |
| 起点、终点、任务期限、速度/加速度限制、物理参数 | JSON 的 `environment`；全部可用字段见 `EnvConfig` |
| 进展、到达、感知等权重和归一化尺度 | JSON 的 `reward`；具体奖励表达式在 `rewards.py` |
| 学习率、折扣、GAE、clip、更新轮数、熵系数 | JSON 的 `ppo`；全部可用字段见 `PPOConfig` |
| 网络层数与每层宽度 | JSON 的 `ppo.hidden_sizes`，例如 `[128, 64]` |
| 临时修改批次数、采样步数、种子 | 命令行参数 |

`--hidden-size 64` 会覆盖 JSON 的网络结构，设置成两层 `[64, 64]`。
命令行没有 `--pi-lr`、`--sensing` 等选项，这些字段通过 JSON 修改。

### 常用命令行参数

| 参数 | 含义 | 未指定时 |
| --- | --- | --- |
| `--mode` | `train` / `evaluate` / `baseline` | `train` |
| `--config` | JSON 配置路径，供训练和 baseline 使用 | 使用 `config.py` 默认值 |
| `--checkpoint` | 评估模型路径 | evaluate 模式必须提供 |
| `--epochs` | 采样和更新的批次数，不是回合数 | JSON 或默认 `100` |
| `--steps-per-epoch` | 每批联合环境交互步数 | JSON 或默认 `2048` |
| `--max-steps` | 一个团队回合允许的最大环境步数 | JSON / checkpoint 或默认 `100` |
| `--hidden-size` | 两个隐藏层的统一宽度 | JSON 或默认 `[128, 128]` |
| `--seed` | 训练随机种子 | JSON 或默认 `7` |
| `--eval-seeds` | 评估回合的种子列表，每个种子一个回合 | `1001 1002 1003 1004 1005` |
| `--device` | PyTorch 运行设备 | 训练取 JSON（示例为 `cuda`）/ 默认 `cpu`；评估默认 `cpu` |
| `--output` | 输出目录，必须位于 `project` 内 | `project/output/<时间戳>` |
| `--no-plots` | 不生成 PNG，仍保存对应模式的数据 | 自动绘图 |

例如 `--max-steps 100` 与默认 `slot_duration=1.0` 对应 100 秒任务期限。
`--epochs` 控制训练预算；`--max-steps` 改变任务本身，二者含义不同。
示例配置选择 `cuda`；不传 `--config` 时，需显式加 `--device cuda`，否则沿用类默认值 `cpu`。


### CPU、GPU 与整体负载

若希望单次改回 CPU，可在训练命令后加 `--device cpu`。
冻结评估默认仍在 CPU 上加载；需要 GPU 评估时加 `--device cuda`。

当前代码用一个环境串行采样，NumPy 负责信道、SINR 和 PCRB 等物理计算，
`project/__init__.py` 默认将 OMP/MKL 线程数设为 1，因此单环境采样时整体 CPU 占用可能较低。
仅把网络切到 GPU 不会迁移 NumPy 物理计算，也未必让这种逐步小批量推理更快。
迁移机器上的版本、硬件和短测比例见 [VERIFICATION.md](VERIFICATION.md)。

新版训练启动时打印实际 `device`、PyTorch 线程数和环境执行方式，每个 epoch 额外打印并记录：

| 字段 | 含义 |
| --- | --- |
| `rollout_seconds` / 终端 `rollout` | 本批联合采样耗时，包含策略前向和环境物理计算 |
| `update_seconds` / 终端 `update` | Buffer 整理及 PPO 网络更新耗时 |
| `rollout_steps_per_second` / 终端 `steps/s` | 本批联合采样步数除以采样时间，不含更新和写盘 |

已有训练日志不会自动补计时，新启动的训练会包含这些字段。
小矩阵和串行 Python 调用不一定能从增加线程中获益。
后续加速应优先减少物理计算中的重复计算，或实现并行环境采样，并用相同交互预算衡量效果。

GPU 训练命令如下，也可以省略 `--device cuda`，由示例配置提供设备：

~~~bat
python -m project.train --config project/example_config.json --device cuda --seed 7 --output project/output/run_seed7_cuda
~~~

GPU 负责网络推理和 PPO 更新，NumPy 环境仍在 CPU 上执行；GPU 利用率和加速幅度取决于实际工作负载。

## 7. 查看输出与重新绘图

所有相对路径均相对于当前工作目录。每次实验建议使用新的 `--output`，显式复用目录会覆盖同名文件，
也可能保留本次未生成的旧文件。**输出目录必须位于 `project` 内。**

| 文件 | 内容与生成时机 |
| --- | --- |
| `config.json` | 本次实际生效配置，运行开始时写入 |
| `training.csv` | 每批奖励、KL、熵、价值损失，以及采样/更新时间和采样速度；仅训练生成 |
| `episodes.csv` | 已完成团队回合的回报、到达、安全及感知指标；训练中有完整回合才生成 |
| `policy.pt` | 两个 Actor、Critic、网络尺寸和配置；全部训练完成后生成 |
| `training_summary.json` | 总交互步数、完整回合成功率、最后未完成片段长度；训练完成后生成 |
| `evaluation.json` / `evaluation.csv` | 评估种子、汇总与逐回合指标；三种模式都生成 |
| `trajectory.npz` | 第一条评估的两机位置、基站位置、黑飞真值/估计轨迹及指标；三种模式都生成 |
| `training.png` | 每批平均步奖励、价值损失、两个 Actor 的 KL 与熵 |
| `episodes.png` | 完整回合回报、累计团队成功率；有完整训练回合才生成 |
| `trajectory.png` | 基站、两机路径与终点、黑飞真实轨迹及初末位置；有估计记录时另画估计轨迹 |
| `evaluation_metrics.png` | 第一条评估轨迹的终点距离、位置 PCRB、通信 SINR 和奖励分量 |

所有 PNG 在运行结束后生成，采用无窗口后端，不会弹出交互窗口。
在训练、评估或 baseline 命令后加 `--no-plots` 可关闭绘图。

已有 CSV 和 NPZ 时，可用下面的单行命令重新绘图，无需重新训练或评估：

~~~bat
python -m project.replot --output project/output/run_seed7
~~~

将 `--output` 改为已有实验目录即可。程序读取该目录的 CSV、`trajectory.npz` 和 `config.json`，
只更新 PNG。旧轨迹未保存基站位置时，从原实验配置读取；旧轨迹未保存目标估计时，不补造估计曲线。

轨迹图中，UAV 1、UAV 2 对应代码和 CSV 的索引 0、1。图中同时显示基站、两机路径和终点，
以及黑飞 UAV 的真实轨迹和初末位置。新评估还保存目标估计轨迹，单独绘制供比较。
**黑飞真值只用于仿真与离线诊断绘图，不传给两个 Actor 或集中 Critic 的观测。**
当前策略收到的是模拟 BS 消息中的目标估计；估计仍采用加噪代理，尚未实现真实测量与滤波闭环。

## 8. 公式演示与测试

~~~bat
REM 查看物理公式及中间结果
python -m project.formula_demo

REM 运行 project 内的单元与集成测试
python -B -m unittest discover -s project/tests -v
~~~

公式拆分及迁移对应关系见 [PHYSICS_MIGRATION.md](PHYSICS_MIGRATION.md)，
已执行的测试和短训练结果见 [VERIFICATION.md](VERIFICATION.md)。

## 从哪里修改

| 文件 | 修改内容 |
| --- | --- |
| config.py | 场景、奖励、PPO 参数；默认值只有这一处 |
| env.py | 同步运动、安全修正、到达退出、任务终止 |
| observations.py | 31 维局部观测、58 维集中状态及归一化 |
| physics.py | 环境物理适配：确定性波束、有效源筛选、调用包内公式 |
| channels.py | 单位换算、莱斯信道、阵列导向和匹配波束 |
| communication.py | 通信 SINR 及各项干扰 |
| sensing.py | BS 与 UAV 的感知 SINR |
| measurements.py | 状态/测量转换和数值 Jacobian |
| crb.py | SINR 转 CRB、CI 融合和测量噪声矩阵 |
| pcrb.py | 运动预测、信息矩阵和 PCRB 递推 |
| formula_demo.py | 独立查看各公式中间结果 |
| rewards.py | 团队奖励纯函数及各分量 |
| core.py | MLP、两个独立 Actor、集中 Critic |
| ppo.py | Buffer、团队 GAE、PPO 更新 |
| train.py | 训练入口和联合采样 |
| evaluate.py | 冻结策略评估、无学习导航对照 |
| metrics.py | 完整回合统计 |
| artifacts.py | JSON/CSV 与 checkpoint 读写 |
| plot.py / plot_trajectory.py | 训练指标图 / 包含基站和黑飞 UAV 的轨迹图 |
| replot.py | 从已保存日志重新绘图的命令行入口 |
| tests/ | 环境、奖励、物理、PPO 及训练集成验证 |

现在 `project` 是自包含的 Python 包：环境、训练器和物理公式全部位于包内，不再导入 `legacy/test.py` 或其他旧环境模块。
可以把整个 `project` 文件夹复制到其他目录，从它的父目录运行 `python -m project.train`；父目录无需放置任何旧项目文件。
原物理函数的拆分位置及一致性验证见 [PHYSICS_MIGRATION.md](PHYSICS_MIGRATION.md)。
PPO 的网络和 Buffer/loss 结构参考本地 Spinning Up；许可证在 LICENSE-spinningup.txt。
只需要 NumPy、PyTorch、Matplotlib，不依赖 Gym、MPI、SciPy。

## 时间步和退出

1. 在动作前复制 `active_before`。Actor 只对活动机采样 raw action。
2. 两机共同执行一次 env.step；限加速度、限速度，裁剪边界，检查整个时隙线段间的最小距离。
3. 冲突时取消两机本步位移，记 safety_intervention，保留实际执行距离。退出机仍是避碰对象和被动散射体。
4. 本时隙活动机提供新的感知贡献；基站始终提供自身感知。
5. 记录首次到达并计算奖励，随后置为非活动。从下一步开始不采样、不上传、不累计任务路径/时间/能耗。
6. 两机全部到达立即成功；到期限仍有未到达者则失败。最后一步到达优先记成功。

到达动作所在时隙的样本仍有效，且运动能耗在速度清零前计算。
到达后在实际到达位置停驻是任务级终端抽象，不是完整的减速/降落/悬停能耗模型。
reset 时已在终点的 UAV 直接记到达时刻 0，不生成重复到达奖励；两机初始全到达可评估但不用于训练。

接口：

~~~python
obs, state, info = env.reset(seed=7)
obs, state, reward, terminated, truncated, info = env.step(raw_actions)
~~~

raw_actions 为 [2,3]。Buffer 保存高斯原始动作和对应 log-prob，环境内部才做 tanh、范数限幅和安全修正。
info 的 active_before 和 active 分别指本步决策前、执行后；sensing_source_mask 指刚完成的时隙。
真正任务期限属于 terminated；truncated 留给外部截断。批次截止在采样器中处理，不重置仍在运行的环境。

## 观测与物理假设

局部字段顺序在 observations.py 中逐段标注：自身位置/速度、终点相对向量、剩余时间、目标估计、
位置 PCRB 对角摘要、队友相对状态、两机活动/到达标记、自身通信和感知 SINR。
来源假设为机载定位和每步可靠、零时延交付的模拟 BS 消息。
通信 SINR 衡量业务链路并计罚，不控制消息丢包；不能宣称已经模拟了实际上传链路。
局部观测不直接读取 target_state。info 中的目标真值只用于物理诊断/画图。

集中状态包括两机状态和终点、标记、时间、目标估计、完整对称 PCRB 上三角、
通信/感知 SINR 和源有效标记。PCRB 按位置与速度各自尺度归一化；状态名称不意味着严格 Markov 性。

第一版刻意保留以下原有简化：

- 目标估计是 truth + noise，未实现真实测量残差、滤波融合或由感知质量决定估计误差的闭环。
- 目标真实运动为固定加速度；PCRB 仍使用真实前态做匀速预测和统一 Jacobian，存在模型失配。
- 原物理函数将功率 SINR 再平方后转换 CRB；沿用此数值代理，不声称已核实物理论文标定。
- 感知源顺序为 BS、UAV 0、UAV 1。只融合有效源；历史仅通过 J_prev 传播，没有重复融合旧源。
- 等权 CI 随源数重新归一，移除差源可能反而降低融合 CRB，不保证源数增加就改善结果。
- rho_pos = trace(PCRB[:3,:3])，单位按位置状态为 m²；rho_all 混有速度，仅用于旧指标对照。
- 波束指向估计目标与活动 UAV，总功率在有效波束间均分；非活动通信列严格为零。
- CSI 复数扰动只作用有效列，再归一功率，是误差代理，不是严格的 RF 硬件误差实现。

因此不能用本版 PCRB 曲线证明真实跟踪误差改善或几何互补。实际测量、滤波闭环、机动模型、
时延丢包、最低跟踪服务要求都留作后续增强。

## 奖励与 PPO 口径

奖励组成：平均进展 - 平均剩余距离代价 + 首次到达 + 团队完成 - 有界感知代价
- 活动机时间/能耗/通信代价 - 安全修正/越界代价 - 超时未到达惩罚。
所有个体聚合固定除以 2，退出后不改分母。

- progress_i = (d_before - d_after) / (v_max * dt)
- distance_i = d_after / (d_after + distance_ref)
- sensing_cost = rho_pos / (rho_pos + rho_ref)
- communication_cost_i = max(0, 1 - SINR_i / gamma_min)
- energy_proxy_i = (speed/v_max)^2 + 0.5*(acceleration/a_max)^2

默认每架首次到达 +40，全部到达额外 +80，期限未完成每架 -50。
rho_ref=0.01 是旧物理探针给出的候选固定尺度，不是跟踪门限；不存在感知失败终止。
距离差不是严格保持最优策略的势函数塑形，能耗代理也不标成焦耳。

团队 GAE 沿完整团队轨迹传播，个体到达不会 finish_path。
真实终止的末端价值为 0，采样批次或外部截断用截断前下一状态的 V(s) bootstrap。
Critic 用全部团队步的折扣 reward-to-go，Actor 只对各自活动步的团队 GAE 标准化并更新。
两个独立 PPO ratio/clipped loss，没有 HAPPO modifier、第三个 BS Actor 或额外联合 BP。
KL、熵、clipfrac 也仅统计该 Actor 的活动样本；熵按三维联合高斯求和。
默认整批更新 4 轮，独立 KL early stop。原始参考 PPO 的 timeout 分支没有直接照搬。

## 实验指标的解释

到达时间均值只在成功到达的样本中计算；无样本记 `null`，通信满足率只统计活动期间。
`team_success_rate` 是完整回合成功数除以回合数，不是成功时隙比例。
`training.csv` 的 `mean_step_reward` 是当前采样批次的平均步奖励；
完整回合回报在 `episodes.csv` 的 `episode_return` 中，两者不能直接当作同一条学习曲线。

当前评估种子改变噪声，起点和终点仍由配置固定。多场景实验需要多份场景配置，
不同算法应保持相同场景、评估种子和环境交互预算，并使用多个训练种子。
PPO 和奖励权重不保证每次到达，应报告实际成功率；短训练和单条轨迹不构成性能结论。
