# 实际量测、泛化与纯／残差 RL 对比

此版本增加在线模拟退火（SA）、实际仿真量测和 EKF 融合，并用相同训练预算比较纯 RL 和残差 RL。旧实验保留在 `experiments/benchmark_20260927/`，旧 `example_config.json` 仍使用 `proxy` 模式；新实验使用 `closed_loop_config.json` 的 `ekf` 模式。两种模式的“不确定性”含义不同，不能直接比较数值大小。

## 一次运行整组实验

所有命令从仓库根目录执行，适用于 Windows CMD；每条命令均为一行。激活已经安装 GPU PyTorch 的 `uav` 环境后执行：

```bat
python -m project.study --config project/study_config.json --jobs 3 --output project/output/my_study
```

`study_config.json` 明确指定 3 个训练种子（7、17、27）、每模型 100 epoch × 2048 个联合环境步、3 个变体及 30 个评估种子（3001–3030）。总计训练 9 个模型。每个模型使用最后一轮 checkpoint，不依据正式测试成绩挑选模型。`--jobs` 控制独立训练进程数，不是单个模型的并行环境数。新配置用 CPU 做逐步采样推理、CUDA 做 PPO 批量更新（`ppo.rollout_device=cpu`、`ppo.device=cuda`）；环境、信道和滤波也在 CPU。小网络的逐步 GPU 调用与动作回传同步开销很大，设备拆分通常更快。默认字段 `rollout_device=null` 保留旧行为；CLI 可显式加 `--rollout-device cpu`。

可以分阶段执行：

```bat
python -m project.study --config project/study_config.json --stage train --jobs 3 --output project/output/my_study
python -m project.study --stage evaluate --output project/output/my_study
python -m project.study --stage summarize --output project/output/my_study
```

训练和评估阶段各自创建新目录，防止混入旧结果；只想重新生成汇总图时用 `summarize`。训练阶段不会恢复已中断的优化器状态。完整参数快照在 `configs/`，各模型进度在 `train/<variant>_seed<seed>/console.log`。

单独训练一个模型或运行规则对比：

```bat
python -m project.train --config project/closed_loop_config.json --control-mode pure --training-distribution randomized --seed 7 --output project/output/pure_seed7
python -m project.train --config project/closed_loop_config.json --control-mode residual --training-distribution randomized --residual-scale 0.25 --seed 7 --output project/output/residual_seed7
python -m project.benchmark --config project/closed_loop_config.json --suite project/study_scenarios.json --methods goal mpc sa --seed-start 3001 --episodes 30 --output project/output/rules_ekf
```

旧冻结策略的边界、速度和 SA 实验可独立复跑：

```bat
python -m project.benchmark --checkpoint project/output/run_seed7_gpu/policy.pt --suite project/study_scenarios.json --scenarios nominal wide_bounds fast_speed wide_fast crossing shifted_goals --methods goal sa mappo --seed-start 3001 --episodes 50 --output project/output/limits_sa
```

此 checkpoint 是本地旧训练文件，权重不随 Git 上传；没有它时先运行旧训练命令生成模型，或指定自己的 checkpoint，并在结论中记录训练预算。

## 感知闭环如何工作

每步执行以下顺序：

1. 两架 UAV 根据各自观测执行运动动作；黑飞按仿真动力学推进。
2. BS 用上一时刻的后验状态做 CV 预测，得到先验位置、速度和协方差。
3. 先验位置决定感知波束；真实位置仅用于仿真传播与生成带噪声的传感器量测。
4. 每个传感器使用自己的几何量测函数和 Jacobian。BS 单基地距离为两倍目标距离；UAV 双基地距离为 BS→目标→UAV 的传播路径长度。其余量测为接收端方位角、俯仰角和传播路径距离率。
5. BS 自身量测参与更新；UAV 只有在本步活动且通信 SINR 达标时上传。已到达者下一步停止感知、上传和动作，仍保留实体散射与安全影响。这里沿用现有通信 SINR 作为上传成功代理，未另建上行信道、排队或时延模型；BS 后验广播仍按可靠、零时延处理。
6. EKF 依次融合独立条件测量，使用角度残差 wrap 和 Joseph 协方差更新。后验估计及不确定性写入下一步 Actor/Critic 观测。

初始化只采样一次有误差的状态先验。后续不会每步把“真值＋状态噪声”当作估计。关闭量测后，估计只根据自身历史预测，因此可能逐渐偏离目标。这与旧代理模型不同。

量测噪声使用旧 SINR→CRB 公式作为方差代理，并设置物理单位的标准差下限：默认路径长度 1 m、角度 0.5°、路径距离率 0.5 m/s。初始位置／速度标准差为 10 m／1 m/s。旧 CRB 并未经过实际设备校准，新的测量也是仿真测量，并非接入实物雷达。条件独立噪声假设下采用顺序 EKF；若未来建模相关误差，需要扩展联合噪声协方差或使用一致的相关信息融合，不能重复累计相关量测。

保留的 `rho_pos` / `pcrb` 字段用于旧接口兼容；当 `uncertainty_kind=ekf_covariance` 时，它们表示后验协方差及其位置块迹，**不是 PCRB，也不是实际跟踪误差**。`tracking_prefix_rmse` 才是估计与仿真真值的三维位置误差，在相同前 10 步计算，单位 m。真值只用于离线评分和黑色诊断轨迹。

## 纯 RL、残差 RL 和随机场景

- `pure_fixed`：Actor 直接输出动作，固定起终点训练。
- `pure_randomized`：相同纯 RL 网络和优化器，按回合随机化任务。
- `residual_randomized`：同样的随机训练分布，基础目标导航加两个 Actor 学出的加速度修正。

残差执行式为 `a = limit_norm(a_goal + residual_scale * amax * tanh(raw))`。`a_goal` 是原 Goal 控制器经过环境约束后的实际加速度，零残差能恢复基础导航。`residual_scale=0.25` 是每个坐标分量的比例，不是整个残差向量的范数上限；合成动作最终仍受总加速度范数上限限制。策略初始确定性均值为零，训练期间仍有随机探索。残差结构提供先验，但不保证避碰或任务完成。

训练记录原始 Gaussian 样本及其 log-prob；动作适配只发生在执行阶段，所以 PPO 的概率比始终在同一原始动作空间计算。Gaussian entropy 也属于该空间，不代表实际加速度的熵。

随机分布为 25% 原始任务、25% 连续采样的交叉任务、50% 连续起终点任务。连续任务覆盖正向／反向目标；目标间保留间距，期限至少有直线到达下界加 10 步的裕量。期限和 CSI 随任务采样，速度与世界边界保持固定，以便单独研究改变这些限制的效果。几何裕量不构成完整可行性证明。场景随机数与测量随机数分开；正式测试种子不参与训练或参数选择。本轮黑飞初始状态与机动模型保持固定，目标运动的分布外泛化尚未验证；后续可单独随机化目标初始位置、速度、加速度与量测质量，并保留独立机动测试集。

泛化还可进一步采用相对坐标特征、逐步增加难度的课程、观测／动力学随机化、适度记忆网络，以及独立验证集上的模型选择。这些是后续选项；当前实际实现和验证的是目标条件控制与上述多场景训练。扩大训练分布会增加样本需求，不能保证相同步数就比固定场景训练好。

## 比较口径与结果图

`study_scenarios.json` 包含原始任务、不同终点、交叉、反向终点、扩大边界、提高速度、同时改变两者、较差 CSI、较短期限，以及关闭测量／仅 BS 测量消融。扩大边界保持起终点不变；速度从 5 提至 8 m/s，加速度仍为 5 m/s²。它们是冻结模型的限制变更测试，不代表已经在新限制下重新训练。残差基础导航会读取当前已知速度／加速度上限；纯 Actor 的 31 维输入未显式包含这些上限，因此此项比较的是整个控制系统的适应性，不能把全部差异归于神经网络学到的泛化。若以后随机化这些限制，应同时考虑把限制参数作为条件输入。改变速度上限同时改变原奖励的进展与能耗归一化，**跨速度场景的回报或能耗代理不能直接解释为效率改善**；优先看成功率、秒数、米数和实际跟踪 RMSE。

SA 是有模型的集中短时域导航参考：默认 horizon=4、16 次 Metropolis 搜索，每步固定 5 个参考序列＋16 个提案，共 21 次运动模型序列评估。代价包括目标距离、控制努力、间距和越界；只使用当前可用状态及目标，不查询真实未来状态或未来噪声。它不优化完整 ISAC 物理目标，也不保证全局最优。应结合每步决策耗时理解与执行独立 Actor 的计算差异。

结果文件：

| 文件 | 查看内容 |
| --- | --- |
| `policy_comparison.png` | 各场景的成功率与限制完成时间；失败耗时按该场景期限计 |
| `learning_curves.png` | 跨 3 个训练种子的中位训练曲线及最小—最大范围；不同训练分布的训练回报不可单独比较能力 |
| `trajectory_comparison.png` | 固定第一训练／评估种子的路径；基站、友机、终点、黑飞真值和估计同时显示 |
| `tracking_ablation.png` | 相同 Goal 路径下关闭量测、BS-only、多源融合的误差与协方差变化 |
| `seed_summary.csv` | 每个模型单独的测试结果，判断训练稳定性 |
| `study_summary.csv` / `REPORT.md` | 跨训练种子均值、样本标准差和范围；规则策略的评估置信区间 |
| `eval/` | 各模型和规则基线的逐回合原始数据、配置、代码／checkpoint 哈希及首种子轨迹 |

优先看成功率，然后看成功时间／限制时间、边界与安全干预，最后结合相同前缀的跟踪误差、通信与计算量。不同长度回合的全程平均跟踪误差不可直接当作感知优势。环境拒绝危险动作可导致零碰撞，仍需查看安全干预次数。

统计先平均每个 checkpoint 的测试回合，再跨独立训练种子汇总；不能把 3 个模型的 90 个测试回合当成 90 个独立训练种子。只有 3 个训练种子，方差估计仍有限。先前代理模型结果与新 EKF 结果分开归档。

## 修改位置

| 要修改的内容 | 文件 |
| --- | --- |
| 新实验环境、奖励、PPO默认值 | `closed_loop_config.json` |
| 训练种子、预算、变体、测试种子 | `study_config.json` |
| 测试场景与 SA 参数 | `study_scenarios.json` |
| 测量几何、Jacobian、噪声、EKF | `tracking.py` |
| 先验波束、SINR、通信上传门控 | `physics.py` |
| 原始动作→基础导航＋残差 | `control.py` |
| 训练场景分布和范围 | `training_scenarios.py` |
| PPO采样、raw action／log-prob | `rollout.py`；优化见 `ppo.py` |
| 模拟退火与预测运动模型 | `annealing.py`、`planning.py` |
| 多模型运行、统计与绘图 | `study.py`、`study_evaluate.py`、`study_summary.py`、`study_plots.py` |

方法背景：[模拟退火原始论文](https://www.science.org/doi/10.1126/science.220.4598.671)、[EKF说明](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)、[Residual Reinforcement Learning](https://arxiv.org/abs/1812.03201)、[Dynamics Randomization](https://arxiv.org/abs/1710.06537)。这些方法文献不构成本项目性能优越的证据，实际结论以归档实验为准。

本次主比较使用 `pure_fixed_cpu`、`pure_randomized`、`residual_randomized`，各 3 个训练种子，全部 CPU 采样、CUDA 更新、CPU 评估。原先已经启动的三个 `pure_fixed` GPU 采样模型完整保留为辅助实验，不与主比较的固定场景结果合并。安排与配置快照分别见 `primary_comparison_plan.json`、`execution_note.json` 和 `configs/`。默认新配置统一使用 CPU 采样。
