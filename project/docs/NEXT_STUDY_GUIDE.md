# 感知规划、奖励消融与学习预算实验

[English](NEXT_STUDY_GUIDE.en.md) · [中文首页](../../README.md)

这一轮按“路径诊断 → 感知规划 → 观测与奖励 → 训练与泛化 → 统计归档”执行。历史实验目录保持原样。新实验的数值结论以对应输出和最终归档报告为准；单元测试通过不表示策略已经学好。

## 运行

以下都是从仓库根目录执行的 Windows CMD 单行命令。先激活已安装 CUDA PyTorch 的 `uav` 环境。输出目录必须是新的目录。

```bat
conda activate uav
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -B -m project.sensing_diagnostics --output project/output/path_diagnostic --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
python -B -m project.calibration_diagnostics --output project/output/filter_diagnostic --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
python -B -m project.planner_study --grid project/planner_study_dev_grid.json --output project/output/planner_development --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
python -B -m project.next_study --config project/next_study_protocol.json --output project/output/my_next_study --stage all --jobs 3
```

完整研究配置包括 24 个模型、3 个独立训练种子，以及预先指定的预算快照；它是批量研究命令，耗时明显长于一次训练。训练使用 CPU 逐步采样、CUDA 批量 PPO 更新，NumPy 环境仍在 CPU 上。`--jobs 3` 并行运行三个独立训练进程，不会把单个环境变成 GPU 环境。

也可分阶段执行：

```bat
python -B -m project.next_study --config project/next_study_protocol.json --output project/output/my_next_study --stage train --jobs 3
python -B -m project.next_study --output project/output/my_next_study --stage evaluate --jobs 3
python -B -m project.next_study --output project/output/my_next_study --stage summarize
```

上面 `train` 与 `all` 二选一，不能对同一个已有目录重复启动训练。评估使用输出目录内冻结的场景配置，完整匹配的已完成评估可跳过。训练快照只用于推理，不包含续训所需的优化器状态；预算曲线来自一次连续训练，不是反复重启训练。

## 修改入口

| 内容 | 文件 | 修改要点 |
|---|---|---|
| 实验组、种子、预算、奖励系数 | `next_study_protocol.json` | 开发/验证/测试种子分开；正式运行前冻结 |
| 测试场景、时间窗口、规划参数 | `next_study_scenarios.json` | 每次只改变声明的条件；反向任务窗口为5步，其余10步 |
| 仿真量测与 EKF | `tracking.py`, `physics.py` | 真值只用于生成测量及离线评分 |
| 噪声下限与量测尺度 | `config.py`, `tracking.py` | 当前是未校准 CRB 方差代理，不为追求收益任意降低下限 |
| 感知 SA/MPC | `sensing_planning.py`, `nominal_links.py` | 从估计均值和协方差预测；无未来噪声和目标真值 |
| 观测字段与版本 | `observations.py`, `observation_specs.py` | v1为31/58维，v2为67/72维；旧权重继续按v1加载 |
| 奖励 | `rewards.py`, `RewardConfig` | legacy或log1p；v2输入尺度与奖励的rho_ref分离 |
| 训练任务与课程 | `training_scenarios.py`, `domain_randomization.py` | 参数范围集中声明；课程按累计交互步数切换 |
| 模仿初始化 | `behavior_cloning.py` | 只拟合Actor均值的实际归一化加速度；Critic和log_std保持原初始化 |
| PPO与预算快照 | `train.py`, `rollout.py`, `training_helpers.py` | 复用既有PPO/GAE；示范步数计入总预算 |
| 隐藏目标机动 | `target_motion.py` | 默认保持原匀加速行为；测试可切换加速度 |
| 指标、汇总、绘图 | `window_metrics.py`, `next_study_summary.py`, `next_study_plots.py` | 先按评估回合平均，再跨独立训练种子统计 |

## 训练口径

主比较在100×2048=204800个联合环境交互后进行。纯RL、课程RL、Goal模仿初始化RL、残差导航RL继续训练到300和500 epoch并保存快照。观测版本与奖励消融组训练到100 epoch。Goal示范的8192步占总预算的前4个逻辑epoch，PPO从第5个逻辑epoch开始；离线模仿更新不计为新增环境交互，但单独记录计算时间。

课程前50000步使用基础任务，50000–149999步使用随机几何任务，之后使用完整领域随机化。已知运动限制、边界、基站和通信阈值进入v2观测；目标真实初态、加速度与未来转向不进入Actor/Critic。随机化的具体区间见 `domain_randomization.DOMAIN_RANGES`。

Goal示范只教授基础导航，可能包含交叉任务失败；后续PPO仍需学习协同。带示范的纯Actor不是“从零训练的纯RL”，图表应明确标记。

## 规划与奖励的区别

导航SA使用归一化距离，MPC使用米制距离；两者原生代价尺度不同，感知权重不能直接横向比较。新增代价是预测时域内平均的 `log1p(trace(P_pos)/uncertainty_ref)` 与通信短缺平滑代理；两机均到达后停止累计。RL奖励则在每个真实环境步计算。规划权重不机械地等同于RL奖励系数。

预测信道使用平均功率之比和估计目标方向，没有随机CSI扰动。它不是随机SINR的精确期望；模型误差与短时域搜索限制均需结合实际RMSE解释。MPC/SA为集中规划，RL执行两个局部Actor，模型先验与计算量不同。

## 读结果

首先看成功率、安全干预、碰撞和限制完成时间。失败时间计为期限，不能当成有效慢到达。再看相同窗口的实际跟踪RMSE、EKF位置协方差迹、通信满足率。第1–3步与第4至窗口末步仅作机制分解，不替代保留初始化误差的完整窗口。

NEES/NIS和95%覆盖率是诊断指标；回合内时步相关，不能当作独立卡方检验。协方差变小不自动表示真实误差变小。不同奖励配置的原始回报不能直接比较，零碰撞也不自动表示没有依赖安全干预。

轨迹图固定展示首个预定训练/测试种子，显示BS、两架友机、各自目标、黑飞真值与估计；黑飞真值只出现在仿真与诊断中。统计图应与独立训练种子波动一起阅读，不按最终测试结果挑选checkpoint。

```bat
python -B -m unittest discover -s project/tests -v
```
