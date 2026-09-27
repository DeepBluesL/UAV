# Baseline 与典型导航方法对比

命令均从仓库根目录执行，使用 Windows CMD 单行格式。主入口是 `python -m project.benchmark`；原有 `python -m project.train --mode baseline` 仍只运行 GoalController。

## 运行

已有模型时，以 checkpoint 中保存的环境和奖励作为共同基准，对比五种无训练方法与冻结 MAPPO：

```bat
python -m project.benchmark --checkpoint project/output/run_seed7_gpu/policy.pt --suite project/benchmark_config.json --seed-start 2001 --episodes 100 --device cpu --output project/output/comparison_seed7
```

刚克隆仓库没有模型时，可以先跑五种规则方法；输出目录必须是 `project` 内尚不存在的目录：

```bat
python -m project.benchmark --config project/example_config.json --suite project/benchmark_config.json --seed-start 2001 --episodes 100 --output project/output/comparison_rules
```

只验证几个方法和场景的运行流程：

```bat
python -m project.benchmark --config project/example_config.json --methods goal pd apf mpc --scenarios nominal crossing --eval-seeds 1901 1902 --output project/output/comparison_quick
```

`--episodes` 是**每个场景、每种方法**的回合数，默认 100；默认种子连续取 2001–2100。显式 `--eval-seeds` 会覆盖这组种子。全套五场景、六方法共 3000 回合。`--device` 只控制冻结神经网络的推理设备，本轮统一在 CPU 上评估，原有 GPU 训练配置不受影响。

`--checkpoint` 和 `--config` 互斥：禁止在不知情时更换模型的基准场景。场景变化集中写在 [benchmark_config.json](../benchmark_config.json) 的 `scenarios`；`--scenarios` 可选择子集。`--reference` 指定配对比较基准，默认 `goal`，它必须包含在 `--methods` 中。

## 方法与可修改位置

| 方法 | 实现及默认参数 | 信息与限制 |
| --- | --- | --- |
| `random` | [baselines.py](../baselines.py)：每维 raw action 均匀取 [-1,1]，每回合独立重置随机流 | 最低限度检查；不应只击败随机策略就声称有优势 |
| `goal` | [evaluate.py](../evaluate.py)：朝终点的目标速度，再计算一步加速度 | 快速直线导航，不主动规避队友；由环境安全层拒绝危险动作 |
| `pd` | [baselines.py](../baselines.py)：a = kp·目标位移 + kd·(目标速度−当前速度)，kp=1，kd=1.5，减速半径15 m | 目标速度在近终点处降低；加速度及速度仍遵循环境限幅 |
| `apf` | [baselines.py](../baselines.py)：PD 吸引项、队友间距排斥、边界排斥 | APF 风格基线；对称交叉路径可能出现局部极小或死锁，不含专门解困规则 |
| `mpc` | [mpc.py](../mpc.py)：3步预测，每架活动机7种候选，最多49种联合组合 | 集中有限候选导航 MPC；候选加速度在预测窗内保持不变，只执行第一步；不是连续优化器或完整 ISAC 最优控制 |
| `mappo` | [core.py](../core.py)：checkpoint 的两个独立 Actor，确定性执行 | 各 Actor 使用自己的观测；评估不调用 Critic，不训练、不微调 |

APF 默认队友增益18、影响参数12 m、边界增益10、边界范围10 m；MPC 默认控制代价0.03、速度代价0.02、间距代价8、末端权重2。完整参数会写入每次实验的 `manifest.json`；在套件 JSON 的 `methods.pd/apf/mpc` 中覆盖构造参数即可，不需要改运行器。

MPC 预测包含加速度、速度和边界约束、整个时隙的两机最小距离、到达后冻结；若全部候选不安全则请求制动，最终仍依赖环境安全层。它联合使用两机的观测和目标，是集中规划参考，信息共享方式与分散 Actor 不同。所有方法只使用公开观测和已知运动学/边界参数，不读取黑飞真值或未来噪声。

## 场景与实验控制

默认套件在查看正式结果前固定如下。调试只用1901、1902；正式种子为2001–2100，没有挑选成功种子。

| 场景 | 相对基准配置的变化 | 用途 |
| --- | --- | --- |
| `nominal` | 无 | 与训练条件一致的主要比较 |
| `noisy_csi` | imperfect_csi_beta=0.3；基准为0.1 | 估计/波束扰动增强 |
| `tight_deadline` | max_steps=30 | 原几何下更紧的任务期限；默认1秒步长时为30秒 |
| `crossing` | 起点(30,−25,70)/(30,25,70)，终点(130,25,70)/(130,−25,70) | 同高度相交航路和避碰协调 |
| `shifted_goals` | 终点改为(120,55,90)/(100,−40,45) | 未见过的目标位置 |

后四项属于**指定的分布移位压力测试**。训练时起终点固定，不能把这些结果等同于经过多场景训练后的表现。相同场景内所有方法使用相同奖励、运动约束、时限和评估种子；固定几何上的100个噪声种子也不等于100种不同几何场景。

同一 seed 只保证环境从同一随机流开始：方法导致的活动源数量不同，会改变后续抽样次数。因此这里称“配对评估种子”，不称严格共同随机数。

## 指标与区间

- 首先看团队成功率，报告95% Wilson区间；即使100/100成功，区间下界仍约96.3%。
- 限制完成时间：成功按实际耗时，失败按该场景的任务期限计。成功回合耗时另报样本数；不能只看成功子集。
- 同时报告安全干预、实际碰撞和越界请求。零碰撞不能说明策略没有危险动作，可能是安全层拦截的结果。
- 两机路程和能耗代理取和；通信满足率按两机活动时隙加权。能耗不是焦耳。
- 感知主比较使用第1至10步的 `rho_prefix_mean`，排除初始先验。不足10步记为缺失并报告样本数，绝不补造后续感知；若出现缺失，应说明可用样本选择的影响。
- 保留原全回合 `rho_pos_mean` 作为补充，因为回合长度不一样，它不能单独证明感知更好。
- 连续均值和同种子配对差给出2000次 bootstrap 的95%区间。默认差为“当前方法−goal”：回报正值较好，耗时/PCRB负值较好。多个比较未经多重检验校正，属于探索性结果。
- 计时只包含预热后的 `policy.act` 墙钟时间，不含 NumPy 物理环境；单次实验的计时不能当作稳定硬件性能排名。

当前 MAPPO 只有训练种子7的已保存模型。本轮区间只反映该固定模型的评估波动；评估种子不能代替独立训练种子，不能由本轮推出 MAPPO 训练稳定性或优于其他学习算法。

## 结果文件与实现

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 基准配置、完整种子列表、场景、控制器参数、模型SHA-256、源码SHA-256与运行环境 |
| `episodes.csv` | 每一回合的全部指标；每完成一组就写盘 |
| `summary.csv/json` | 各场景/方法的均值、标准差、有效样本数及区间 |
| `paired.csv/json` | 相对参考方法的同seed差值和区间 |
| `REPORT.md` | 自动生成的分场景表格、统计口径与结论边界 |
| `comparison_*.png` | 成功率、耗时、回报、前缀PCRB、越界和安全干预 |
| `trajectories/<场景>/<方法>/` | 第一条评估种子的NPZ与轨迹图，包含基站、两机、黑飞真值/估计；不挑选最佳轨迹 |

`benchmark.py` 负责运行；`benchmark_setup.py` 负责加载策略/场景；`benchmark_metrics.py` 负责统计；`benchmark_plots.py` 和 `benchmark_report.py` 负责图表与报告。可用 `--no-plots` 跳过 PNG；不完整运行的 manifest 标记为 running，不应作为完整实验引用。

物理限制仍与主环境一致：目标估计为加噪代理，未形成真实测量滤波闭环；PCRB不是实测跟踪误差，等权CI也不保证增加感知源一定改善结果。

## 方法来源

APF 的吸引/排斥思路参考 [Khatib, 1986](https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf)；滚动有限时域控制思路参考 [Mayne et al., 2000](https://www.sciencedirect.com/science/article/pii/S0005109899002149)。本仓库采用针对现有运动模型的简化实现，不声称复现这些论文全部算法或其稳定性保证。
