# 物理公式迁移

> 此文档保留原代理模式的说明与历史结果。新增仿真量测、EKF、SA、纯／残差 RL 和随机训练的接口见 [闭环实验指南](CLOSED_LOOP_STUDY.md)。


本文所有命令都从仓库根目录执行，并按 Windows CMD 单行书写。

project 现在是自包含的 Python 包。运行新环境不需要 `legacy/test.py` 或旧版环境文件。
原脚本保留在 [`../../legacy/test.py`](../../legacy/test.py) 作历史对照，公式函数已经实质迁入下面六个模块，没有动态加载旧文件或 sys.path 补丁。

## 函数对应关系

| 新文件 | 从 `legacy/test.py` 迁入的函数 |
| --- | --- |
| channels.py | norm、db_to_linear、dbm_to_watt、abs2、abs2_scalar、normalize、rician_factor、beta_c_bs_uav、beta_s_bs_target、beta_s_two_hop、direction_cosines、steering_vector_upa、make_matched_beams |
| communication.py | uav_comm_sinr_eq10_parts |
| sensing.py | bs_sensing_sinr_eq6_parts、uav_sensing_sinr_eq12_parts |
| measurements.py | make_measurement_vector、make_cartesian_state、cartesian_state_to_measurement、measurement_function_h、numerical_jacobian |
| crb.py | crb_from_sensing_sinr、ci_fusion_crb、measurement_noise_cov_from_fused_crb |
| pcrb.py | state_transition_matrix、process_noise_covariance、predict_cartesian_state、symmetrize、safe_inverse、prior_fim_eq25、data_fim_eq26、pcrb_eq24_to_eq28_parts |

共 32 个物理/数值函数，保留函数名、参数默认值和返回字段。
DIST_EPS、FLOAT_TINY 保留在 channels.py，其他模块按需显式导入。

原 main() 的逐公式演示整理为 formula_demo.py，保留 print_parts() 和 main() 入口，
增加 run_demo(seed=42) 返回中间结果供检查。配置从 EnvConfig 读取，不在公式模块顶层放置演示场景。
打印布局有所简化；实际公式没有变化。

## 新数据流

env.py → physics.py → 包内 channels / communication / sensing / crb / pcrb。
pcrb.py → measurements.py；measurements.py 只从 channels.py 读取距离容差。
所有内部依赖使用显式相对导入，没有重新导入 `legacy/test.py` 的兼容转发层。

physics.py 仍负责活动源和时序，不将这些环境规则混入单个 SINR/CRB 函数。
两个 Actor、集中 Critic、奖励、到达退出和 GAE 逻辑沿用上一版。

## 保持不变的口径

- 感知 SINR 再平方后转换 CRB；CI 对当前有效源重新等权。
- 通信 direct sensing beam 干扰开关的默认值为 False，True 分支保留。
- 阵列方向、链路项和随机数调用顺序保持不变。
- 测量状态顺序 [x,y,z,vx,vy,vz]；测量顺序 [d,theta,phi,radial_velocity]。
- 两个历史球坐标转换函数分别保留不截断/截断 arccos 输入的行为。
- Jacobian 的第 2 个输出分量按方位角处理跨 ±pi 差分。
- PCRB 保留匀速 prior；当前环境仍以固定加速度推进真实目标。
- safe_inverse 的奇异矩阵伪逆分支保留，不新增固定 jitter。
- rho 仍是完整六维 trace，rho_pos 仍在适配层计算位置子块的迹。

这次工作是模块迁移，不是新增测量滤波或修订论文公式。上一版记录的物理简化仍适用。

## 使用与验证

从仓库根目录运行：

~~~bat
python -m project.formula_demo
python -m unittest discover -s project/tests -v
python -m project.train --config project/example_config.json
~~~

- 临时只读对照脚本将新函数与 `legacy/test.py` 按固定种子比较；这些旧文件引用不写入永久测试。
- 永久公式测试使用可手算值、协方差和功率约束，不依赖旧目录。
- test_standalone.py 只复制 project 源码到空目录，阻断所有旧模块导入，执行公式演示、
  8 步短训练、模型保存和评估。临时文件在测试结束时清理。
- 迁移后的 PPO 与导航对照评估，和迁移前已保存的各回合统计、奖励及全部轨迹字段逐项完全一致。

具体最新测试数量和结果见 [VERIFICATION.md](VERIFICATION.md)。
