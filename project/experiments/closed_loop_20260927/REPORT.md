# MAPPO study 汇总

训练变体：pure_fixed_cpu, pure_randomized, residual_randomized；独立训练种子：7, 17, 27。
每个 checkpoint 先平均其 evaluation episodes，再跨训练种子统计均值、样本标准差和范围；规则方法的区间只表示评估回合不确定性。
这些结果不构成全局最优或训练收敛证明。

## 主要图表

- [策略比较](policy_comparison.png)
- [训练曲线](learning_curves.png)
- [轨迹比较](trajectory_comparison.png)

## 主要任务指标

学习方法写作“跨训练种子均值 ± 训练种子样本标准差”；规则方法括号为 evaluation episodes 的 95% 区间，两类区间含义不同。

Goal 是独立目标导航，MPC／SA 使用集中式运动模型规划，RL 在执行时使用各自局部 Actor；它们的先验、协调方式和决策耗时不同，应结合安全干预与 `policy_ms_per_step` 解读。

回报是进展、距离、时间、能耗、通信、感知和终止项的加权和，只宜在同一场景与同一奖励配置内比较；尤其不能用跨速度场景的回报差异直接声称效率提升。

| 场景 | 方法 | 成功率 | 限制时间 (s) | 回报 |
|---|---|---:|---:|---:|
| bs_only | pure_fixed_cpu | 0.6444 ± 0.558 | 71.19 ± 36.8 | 147.8 ± 118 |
| bs_only | pure_randomized | 0.7667 ± 0.376 | 64.29 ± 35.1 | 171 ± 109 |
| bs_only | residual_randomized | 1 ± 0 | 27.01 ± 0.0192 | 263.6 ± 1.04 |
| crossing | pure_fixed_cpu | 0 ± 0 | 100 ± 0 | -134.2 ± 3.92 |
| crossing | pure_randomized | 0 ± 0 | 100 ± 0 | -49.87 ± 10.9 |
| crossing | residual_randomized | 1 ± 0 | 22 ± 0 | 256.5 ± 0.459 |
| fast_speed | pure_fixed_cpu | 1 ± 0 | 61.04 ± 38.6 | 196 ± 35 |
| fast_speed | pure_randomized | 0.3333 ± 0.577 | 92.73 ± 12.6 | 17.09 ± 140 |
| fast_speed | residual_randomized | 1 ± 0 | 17 ± 0 | 225.5 ± 0.321 |
| noisy_csi | pure_fixed_cpu | 1 ± 0 | 65.24 ± 34.5 | 239.5 ± 30.3 |
| noisy_csi | pure_randomized | 0.3556 ± 0.559 | 93.58 ± 11.1 | 95.2 ± 95.8 |
| noisy_csi | residual_randomized | 1 ± 0 | 27 ± 0 | 264.7 ± 1.02 |
| nominal | pure_fixed_cpu | 1 ± 0 | 54.89 ± 24.8 | 241.1 ± 31.2 |
| nominal | pure_randomized | 0.3333 ± 0.577 | 93.56 ± 11.2 | 89.81 ± 102 |
| nominal | residual_randomized | 1 ± 0 | 27 ± 0 | 264.7 ± 1.02 |
| prediction_only | pure_fixed_cpu | 0 ± 0 | 100 ± 0 | -269.5 ± 17 |
| prediction_only | pure_randomized | 0 ± 0 | 100 ± 0 | -318.6 ± 38.8 |
| prediction_only | residual_randomized | 1 ± 0 | 27.33 ± 0.577 | 260.3 ± 1.36 |
| reverse_goals | pure_fixed_cpu | 0 ± 0 | 100 ± 0 | -311.6 ± 3.54 |
| reverse_goals | pure_randomized | 0.1222 ± 0.212 | 97.79 ± 3.83 | -61.01 ± 67.8 |
| reverse_goals | residual_randomized | 1 ± 0 | 6.333 ± 0.577 | 185.9 ± 0.751 |
| shifted_goals | pure_fixed_cpu | 0 ± 0 | 100 ± 0 | -15.87 ± 111 |
| shifted_goals | pure_randomized | 0.3333 ± 0.577 | 76.37 ± 40.9 | 62.63 ± 156 |
| shifted_goals | residual_randomized | 1 ± 0 | 20 ± 0 | 240.8 ± 0.553 |
| tight_deadline | pure_fixed_cpu | 0.1 ± 0.173 | 30 ± 0 | 74.18 ± 73.6 |
| tight_deadline | pure_randomized | 0 ± 0 | 30 ± 0 | 20.85 ± 57.8 |
| tight_deadline | residual_randomized | 1 ± 0 | 27 ± 0 | 264.8 ± 0.99 |
| wide_bounds | pure_fixed_cpu | 0.6667 ± 0.577 | 63.34 ± 36.5 | 133.1 ± 219 |
| wide_bounds | pure_randomized | 0.05556 ± 0.0962 | 99.79 ± 0.366 | 74.39 ± 13.1 |
| wide_bounds | residual_randomized | 1 ± 0 | 27 ± 0 | 264.7 ± 1.02 |
| wide_fast | pure_fixed_cpu | 0.6556 ± 0.568 | 70.37 ± 45.5 | 87.37 ± 218 |
| wide_fast | pure_randomized | 0 ± 0 | 100 ± 0 | -40.81 ± 55.4 |
| wide_fast | residual_randomized | 1 ± 0 | 17 ± 0 | 225.5 ± 0.321 |
| bs_only | goal | 1 (0.886, 1) | 27 (27, 27) | 265.6 (265, 266) |
| bs_only | mpc | 1 (0.886, 1) | 27 (27, 27) | 265.6 (265, 266) |
| bs_only | sa | 1 (0.886, 1) | 27 (27, 27) | 265.6 (265, 266) |
| crossing | goal | 0 (0, 0.114) | 100 (100, 100) | -1928 (-1.93e+03, -1.93e+03) |
| crossing | mpc | 1 (0.886, 1) | 25 (25, 25) | 254 (254, 254) |
| crossing | sa | 1 (0.886, 1) | 26.17 (25.5, 27) | 256.4 (256, 257) |
| fast_speed | goal | 1 (0.886, 1) | 17 (17, 17) | 226.1 (226, 226) |
| fast_speed | mpc | 1 (0.886, 1) | 17 (17, 17) | 226.2 (226, 226) |
| fast_speed | sa | 1 (0.886, 1) | 17 (17, 17) | 226 (226, 226) |
| noisy_csi | goal | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| noisy_csi | mpc | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| noisy_csi | sa | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| nominal | goal | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| nominal | mpc | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| nominal | sa | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| prediction_only | goal | 1 (0.886, 1) | 27 (27, 27) | 262.3 (262, 262) |
| prediction_only | mpc | 1 (0.886, 1) | 27 (27, 27) | 262.3 (262, 262) |
| prediction_only | sa | 1 (0.886, 1) | 27 (27, 27) | 262.3 (262, 262) |
| reverse_goals | goal | 1 (0.886, 1) | 6 (6, 6) | 187.9 (188, 188) |
| reverse_goals | mpc | 1 (0.886, 1) | 6 (6, 6) | 187.9 (188, 188) |
| reverse_goals | sa | 1 (0.886, 1) | 6.033 (6, 6.1) | 187.8 (188, 188) |
| shifted_goals | goal | 1 (0.886, 1) | 20 (20, 20) | 242.3 (242, 242) |
| shifted_goals | mpc | 1 (0.886, 1) | 20 (20, 20) | 242.3 (242, 242) |
| shifted_goals | sa | 1 (0.886, 1) | 20 (20, 20) | 242.3 (242, 242) |
| tight_deadline | goal | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| tight_deadline | mpc | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| tight_deadline | sa | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| wide_bounds | goal | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| wide_bounds | mpc | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| wide_bounds | sa | 1 (0.886, 1) | 27 (27, 27) | 266.8 (267, 267) |
| wide_fast | goal | 1 (0.886, 1) | 17 (17, 17) | 226.1 (226, 226) |
| wide_fast | mpc | 1 (0.886, 1) | 17 (17, 17) | 226.2 (226, 226) |
| wide_fast | sa | 1 (0.886, 1) | 17 (17, 17) | 226 (226, 226) |

## 前缀跟踪误差与不确定性

`tracking_prefix_rmse` 是 EKF 估计相对诊断真值的 RMSE；`rho_prefix_mean` 是 EKF 后验位置协方差迹（m²），两者不可互换。

| 场景 | 方法 | Prefix RMSE | EKF position covariance trace (m²) |
|---|---:|---:|---:|
| nominal | pure_fixed_cpu | 0.7366 | 0.18037 |
| nominal | pure_randomized | 0.74005 | 0.1585 |
| nominal | residual_randomized | 0.70661 | 0.15741 |
| shifted_goals | pure_fixed_cpu | 0.73948 | 0.18173 |
| shifted_goals | pure_randomized | 0.75163 | 0.16168 |
| shifted_goals | residual_randomized | 0.72118 | 0.16428 |
| nominal | goal | 0.70565 | 0.15745 |
| nominal | mpc | 0.70565 | 0.15745 |
| nominal | sa | 0.70583 | 0.15796 |
| shifted_goals | goal | 0.72095 | 0.165 |
| shifted_goals | mpc | 0.72095 | 0.165 |
| shifted_goals | sa | 0.72095 | 0.165 |
