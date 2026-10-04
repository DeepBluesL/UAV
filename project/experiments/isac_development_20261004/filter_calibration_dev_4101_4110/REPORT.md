# Filter calibration development diagnostic / 滤波校准开发诊断

## 中文摘要

本诊断固定使用 GoalController、nominal/crossing 场景和开发种子 4101–4110，只重放前 10 步。滤波数值、随机数顺序、CRB 代理及噪声下限均未改变。位置 NEES 使用三维 95% 阈值 7.8147279；每源 NIS 使用四维 95% 阈值 9.487729。覆盖率是描述性结果；同一回合内的相关时步不能视为独立卡方检验。

最明显的问题是初始化后的短暂不一致，而不是持续失准。nominal 第 1–3 步位置 NEES 均值为 26.00，中位数 3.51，最大值 428.54，经验 95% 覆盖率为 76.7%；第 4–10 步均值降至 2.40，中位数 1.87，最大值 9.10，覆盖率为 98.6%。crossing 也从第 1–3 步的均值 10.75、覆盖率 86.7%，改善到第 4–10 步的均值 2.50、覆盖率 97.1%。保留全部种子后的 first-10 覆盖率为 nominal 92.0%、crossing 94.0%。

每源 NIS 同样显示早期长尾。nominal 第 1–3 步 BS/UAV0/UAV1 的均值分别为 5.21/11.46/18.67，覆盖率为 90.0%/86.7%/80.0%；第 4–10 步均值降至 3.31/3.96/3.62，覆盖率升至 97.1%/92.9%/95.7%。crossing 第 1–3 步均值为 4.46/5.70/7.13，覆盖率 93.3%/86.7%/80.0%；第 4–10 步为 3.17/3.41/4.18，覆盖率 98.6%/94.0%/93.9%。crossing 后段 UAV 统计仅包含实际交付的 50 和 49 个量测，未把缺失上传当成零 NIS。

这些数字支持先检查初始状态分布、首次几次非线性更新和各源顺序融合，而不是立即归咎于路径规划器。第 4–10 步的均值和覆盖率接近合理尺度，但 10 个开发种子和相关时步不足以宣称滤波器已严格校准，更不能代表硬件校准。后续 sensing-aware planner 应保留完整 first-10 指标，同时单列第 1–3 步与第 4–10 步，避免初始化离群掩盖可能由动作控制的后段几何效应。

## English summary

The inconsistency is concentrated in the first three updates. Nominal position NEES changes from mean 26.00 and 76.7% empirical coverage at steps 1–3 to mean 2.40 and 98.6% coverage at steps 4–10. Crossing changes from 10.75/86.7% to 2.50/97.1%. Source NIS shows the same early heavy tails, especially nominal UAV updates, while later means are near the four-dimensional reference scale.

This supports auditing initialization and the first nonlinear sequential updates before blaming the planner. All seeds and early failures remain in the data. The sample is small and temporally correlated, so these figures are descriptive consistency checks rather than independent chi-square tests or evidence of hardware calibration. No filter setting or measurement floor was tuned.

Raw per-step records are in `steps.csv`; window/source summaries are in `summary.csv`; hashes, thresholds, seeds, and provenance are in `manifest.json`.
