# Closed-Loop Study: Measured Results

These results come from the archived run at [experiments/closed_loop_20260927](../experiments/closed_loop_20260927/). The primary study used three training seeds per RL variant and 204,800 joint environment steps per checkpoint. Metrics first average the 30 evaluation episodes for each checkpoint and then summarize across three training seeds; three seeds provide only a limited estimate of training variability.

The primary comparison contains 3,960 evaluation episodes: 2,970 from nine RL checkpoints and 990 from three rule controllers. Another 990 episodes from the earlier GPU-rollout `pure_fixed` checkpoints are auxiliary and excluded from the primary aggregation. The previous proxy-sensing benchmark contains 900 separate episodes and is not mixed with the EKF study.

## Main findings

- `residual_randomized` succeeded in all 990 evaluated episodes across 11 scenarios and three training seeds, with no recorded safety interventions. On crossing paths it completed in 22.0 s, compared with 25.0 s for MPC and 26.17 s for SA; all three had 100% success and zero safety interventions in that scenario.
- Residual RL was not universally better than its Goal prior. In the nominal scenario both completed in 27.0 s with 100% success, while mean episode return was 264.697 for residual RL and 266.818 for Goal. The learned residual therefore retained broad task completion in this study without establishing a universal reward advantage.
- `pure_fixed_cpu` reached 100% nominal success but 0% on crossing and shifted-goal tests. Its mean nominal restricted time was 54.89 s and varied substantially across the three seeds.
- `pure_randomized` reached only 33.3% mean success on nominal and shifted goals and 0% on crossing. With the same 204,800-step budget, widening the training distribution did not produce a reliable pure policy. This is evidence about this implementation and budget, not evidence that task randomization is generally harmful.

## Limit changes

The fixed-task pure policy did not use a higher speed limit effectively: mean restricted time changed from 54.89 s at the nominal 5 m/s limit to 61.04 s at 8 m/s. Its success rate also fell from 100% under nominal bounds to 66.7% with wider bounds. In contrast, residual RL and Goal changed from 27.0 s at 5 m/s to 17.0 s at 8 m/s, both with 100% success.

This comparison is between complete controllers, not only neural networks. The residual controller's Goal component reads the current known motion limits, while the pure Actor's 31-dimensional observation does not include the speed, acceleration, or world-bound parameters. Changing the speed limit also changes progress and energy normalization. Cross-speed returns and energy proxies should therefore not be described as direct efficiency gains.

## Tracking ablation

Using the Goal controller and paired evaluation seeds, ten-step position RMSE was:

| Measurement setting | Mean RMSE | Median RMSE |
| --- | ---: | ---: |
| Prediction only | 19.465 m | 20.080 m |
| BS measurement only | 4.726 m | 1.269 m |
| BS and UAV fusion | 0.706 m | 0.331 m |

All 30 paired seeds improved when UAV measurements were added to the BS measurement. The mean full-fusion minus BS-only difference was −4.021 m, with a paired percentile-bootstrap 95% interval of [−6.805, −1.835] m (200,000 resamples; RNG seed 20260927).

These numbers validate the simulated measurement and EKF loop under its stated assumptions: conditionally independent measurement noise, SINR-derived variance proxies, physical-unit noise floors, reliable zero-delay posterior broadcast, and communication-SINR-gated UAV uploads. They do not constitute validation against physical radar hardware.

The navigation study does not demonstrate an active-sensing decision advantage. `residual_randomized` still achieved 100% navigation success in the prediction-only ablation, and the reward/control comparison did not isolate a learned action chosen specifically to improve future measurements. The tracking result supports multisensor estimation quality under a fixed controller, not a claim that RL learned an optimal sensing trajectory.

## Interpretation limits

- Target initial state and motion were fixed during training and evaluation. No out-of-distribution claim is supported for different target maneuvers.
- Residual RL includes a strong Goal-controller prior. Its result should be compared both with Goal and with pure RL, and should not be attributed solely to learned behavior.
- The wider-bound and faster-speed cases change known controller limits. Because the residual base reads those limits and the pure Actor does not, they measure whole-system adaptation.
- Zero collision counts can coexist with rejected unsafe commands; safety interventions must be reported separately. The residual policy recorded none in this study.
- SA is a centralized model-based short-horizon reference with a different computation pattern from two independent Actors. It is not a global optimizer or a full ISAC oracle.
- Three training seeds are enough to reveal large instability, but not to estimate algorithm-level performance precisely.

The experiment protocol and file definitions are described in [CLOSED_LOOP_STUDY.en.md](CLOSED_LOOP_STUDY.en.md). Consult the archived `study_summary.csv`, `seed_summary.csv`, `tracking_ablation_statistics.json`, configurations, source hashes, and per-episode files for exact values and provenance.
