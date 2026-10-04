# Sequential ISAC study plan / 顺序研究计划

Status: in progress. Started 2026-10-04 (Asia/Taipei).

The user authorized all five stages below, in order, and publication to GitHub. Previous experiments remain immutable. New Python modules stay under project/ and are kept small and readable. Reuse existing PPO/core conventions. UAV arrival freezes/exits and hidden target truth must remain intact.

## Completion requirements

1. **Controllability diagnostics**: implement and run feasible path comparisons; report noise-floor hit rates by source/component, fixed-window actual RMSE and covariance, communication, navigation cost and failures. Distinguish sensor-noise saturation from geometry sensitivity. Do not invent hardware calibration.
2. **Sensing-aware planning**: implement belief-only SA/MPC cost forecasts with no target truth or actual future noise; compare with navigation references at declared search budgets. Validate gains or explicitly retain a negative result.
3. **Observation/reward changes**: versioned richer observations with constraint/time/belief information; legacy checkpoint compatibility; selectable uncertainty penalty and reward-scale ablation, after diagnosing usable signal. No policy access to truth.
4. **Learning and generalization**: pure RL curriculum and Goal-supervised initialization; controlled budget checkpoints; compare with residual RL using independent training seeds. New holdout tasks/seeds for target motion, limits, and sensor/upload conditions, separated from development/tuning data. Report training failure separately from generalization failure.
5. **Evidence and publication**: matched navigation/sensing training ablations, common-window sensing/communication metrics, navigation-quality tradeoff plots, independent training-seed statistics, readable Chinese/English instructions and conclusions. Archive reproducible data without weights, verify tests and plots, commit and push GitHub and verify remote SHA.

## Invariants and evidence

- No real/forecast target truth enters Actor/Critic/planners. Truth is permitted only in simulator measurements and offline diagnostics.
- Arrival/exit and safety semantics preserved; no rewarded post-arrival loitering.
- Planning predictions use beliefs and independent expected/nominal channels, not a cloned simulator's hidden future.
- Failed runs and methods remain in reports; final checkpoints/budgets are predeclared, no test-set checkpoint selection.
- Results of changed observation/reward/physical settings are separate from historical 2026-09-27 results.
- Every new module has an explicit purpose; tests check meaningful physical/learning/protocol invariants.

## Execution log

- 2026-10-04: verified clean worktree at b77fbe64030e3589ee77ab526d5721b23fd84eab; created codex/isac-next-study. Stage 1 implementation started. Stages 2–4 receive read-only design audits while diagnostics are prepared.

- Stage 1 executed: `project/output/sensing_diagnostics_step1_dev_4101_4110/`, 140 episodes. Noise floors bind frequently; simple feasible deviations yield modest/variable prefix estimation changes, no strong sensing gain claimed. Step 2 is now implementing belief-only planning.
- Prospective learning comparisons and disjoint seeds are recorded in `next_study_protocol.json`; BC demonstrations count inside the shared interaction budget. Sensing weights remain explicitly pending the development-only planning sweep.

- Stage 1 supplement completed: 20 ten-step filter-calibration replays on development seeds, with per-source NIS and position NEES/coverage. Large inconsistencies concentrate in early updates; sensor floors and filter dynamics remain unchanged.
- Stage 2 completed: 14 pilot + 260 development + 70 held-out validation episodes. Moderate planner weights expose small sensing/communication gains with navigation-time costs; extreme weights cause deadline failures. Selected SA (2,0) and MPC (2000,0) remain fixed even where validation gains fail to replicate. SA search sensitivity uses horizon4 and16/64 iterations with matched sensing/navigation budgets.
- Stage 3 implementation completed: v1 compatibility, v2 public obs67/state72, independent observation/reward scales, legacy/log1p reward. Dynamic-dimension PPO rollout/update/save/reload verified.
- Stage 4 implementation completed: declared domain randomization, absolute-step curriculum, Goal BC with total-interaction accounting, uninterrupted100/300/500 budget snapshots, and hidden maneuver holdouts. Tiny CUDA training/evaluation and reporting integration completed; these are engineering checks, not learning results.
- Protocol frozen for formal runs: 24 models, 14,745,600 total training interactions (including demonstrations), 48 prespecified learned snapshots, 15 scenarios and 30 untouched final seeds. Expected final evaluation: 23,850 episodes including five rule methods. Full unit suite:127 passing tests before formal launch. Formal learning/results/publication remain pending.
