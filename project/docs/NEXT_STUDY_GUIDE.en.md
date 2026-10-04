# Sensing planning, reward ablations, and learning budgets

[简体中文](NEXT_STUDY_GUIDE.md) · [English home](../../README.en.md)

The sequence is path controllability, belief-aware planning, observation/reward changes, learning/generalization, and evidence archiving. Historical experiments stay separate. Passing implementation tests does not establish policy performance.

## Run from Windows CMD

Use a new output directory for each study. Commands run from the repository root; each command occupies one line.

```bat
conda activate uav
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -B -m project.sensing_diagnostics --output project/output/path_diagnostic --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
python -B -m project.calibration_diagnostics --output project/output/filter_diagnostic --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
python -B -m project.planner_study --grid project/planner_study_dev_grid.json --output project/output/planner_development --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
python -B -m project.next_study --config project/next_study_protocol.json --output project/output/my_next_study --stage all --jobs 3
```

The full study trains 24 models across three independent seeds and evaluates prespecified budget snapshots. This is a substantial batch experiment. CPU rollout inference and CUDA PPO updates run in each process; NumPy physics remains on the CPU. Three jobs means three independent training processes.

Alternatively run stages separately, choosing `train` instead of `all` for the initial command:

```bat
python -B -m project.next_study --config project/next_study_protocol.json --output project/output/my_next_study --stage train --jobs 3
python -B -m project.next_study --output project/output/my_next_study --stage evaluate --jobs 3
python -B -m project.next_study --output project/output/my_next_study --stage summarize
```

Evaluation reads the frozen suite inside the output directory and can skip verified complete evaluations. Training requires a new directory. Snapshots are inference artifacts, not optimizer-resume checkpoints; the budget curve comes from uninterrupted training.

## Editing map

| Concern | Files |
|---|---|
| Variants, disjoint seeds, budgets, sensing coefficients | `next_study_protocol.json` |
| Holdout scenarios, fixed windows, planner settings | `next_study_scenarios.json` |
| Simulated measurements and EKF | `tracking.py`, `physics.py` |
| Declared noise floors and configuration | `config.py` |
| Belief-only SA/MPC and nominal channel forecast | `sensing_planning.py`, `nominal_links.py` |
| Versioned public observation layout | `observations.py`, `observation_specs.py` |
| Legacy/log1p reward | `rewards.py`, `RewardConfig` |
| Curriculum and declared randomization ranges | `training_scenarios.py`, `domain_randomization.py` |
| Goal-supervised initialization | `behavior_cloning.py` |
| PPO rollout, budget accounting, snapshots | `train.py`, `rollout.py`, `training_helpers.py` |
| Hidden target maneuver | `target_motion.py` |
| Fixed-window metrics and hierarchical plots | `window_metrics.py`, `next_study_summary.py`, `next_study_plots.py` |

## Protocol

The primary budget is 100×2048=204800 joint environment interactions. Domain-randomized pure RL, curriculum, Goal-initialized pure RL, and residual navigation continue to epochs300/500. Observation/reward ablations stop at100. The8192 Goal-demonstration steps consume the first four logical epochs, so PPO starts at epoch5; offline supervised updates add computation but no environment interactions. Goal cloning fits executed normalized acceleration after tanh and norm clipping; the critic and log standard deviations retain their initial values. Demonstrations may fail crossing tasks, and the method is not from-scratch RL.

Curriculum phases are base tasks before50000 interactions, randomized geometry until150000, then full domain randomization. Public v2 input includes known motion limits, bounds, BS position, communication threshold and full belief covariance. Hidden target initial truth, acceleration and future turns never enter actors or critic. V1 remains31/58-dimensional and v2 is67/72; old checkpoints retain v1 behavior.

Navigation SA uses normalized distances while MPC uses meters. Their sensing coefficients are therefore not directly comparable. The forecast adds horizon-mean uncertainty and a smooth link-shortfall surrogate, stops mission costs after both arrivals, and uses ratios of expected channel powers rather than the exact expected random SINR. RL rewards are computed at each real step; planner weights do not mechanically transfer to reward weights. Centralized planners also have different model/coordination priors from the two local RL actors.

Read success, safety interventions, collisions and restricted completion time before tracking metrics. Failures receive the deadline. Compare actual RMSE, posterior covariance and communication over the same window: five steps for the short reverse task, ten otherwise. Early1–3 and later4–window-end metrics diagnose initialization transients without replacing the full window. NEES/NIS coverage is descriptive because within-episode samples are correlated; smaller covariance does not prove smaller realized error. Raw returns are not comparable across reward configurations.

Plots use the first declared seed, showing the BS, friendly UAVs, goals, and rogue truth/estimate. Truth remains simulator/diagnostic-only. Average evaluation episodes within each trained model, then summarize independent training seeds; do not select checkpoints on final test outcomes. The original SINR-to-CRB relation remains an uncalibrated variance proxy and noise floors are not lowered to manufacture gains.

```bat
python -B -m unittest discover -s project/tests -v
```
