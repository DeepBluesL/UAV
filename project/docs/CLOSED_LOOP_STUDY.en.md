# Closed-Loop Measurement and RL Study

This study adds simulated sensor measurements, EKF fusion, online simulated annealing (SA), randomized training tasks, and residual RL. It compares pure and residual policies with equal environment-step budgets. This guide describes the protocol and artifacts; measured findings are reported separately after all runs finish.

The historical `example_config.json` uses the `proxy` sensing mode. The new `closed_loop_config.json` uses `ekf`. Their uncertainty values have different meanings and must not be compared directly.

## Run the study

Run every command from the repository root. The examples are single-line commands for Windows CMD. Activate the `uav` Conda environment with a CUDA-enabled PyTorch installation first.

Run training, evaluation, aggregation, and plotting:

```bat
python -m project.study --config project/study_config.json --jobs 3 --output project/output/my_study
```

The supplied study configuration uses three training seeds, three policy variants, 100 epochs × 2,048 joint environment steps per model, and 30 common evaluation seeds. It trains nine models. `--jobs` controls independent training processes; it does not create parallel environments within one model. Each model is evaluated from its final checkpoint, without selecting checkpoints on the test results.

The primary comparison uses `pure_fixed_cpu`, `pure_randomized`, and `residual_randomized`, with three seeds each and CPU rollout plus CUDA updates throughout. Three earlier `pure_fixed` runs that used GPU rollout were allowed to finish and are retained as an auxiliary execution check. They are not pooled with the primary fixed-task models. The run-specific device choice and source snapshot are recorded with each artifact.

Run the stages separately when needed:

```bat
python -m project.study --config project/study_config.json --stage train --jobs 3 --output project/output/my_study
python -m project.study --stage evaluate --output project/output/my_study
python -m project.study --stage summarize --output project/output/my_study
```

Training does not resume optimizer state after interruption. Use a new output directory for a new study. Configuration snapshots are written under `configs/`, and each model writes progress to `train/<variant>_seed<seed>/console.log`.

Train one model or run rule-based references:

```bat
python -m project.train --config project/closed_loop_config.json --control-mode pure --training-distribution randomized --rollout-device cpu --device cuda --seed 7 --output project/output/pure_seed7
python -m project.train --config project/closed_loop_config.json --control-mode residual --training-distribution randomized --residual-scale 0.25 --rollout-device cpu --device cuda --seed 7 --output project/output/residual_seed7
python -m project.benchmark --config project/closed_loop_config.json --suite project/study_scenarios.json --methods goal mpc sa --seed-start 3001 --episodes 30 --output project/output/rules_ekf
```

`rollout_device=cpu` runs step-by-step policy inference on the CPU, where the NumPy environment, channels, and filter already run. `device=cuda` moves the policy back to the GPU for PPO batch updates. This avoids repeated small GPU launches and CPU/GPU synchronization during collection. It does not change the sampled-action or PPO objective definitions. The backward-compatible default `rollout_device=null` uses the update device for both phases.

## Measurement feedback loop

At each step:

1. Each UAV acts from its local observation, and the simulated unauthorized UAV advances.
2. The base station predicts the target state and covariance from the previous EKF posterior.
3. The predicted position determines sensing beams. Ground truth is used only by propagation and simulated measurement generation.
4. Each sensor uses its own geometry and Jacobian. The BS range is monostatic round-trip distance. A UAV range is the bistatic BS–target–UAV path length. Measurements also include receiver azimuth, elevation, and path range rate.
5. The BS measurement is fused locally. A UAV uploads only while active and when its communication SINR passes the threshold. This SINR is an upload-success proxy; the model has no separate uplink queue, channel, or delay. Posterior broadcast is reliable and instantaneous.
6. The EKF applies wrapped angular innovations and Joseph-form covariance updates. Its posterior estimate and covariance enter the next Actor/Critic observation.

The filter starts from one noisy prior. It does not receive a fresh “truth plus noise” state each step. With measurements disabled, it only predicts from its own history and can drift from the target.

Measurement variances are derived from the existing SINR-to-CRB formulas and physical-unit noise floors: 1 m path length, 0.5° angles, and 0.5 m/s path range rate by default. Initial position and velocity standard deviations are 10 m and 1 m/s. These are simulated assumptions, not calibration from physical radar hardware. Sequential EKF updates assume conditionally independent measurement noise.

For compatibility, `rho_pos` and `pcrb` remain in saved data. When `uncertainty_kind=ekf_covariance`, they contain posterior covariance and its position-block trace; they are neither a PCRB nor realized tracking error. `tracking_prefix_rmse` is the three-dimensional position error against simulation truth over the common first ten steps. Ground truth is used for offline scoring and diagnostic plots, not policy input.

## Pure, residual, and randomized policies

- `pure_fixed`: Actors directly produce actions and train on the fixed nominal task.
- `pure_randomized`: the same Actor/Critic and optimizer train on tasks sampled per episode.
- `residual_randomized`: randomized training with goal navigation plus learned Actor residuals.

Residual execution is

```text
a = norm_limit(a_goal + residual_scale * amax * tanh(raw))
```

`a_goal` is the executable acceleration produced by the existing Goal controller after environment constraints. A zero residual therefore reproduces Goal navigation. `residual_scale=0.25` bounds each residual coordinate, not the residual vector norm; the combined command still obeys the total acceleration-norm limit. Actor output layers start with zero deterministic means, while Gaussian exploration remains active during training.

The rollout buffer stores the original Gaussian sample and its log-probability. Action adaptation occurs only before `env.step`, so PPO probability ratios and Gaussian entropy remain defined in the original action space.

Randomized training samples 25% nominal tasks, 25% crossing tasks, and 50% continuous start/goal tasks. Continuous tasks include forward and reverse goals. Goal separation is preserved, and deadlines include at least ten steps beyond the straight-line arrival lower bound. Deadline and CSI difficulty vary; speed, acceleration, and world bounds remain fixed. This geometric margin is not a proof that every task is collision-free or easy for a policy. Scenario sampling has an RNG independent of sensor and channel noise, and official evaluation seeds are not used for training or tuning.

## Evaluation scope

`study_scenarios.json` includes nominal, shifted, crossing, reverse-goal, wider-bound, higher-speed, noisy-CSI, short-deadline, and measurement-ablation cases. A changed speed limit also changes reward normalization, so returns and energy proxies should not be interpreted as physical efficiency across different speed scenarios. Prefer success rate, seconds, path length in meters, tracking RMSE, and safety events.

SA is a centralized, model-based, short-horizon navigation reference. By default it evaluates five reference sequences and 16 Metropolis proposals over a four-step horizon. It uses current available state and goals, without querying future ground truth or future noise. It does not optimize the full ISAC objective and is not guaranteed globally optimal. Report its decision time alongside independent-Actor inference.

Statistics first average evaluation episodes within each checkpoint, then summarize across independent training seeds. Do not treat all episodes pooled across three checkpoints as independent training seeds. Three training seeds provide only a limited estimate of training variability.

## Output files

| File or directory | Contents |
| --- | --- |
| `configs/` | Frozen study, environment, scenario, and per-run configuration snapshots |
| `train/<variant>_seed<seed>/` | Checkpoint, training CSV files, summaries, and console log for one model |
| `eval/` | Per-episode data, hashes, configurations, and first-seed traces for policies and rule baselines |
| `seed_summary.csv` | Test result for each individual trained checkpoint |
| `study_summary.csv` | Aggregation across training seeds |
| `REPORT.md` | Generated measured-results report |
| `policy_comparison.png` | Success and restricted completion time by scenario |
| `learning_curves.png` | Median training curves and min–max range across seeds |
| `trajectory_comparison.png` | Fixed-seed paths, goals, base station, and target truth/estimate |
| `tracking_ablation.png` | Measurement-off, BS-only, and multisensor tracking comparison |

Inspect success first, then restricted/success time, boundary requests and safety interventions, followed by common-prefix tracking error, communication, and computation time. A zero collision count can coexist with many rejected unsafe actions, so include safety interventions. Full-episode tracking averages are not directly comparable when episode lengths differ.

## Main configuration files

| Purpose | File |
| --- | --- |
| Closed-loop environment, rewards, and PPO defaults | `project/closed_loop_config.json` |
| Training seeds, budgets, variants, and evaluation seeds | `project/study_config.json` |
| Evaluation scenarios and SA parameters | `project/study_scenarios.json` |
| Measurement geometry and EKF | `project/tracking.py` |
| Residual action mapping | `project/control.py` |
| Training-task distribution | `project/training_scenarios.py` |
| Raw-action rollout and PPO update | `project/rollout.py`, `project/ppo.py` |
| Study orchestration and reporting | `project/study.py`, `project/study_evaluate.py`, `project/study_summary.py`, `project/study_plots.py` |

The Chinese companion guide is [CLOSED_LOOP_STUDY.md](CLOSED_LOOP_STUDY.md).
