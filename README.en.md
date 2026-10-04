# UAV: Dual-UAV Navigation and Collaborative Sensing

[简体中文](README.md) | **English**

A MAPPO-based simulation of integrated sensing and communication for two UAVs. Two independent Actors control the friendly UAVs, a centralized Critic learns the team return, and the base station uses deterministic beamforming rules. Both UAVs must safely reach their destinations before the deadline while collaboratively sensing a moving target. A UAV stops acting and uploading sensing data after arrival.

The `ekf` mode implements noisy measurements → fusion filter → target estimate → subsequent decisions. Target truth is used for simulation and offline scoring; it is not provided directly to the Actors, Critic, or planners. The measurement-noise model has not been calibrated against physical sensors.

## Current Features and Experiment Status

- The project supports pure RL, residual RL built on Goal navigation, and rule-based Goal, PD, artificial-potential-field, MPC, and simulated-annealing controllers.
- The new study adds sensing-aware SA/MPC, v2 observations, reward ablations, curriculum learning, and Goal behavioral-cloning initialization. Its code and development-validation evidence are merged into `main`.
- **As of 2026-10-05, the formal 24-model study has not completed final aggregation and archival.** Completed results are documented in the [historical closed-loop study](project/docs/CLOSED_LOOP_RESULTS.en.md) and the [current development-validation archive](project/experiments/isac_development_20261004/README.en.md). Interpret them separately from the current final evaluation.

## Project Structure and Configurations

```text
UAV/
├── project/          Current implementation: environment, physics, filtering, PPO, and experiment entry points
│   ├── docs/         Usage guides, study protocols, and results documentation
│   ├── experiments/  Published experiment data and figures
│   ├── tests/        Unit and integration tests
│   └── output/       Local models, logs, and results; ignored by Git
├── legacy/           Historical code for reference only
├── requirements.txt  Project dependency entry point
└── README.en.md      English documentation
```

| Configuration | Purpose |
|---|---|
| [closed_loop_config.json](project/closed_loop_config.json) | One EKF closed-loop training run with CPU rollout inference and CUDA batch updates |
| [next_study_protocol.json](project/next_study_protocol.json) | Full study: 24 models, three training seeds, and 48 prespecified snapshot evaluations |
| [example_config.json](project/example_config.json) | Historical `proxy` mode for reproducing older experiments |

## Installation

All commands below are single-line **Windows CMD** commands. Run every Python command from the repository root. Skip repository and environment creation if they already exist.

```cmd
git clone https://github.com/DeepBluesL/UAV.git
cd UAV
conda create -n uav python=3.12 -y
conda activate uav
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

This example uses the CUDA 13.0 PyTorch wheel index. For other GPU and driver combinations, select a build on the [official PyTorch installation page](https://pytorch.org/get-started/locally/). Physics simulation runs in NumPy on the CPU, while the GPU handles batched network updates, so GPU utilization will not remain high continuously.

## Single-Run Training, Evaluation, and Plotting

First run a short training job to check the workflow. This does not demonstrate convergence:

```cmd
python -m project.train --config project/closed_loop_config.json --control-mode residual --epochs 2 --steps-per-epoch 128 --seed 7 --eval-seed 101 102 --output project/output/quick_check
```

Run closed-loop residual RL. Change `--control-mode residual` to `--control-mode pure` for pure RL:

```cmd
python -m project.train --config project/closed_loop_config.json --control-mode residual --seed 7 --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7
```

Evaluation reads the configuration saved in the model. The following uses the CPU for small-batch inference; training still uses CUDA updates according to its configuration:

```cmd
python -m project.train --mode evaluate --checkpoint project/output/run_seed7/policy.pt --device cpu --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7_eval
python -m project.replot --output project/output/run_seed7
```

Results include `policy.pt`, CSV/JSON logs, and PNG figures. Trajectory plots show the base station, friendly UAVs, their destinations, and the target's true and estimated trajectories. Use a new output directory for every run: a single training run may overwrite files with the same names, while batch studies reject an existing output directory. Optimizer state is not saved, so exact training resumption is unsupported. For CPU-only training, add `--device cpu --rollout-device cpu`.

## Baselines and the Full Study

Compare the six default rule-based methods: random actions, Goal, PD, artificial potential fields, navigation MPC, and SA:

```cmd
python -m project.benchmark --config project/closed_loop_config.json --seed-start 2001 --episodes 30 --output project/output/comparison_rules
```

Run the complete new study, including training, fixed-budget evaluation, and aggregation. It takes substantially longer than a single training run:

```cmd
python -m project.next_study --config project/next_study_protocol.json --output project/output/my_next_study --stage all --jobs 3
```

`--jobs 3` starts three independent processes. The primary comparison uses the same budget of 204,800 joint environment interactions, including behavioral-cloning demonstrations; selected variants also save 300/500-epoch snapshots. The complete formal evaluation plan contains 15 scenarios, 30 evaluation seeds, and 23,850 episodes. See the [new study guide](project/docs/NEXT_STUDY_GUIDE.en.md) for staged commands, seed roles, and parameter changes. The historical nine-model study is described in the [closed-loop study guide](project/docs/CLOSED_LOOP_STUDY.en.md).

Inspect success rate, safety interventions, collisions, and completion time before comparing tracking RMSE, covariance, and communication over the same time window. Lower covariance does not necessarily mean lower realized error, and total returns are not directly comparable across reward configurations. In the current formal protocol, equal-reward pure/residual comparisons are limited to navigation; nonzero sensing-reward ablations are performed within residual RL.

## Where to Make Changes

| Content | Files |
| --- | --- |
| Scenarios, hyperparameters, and experiment groups | `project/config.py` and the corresponding JSON configuration |
| Motion, exit semantics, observations, and rewards | `project/env.py`, `observations.py`, `observation_specs.py`, `rewards.py` |
| Channel, communication, and sensing physics | `project/physics.py`, `channels.py`, `communication.py`, `sensing.py`, `crb.py`, `pcrb.py` |
| Simulated measurements and fusion filtering | `project/tracking.py` |
| Networks, PPO, and action mapping | `project/core.py`, `ppo.py`, `rollout.py`, `control.py` |
| Randomized scenarios, curriculum, and imitation initialization | `project/training_scenarios.py`, `domain_randomization.py`, `behavior_cloning.py` |
| Sensing planning and result aggregation | `project/sensing_planning.py`, `nominal_links.py`, `next_study_summary.py`, `next_study_plots.py` |

```cmd
python -B -m unittest discover -s project/tests -v
```

More documentation: [module index](project/README.md) · [user guide](project/docs/GUIDE.md) · [physics formula migration](project/docs/PHYSICS_MIGRATION.md) · [baseline guide](project/docs/BASELINES.md) · [experiment archive](project/experiments/README.md). Some detailed pages are available only in Chinese where no English companion exists.

The PPO implementation references OpenAI Spinning Up. The license for the relevant third-party code is retained in [LICENSE-spinningup.txt](project/LICENSE-spinningup.txt); it is not a licensing statement for the repository as a whole.
