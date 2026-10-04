# UAV: Dual-UAV Navigation and Collaborative Sensing

[简体中文](README.md) | **English**

A MAPPO-based simulation of integrated sensing and communication for two UAVs: two independent Actors control UAV motion, while a centralized Critic learns the team return. The task is to reach each UAV's destination within the time limit while balancing sensing, communication, and safety.

## Project Structure

```text
UAV/
├── project/                 Current main implementation; environment, physics formulas, and PPO are contained in the package
│   ├── train.py             Entry point for training, evaluation, and navigation baselines
│   ├── example_config.json  Experiment configuration (uses CUDA by default)
│   ├── tests/               Unit and integration tests
│   ├── docs/                Detailed usage guide, formula migration, and verification records
│   └── output/              Local models, logs, and plots; not uploaded to Git
├── legacy/                  Legacy environment, trainer, and 2uav reference code
├── requirements.txt         Installation entry point; references project/requirements.txt
├── README.md                Chinese
└── README.en.md             English
```

## Installation

All commands below are for **Windows CMD**, with one command per line. If you already have the repository and the `uav` environment, simply enter the repository root and activate the environment.

```bat
git clone https://github.com/DeepBluesL/UAV.git
cd UAV
conda create -n uav python=3.12 -y
conda activate uav
```

Install the GPU version of PyTorch first, then install the project dependencies. The commands below use the CUDA 13.0 wheel index; for other GPU or driver combinations, select the appropriate build on the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

```bat
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Training and Evaluation

Run all commands from the repository root (the parent directory of `project`). For a full training run:

```bat
python -m project.train --config project/example_config.json --seed 7 --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7
```

To check that the workflow runs, first perform a short training run; it cannot be used to assess convergence:

```bat
python -m project.train --config project/example_config.json --epochs 2 --steps-per-epoch 128 --seed 7 --eval-seed 101 102 --output project/output/quick_check
```

Evaluate the model saved after training:

```bat
python -m project.train --mode evaluate --checkpoint project/output/run_seed7/policy.pt --device cuda --eval-seed 1001 1002 1004 1005 --output project/output/run_seed7_eval
```

Regenerate plots and run tests:

```bat
python -m project.replot --output project/output/run_seed7
python -B -m unittest discover -s project/tests -v
```

The example configuration uses the GPU. To run on the CPU, add `--device cpu` to the training or evaluation command. The neural networks can use the GPU; the NumPy physics environment still executes serially on the CPU. `--eval-seed` and `--eval-seeds` are equivalent.

Results are saved in the specified subdirectory under `project/output/`, including the `policy.pt` model, training and evaluation logs, and PNG plots. Trajectory plots show the base station, both UAVs, and the rogue UAV's ground-truth and estimated trajectories. **Use a different output directory for each experiment**; files with the same name will be overwritten. Resuming training from a checkpoint is not currently supported.

## Baseline Comparisons

Compare random actions, goal seeking, PD, artificial potential fields, and short-horizon navigation MPC, and online simulated annealing (SA) under the same evaluation protocol:

```bat
python -m project.benchmark --config project/example_config.json --seed-start 2001 --episodes 100 --output project/output/comparison_rules
```

See the [benchmark guide](project/docs/BASELINES.md) for adding a trained MAPPO checkpoint, scenarios, and statistics, and the [results analysis](project/docs/BASELINE_RESULTS.md) for measured findings (both in Chinese).

## Measurement Feedback, Generalization, and Residual RL

The new mode connects simulated noisy measurements, EKF fusion, posterior estimates, and subsequent decisions. The study compares fixed-task pure RL, randomized-task pure RL, and randomized-task residual RL with equal budgets and three training seeds each. It also tests bounds, speed limits, and sensing ablations:

```bat
python -m project.study --config project/study_config.json --jobs 3 --output project/output/my_study
```

This trains nine models and evaluates them on common scenarios and seeds. The new configuration uses CPU rollout inference and CUDA batch updates to reduce per-step synchronization overhead. See the [English closed-loop study guide](project/docs/CLOSED_LOOP_STUDY.en.md), its [Chinese companion](project/docs/CLOSED_LOOP_STUDY.md), and the [measured SA / limit-change results](project/docs/LIMITS_SA_RESULTS.md) (Chinese). Use `closed_loop_config.json` for the new mode; `example_config.json` retains the proxy model for historical comparisons.

Measured closed-loop findings are summarized in the [English results companion](project/docs/CLOSED_LOOP_RESULTS.en.md), with the complete reproducibility archive under [`project/experiments/closed_loop_20260927`](project/experiments/closed_loop_20260927/README.md).

## Where to Make Changes

| Content | Location |
| --- | --- |
| Scenario, reward weights, learning rate, and training budget | [example_config.json](project/example_config.json); see [config.py](project/config.py) for all fields |
| Motion, per-UAV exit on arrival, and observations | `project/env.py`, `project/observations.py` |
| Reward expressions | [rewards.py](project/rewards.py) |
| Networks, GAE, and PPO updates | `project/core.py`, `project/ppo.py` |
| Channels, SINR, CRB/PCRB | `project/channels.py`, `communication.py`, `sensing.py`, `measurements.py`, `crb.py`, `pcrb.py` |
| Training statistics and plotting | `project/train.py`, `metrics.py`, `plot.py`, `plot_trajectory.py` |

For details, see the [User Guide](project/docs/GUIDE.md), [Physics Formula Migration](project/docs/PHYSICS_MIGRATION.md), and [Verification Records](project/docs/VERIFICATION.md) (in Chinese). See [legacy/README.md](legacy/README.md) for the purpose and usage of the legacy code.

The rogue UAV's ground truth is not an Actor/Critic input. The `proxy` mode retains noisy state estimates and a PCRB proxy; `ekf` uses simulated measurements and recursive fusion. Filter covariance and realized tracking RMSE are reported separately. Measurement noise is not calibrated against physical sensors.

The PPO implementation references OpenAI Spinning Up. The third-party license is retained in [LICENSE-spinningup.txt](project/LICENSE-spinningup.txt). This license applies to the relevant third-party code and does not constitute a licensing statement for the entire repository.

## Sensing Planning and Learning Budgets

The next study adds belief-only sensing SA/MPC, v2 observations, reward ablations, curriculum training, and Goal initialization. See the [English guide](project/docs/NEXT_STUDY_GUIDE.en.md) / [中文指南](project/docs/NEXT_STUDY_GUIDE.md) for single-line CMD commands, budgets, and editing locations.

Development evidence for path controllability, filter consistency, and sensing planning is archived in [English](project/experiments/isac_development_20261004/README.en.md) / [中文](project/experiments/isac_development_20261004/README.md). Formal multi-seed learning results are reported separately.
