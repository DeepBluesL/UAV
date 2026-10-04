# Sequential ISAC study: development evidence archive (2026-10-04)

This directory preserves five completed development/validation studies byte-for-byte. It contains no model weights (`.pt`) and no final-test data. Raw CSV files, NPZ trajectories, figures, reports, run manifests, and source hashes are retained. Each planning study also includes its pre-run `source_snapshot.zip`.

## Contents and scope

| Directory | Scope | Episodes/records |
|---|---|---:|
| `sensing_diagnostics_step1_dev_4101_4110` | 10 development seeds × 2 scenarios × 7 feasible paths | 140 episodes |
| `filter_calibration_dev_4101_4110` | Goal, 2 scenarios, 10 development seeds, first 10 steps; per-source NIS and position NEES | 20 episodes, 600 source-step rows |
| `sensing_planning_pilot_4101` | One development seed, 2 scenarios, 7 planner settings | 14 episodes |
| `sensing_planning_dev_4101_4110` | 10 development seeds, 2 scenarios, 13 settings | 260 episodes |
| `sensing_planning_validation_4201_4205` | 5 reserved validation seeds, 2 scenarios, 7 frozen settings | 70 episodes |

Development seeds 4101–4110 and validation seeds 4201–4205 have distinct roles. No preregistered final-test seed from 5101–5130 appears here. Failed episodes, task regressions, and null results are retained.

## One-line reproduction commands

Use fresh output directories because the runners refuse to overwrite existing results.

```powershell
D:\anaconda3\envs\uav\python.exe -B -m project.sensing_diagnostics --output project/output/sensing_diagnostics_step1_dev_4101_4110_REPRO --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
D:\anaconda3\envs\uav\python.exe -B -m project.calibration_diagnostics --output project/output/filter_calibration_dev_4101_4110_REPRO --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
D:\anaconda3\envs\uav\python.exe -B -m project.planner_study --grid project/planner_study_pilot.json --output project/output/sensing_planning_pilot_4101_REPRO --seeds 4101
D:\anaconda3\envs\uav\python.exe -B -m project.planner_study --grid project/planner_study_dev_grid.json --output project/output/sensing_planning_dev_4101_4110_REPRO --seeds 4101 4102 4103 4104 4105 4106 4107 4108 4109 4110
D:\anaconda3\envs\uav\python.exe -B -m project.planner_study --grid project/planner_study_validation_grid.json --output project/output/sensing_planning_validation_4201_4205_REPRO --seeds 4201 4202 4203 4204 4205
```

## Provenance limits

Each run manifest records the base Git commit, dirty-tree state, configuration, and runtime. STEP1 and calibration manifests store SHA-256 hashes for relevant source files; the three planning directories store a complete pre-run source ZIP and its SHA-256. Because the worktree contained concurrent uncommitted development changes, checking out the recorded commit alone is insufficient for exact reproduction. Match the archived hashes, and treat each planning `source_snapshot.zip` as its authoritative run snapshot.

`archive_manifest.json` lists the relative path, byte count, and SHA-256 of every file in this archive except itself, allowing byte-preservation checks after copying or transfer.
