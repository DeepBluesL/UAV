"""Train, evaluate, or run a baseline with one shared configuration."""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .artifacts import config_dict, load_policy, save_policy, write_csv, write_json
from .behavior_cloning import collect_goal_demonstrations, fit_goal_behavior
from .config import EnvConfig, PPOConfig, RewardConfig
from .core import MAPPOActorCritic
from .env import DualUAVEnv
from .evaluate import GoalController, evaluate
from .metrics import EpisodeRecorder
from .observation_specs import observation_spec
from .ppo import MAPPOBuffer, PPOTrainer
from .rollout import collect_epoch
from .training_scenarios import TrainingScenarioSampler
from .training_helpers import append_csv, print_epoch, save_epoch_checkpoint
from .train_cli import parse_args


def train(environment, reward, config, output):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    sampler = TrainingScenarioSampler(environment, config.training_distribution, config.seed)
    initial_environment = sampler.sample()
    spec = observation_spec(initial_environment)
    ac = MAPPOActorCritic(
        spec.obs_dim, spec.state_dim, hidden_sizes=config.hidden_sizes,
        environment=initial_environment, control_mode=config.control_mode,
        residual_scale=config.residual_scale)
    trainer = PPOTrainer(ac, config)
    rollout_device = config.rollout_device or config.device
    print(f"rollout_device={rollout_device} update_device={config.device} "
          f"torch_threads={torch.get_num_threads()} "
          "environment=NumPy/CPU (one sequential environment)", flush=True)
    env = DualUAVEnv(initial_environment, reward, seed=config.seed)
    bc_stats = {}
    bc_seconds = 0.0
    if config.initialization == "goal_bc":
        bc_start = perf_counter()
        demonstrations = collect_goal_demonstrations(
            env, sampler, config.demonstration_steps)
        bc_stats = fit_goal_behavior(ac, demonstrations, config)
        bc_seconds = perf_counter() - bc_start
        write_json(output / "initialization.json", {
            "initialization": config.initialization,
            "demonstration_steps": config.demonstration_steps,
            "bc_epochs": config.bc_epochs,
            "bc_batch_size": config.bc_batch_size,
            "bc_lr": config.bc_lr,
            "bc_seconds": bc_seconds,
            "loss_space": "executed_normalized_acceleration",
            "unchanged_parameters": ["critic", "actor_log_std"],
            **bc_stats,
        })
        initial_environment = sampler.sample()
        ac.set_environment(initial_environment)
        env = DualUAVEnv(initial_environment, reward, seed=config.seed)
    obs, state, initial_info = env.reset(seed=config.seed)
    if env.terminated:
        raise ValueError("Training needs a nontrivial task; both UAVs already start at their goals")
    recorder = EpisodeRecorder(initial_info)
    buffer = MAPPOBuffer(
        env.obs_dim, env.state_dim, env.action_dim, config.steps_per_epoch, config.gamma, config.lam)
    epochs, episodes = [], []
    demonstration_epochs = config.demonstration_steps // config.steps_per_epoch
    checkpoints = []
    for epoch in range(demonstration_epochs, config.epochs):
        ac.to(rollout_device)
        ac.train()
        rollout_start = perf_counter()
        obs, state, recorder, completed, stats = collect_epoch(
            env, ac, buffer, obs, state, recorder, sampler)
        rollout_seconds = perf_counter() - rollout_start
        ac.to(config.device)
        update_start = perf_counter()
        stats.update(trainer.update(buffer.get()))
        stats.update(rollout_seconds=rollout_seconds,
                     update_seconds=perf_counter() - update_start,
                     rollout_steps_per_second=config.steps_per_epoch / rollout_seconds,
                     rollout_device=rollout_device, update_device=config.device)
        stats.update(
            epoch=epoch + 1,
            ppo_epoch=epoch + 1 - demonstration_epochs,
            environment_steps=(epoch + 1) * config.steps_per_epoch,
            demonstration_steps=config.demonstration_steps,
            ppo_environment_steps=(epoch + 1) * config.steps_per_epoch
                                  - config.demonstration_steps,
            sampler_interactions=sampler.interactions,
            training_phase=sampler.phase,
        )
        stats["completed_episodes"] = len(completed)
        for row in completed:
            row.update(episode=len(episodes) + 1, epoch=epoch + 1)
            episodes.append(row)
        epochs.append(stats)
        write_csv(output / "training.csv", epochs)
        append_csv(output / "episodes.csv", completed)
        if epoch + 1 in config.checkpoint_epochs:
            checkpoints.append(save_epoch_checkpoint(
                output, ac, environment, reward, config, epoch + 1))
            ac.set_environment(env.config)
        print_epoch(stats, config.epochs)
    # The returned/saved policy is evaluated on the requested base scenario.
    ac.to(config.device)
    ac.set_environment(environment)
    save_policy(output / "policy.pt", ac, environment, reward, config)
    # 最后未完成的片段不计入成功率；只保存已完成团队回合。
    write_json(output / "training_summary.json", {
        "environment_steps": config.epochs * config.steps_per_epoch,
        "demonstration_steps": config.demonstration_steps,
        "ppo_environment_steps": config.epochs * config.steps_per_epoch
                                 - config.demonstration_steps,
        "sampler_interactions": sampler.interactions,
        "final_training_phase": sampler.phase,
        "initialization": config.initialization,
        "bc_teacher_scope": "Public-observation goal navigation; demonstrations may fail hard tasks",
        "bc_seconds": bc_seconds,
        **bc_stats,
        "checkpoints": checkpoints,
        "completed_episodes": len(episodes),
        "team_success_rate": float(np.mean([r["team_success"] for r in episodes])) if episodes else None,
        "unfinished_episode_steps": recorder.steps,
        "training_distribution": config.training_distribution,
        "randomized_mixture": (
            {"nominal": .25, "crossing": .25, "continuous": .50}
            if config.training_distribution == "randomized" else None),
    })
    return ac, epochs, episodes


def main():
    args = parse_args()
    if args.mode == "evaluate":
        policy, environment, reward, ppo = load_policy(args.checkpoint, args.device or "cpu")
    else:
        settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        environment = EnvConfig(**settings.get("environment", {}))
        reward = RewardConfig(**settings.get("reward", {}))
        ppo = PPOConfig(**settings.get("ppo", {}))
    overrides = {key: getattr(args, key) for key in
                 ("epochs", "steps_per_epoch", "seed", "device", "control_mode",
                  "residual_scale", "training_distribution", "rollout_device",
                  "initialization", "demonstration_steps", "bc_epochs",
                  "bc_batch_size", "bc_lr", "checkpoint_epochs")
                 if getattr(args, key) is not None}
    if args.hidden_size is not None:
        overrides["hidden_sizes"] = (args.hidden_size, args.hidden_size)
    ppo = replace(ppo, **overrides)
    if args.max_steps is not None:
        environment = replace(environment, max_steps=args.max_steps)

    project_dir = Path(__file__).resolve().parent
    output = (args.output or project_dir / "output" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if not output.is_relative_to(project_dir):
        raise ValueError("Keep generated files under UAV/project, as required by this project")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config_dict(environment, reward, ppo))
    epochs, episodes = [], []
    if args.mode == "train":
        policy, epochs, episodes = train(environment, reward, ppo, output)
    elif args.mode == "baseline":
        policy = GoalController(environment)

    summary, evaluation_rows, trace = evaluate(policy, environment, reward, args.eval_seeds)
    write_json(output / "evaluation.json", {
        "policy": args.mode, "seeds": args.eval_seeds, "summary": summary, "episodes": evaluation_rows})
    write_csv(output / "evaluation.csv", evaluation_rows)
    np.savez_compressed(output / "trajectory.npz", **trace)
    if not args.no_plots:
        from .plot import plot_results
        plot_results(output, epochs, episodes, trace)
    print(
        f"evaluation team_success={summary['team_success_rate']:.1%} "
        f"rho_pos={summary['mean_rho_pos']:.6g}; output={output}", flush=True)


if __name__ == "__main__":
    main()
