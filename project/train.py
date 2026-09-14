"""运行入口：python -m project.train；训练、评估和绘图保持同一组配置。"""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .artifacts import config_dict, load_policy, save_policy, write_csv, write_json
from .config import EnvConfig, PPOConfig, RewardConfig
from .core import MAPPOActorCritic
from .env import DualUAVEnv
from .evaluate import GoalController, evaluate
from .metrics import EpisodeRecorder
from .ppo import MAPPOBuffer, PPOTrainer


def collect_epoch(env, ac, buffer, obs, state, recorder):
    """采样一个批次；个体到达不结束团队轨迹，批次截止不重置环境。"""
    episodes, components, reward_sum = [], {}, 0.
    for step in range(buffer.max_size):
        active_before = env.active.copy()
        actions, logp = ac.act(obs, active_before)
        value = ac.value(state)
        next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
        buffer.store(obs, state, actions, reward, value, logp, active_before)
        recorder.add(reward, info)
        reward_sum += reward
        for key, contribution in info["reward_parts"].items():
            components[key] = components.get(key, 0.) + contribution

        obs, state = next_obs, next_state
        cutoff = step == buffer.max_size - 1
        if terminated or truncated or cutoff:
            # 有限任务期限也属于 terminated；与批次边界重合时仍然取零。
            last_value = 0. if terminated else ac.value(next_state)
            buffer.finish_path(last_value)
        if terminated or truncated:
            if terminated:
                episodes.append(recorder.summary())
            # 外部截断的片段不计为一次完整任务；当前环境本身不产生 truncated。
            obs, state, initial_info = env.reset()
            recorder = EpisodeRecorder(initial_info)
    stats = {"mean_step_reward": reward_sum / buffer.max_size}
    stats.update({f"mean_reward_{k}": v / buffer.max_size for k, v in components.items()})
    return obs, state, recorder, episodes, stats


def train(environment, reward, config, output):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    ac = MAPPOActorCritic(
        DualUAVEnv.obs_dim, DualUAVEnv.state_dim, hidden_sizes=config.hidden_sizes)
    trainer = PPOTrainer(ac, config)
    print(f"device={next(ac.parameters()).device} torch_threads={torch.get_num_threads()} "
          "environment=NumPy/CPU (one sequential environment)", flush=True)
    env = DualUAVEnv(environment, reward, seed=config.seed)
    obs, state, initial_info = env.reset(seed=config.seed)
    if env.terminated:
        raise ValueError("Training needs a nontrivial task; both UAVs already start at their goals")
    recorder = EpisodeRecorder(initial_info)
    buffer = MAPPOBuffer(
        env.obs_dim, env.state_dim, env.action_dim, config.steps_per_epoch, config.gamma, config.lam)
    epochs, episodes = [], []
    for epoch in range(config.epochs):
        ac.train()
        rollout_start = perf_counter()
        obs, state, recorder, completed, stats = collect_epoch(
            env, ac, buffer, obs, state, recorder)
        rollout_seconds = perf_counter() - rollout_start
        update_start = perf_counter()
        stats.update(trainer.update(buffer.get()))
        stats.update(rollout_seconds=rollout_seconds,
                     update_seconds=perf_counter() - update_start,
                     rollout_steps_per_second=config.steps_per_epoch / rollout_seconds)
        stats.update(epoch=epoch + 1, environment_steps=(epoch + 1) * config.steps_per_epoch)
        stats["completed_episodes"] = len(completed)
        for row in completed:
            row.update(episode=len(episodes) + 1, epoch=epoch + 1)
            episodes.append(row)
        epochs.append(stats)
        write_csv(output / "training.csv", epochs)
        write_csv(output / "episodes.csv", episodes)
        print(
            f"epoch={epoch + 1}/{config.epochs} steps={stats['environment_steps']} "
            f"completed={len(completed)} reward/step={stats['mean_step_reward']:.3f} "
            f"Vloss={stats['value_loss']:.3f} KL=({stats['kl_0']:.4f},{stats['kl_1']:.4f}) "
            f"rollout={stats['rollout_seconds']:.2f}s update={stats['update_seconds']:.2f}s "
            f"steps/s={stats['rollout_steps_per_second']:.0f}",
            flush=True)
    save_policy(output / "policy.pt", ac, environment, reward, config)
    # 最后未完成的片段不计入成功率；只保存已完成团队回合。
    write_json(output / "training_summary.json", {
        "environment_steps": config.epochs * config.steps_per_epoch,
        "completed_episodes": len(episodes),
        "team_success_rate": float(np.mean([r["team_success"] for r in episodes])) if episodes else None,
        "unfinished_episode_steps": recorder.steps,
    })
    return ac, epochs, episodes


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-UAV MAPPO training, frozen evaluation, and navigation baseline.")
    parser.add_argument("--mode", choices=("train", "evaluate", "baseline"), default="train")
    parser.add_argument("--config", type=Path, help="JSON containing environment/reward/ppo sections")
    parser.add_argument("--checkpoint", type=Path, help="policy.pt for frozen evaluation")
    parser.add_argument("--output", type=Path, help="Output directory inside UAV/project")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--eval-seeds", "--eval-seed", type=int, nargs="+", default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("--mode evaluate requires --checkpoint")
    if args.mode == "evaluate" and args.config is not None:
        parser.error("Evaluation reads configuration from --checkpoint; omit --config")
    training_options = (args.hidden_size, args.epochs, args.steps_per_epoch, args.seed)
    if args.mode == "evaluate" and any(value is not None for value in training_options):
        parser.error("Evaluation uses saved network/training settings; use --eval-seeds for its RNG seeds")
    return args


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
                 ("epochs", "steps_per_epoch", "seed", "device") if getattr(args, key) is not None}
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