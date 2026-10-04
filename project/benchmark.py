"""统一运行规则基线和冻结 MAPPO：python -m project.benchmark。"""

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import write_csv, write_json
from .benchmark_metrics import enrich_episode, paired_comparisons, summarize
from .benchmark_setup import METHODS, PROJECT_DIR, load_experiment, make_controller
from .env import DualUAVEnv
from .evaluate import rollout_policy


class TimedPolicy:
    """只计决策耗时，物理环境计算留在计时区域之外。"""

    def __init__(self, policy):
        self.policy, self.seconds = policy, 0.

    def act(self, obs, active, deterministic=True):
        start = perf_counter()
        result = self.policy.act(obs, active, deterministic=deterministic)
        self.seconds += perf_counter() - start
        return result

    def set_belief(self, estimated_target_state, tracking_covariance):
        if hasattr(self.policy, "set_belief"):
            self.policy.set_belief(estimated_target_state, tracking_covariance)


def evaluate_episode(policy, env, seed, prefix_steps):
    # 预热不进入统计；之后重置随机策略和环境，保证正式轨迹可复现。
    obs, _, info = env.reset(seed=seed)
    if not env.terminated:
        if hasattr(policy, "set_belief"):
            policy.set_belief(
                info["estimated_target_state"], info["tracking_covariance"])
        policy.act(obs, info["active"], deterministic=True)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    timed = TimedPolicy(policy)
    row, trace = rollout_policy(timed, env, seed)
    row = enrich_episode(row, trace, env.config, prefix_steps)
    row["policy_ms_per_step"] = 1000. * timed.seconds / row["length"] if row["length"] else None
    row["seed"] = seed
    return row, trace


def run_benchmark(args):
    learned, reward, scenarios, methods, seeds, metadata = load_experiment(args)
    if args.reference not in methods:
        raise ValueError("--reference must be among the evaluated methods")
    output = (args.output or PROJECT_DIR / "output" / (
        "benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if not output.is_relative_to(PROJECT_DIR):
        raise ValueError("Keep benchmark output inside project")
    # 一次实验对应一个新目录，避免把旧结果混入本次统计。
    output.mkdir(parents=True, exist_ok=False)
    metadata["controller_settings"] = {}
    for name, cfg in scenarios.items():
        metadata["controller_settings"][name] = {
            method: getattr(make_controller(method, cfg, metadata["suite"], learned), "settings", {})
            for method in methods}
    write_json(output / "manifest.json", metadata)
    rows = []
    for scenario, cfg in scenarios.items():
        for method in methods:
            policy = make_controller(method, cfg, metadata["suite"], learned)
            env = DualUAVEnv(cfg, reward)
            start = perf_counter()
            for index, seed in enumerate(seeds):
                row, trace = evaluate_episode(
                    policy, env, seed, metadata["scenario_prefix_steps"][scenario])
                row.update(scenario=scenario, method=method,
                           prefix_steps=metadata["scenario_prefix_steps"][scenario])
                rows.append(row)
                if index == 0:
                    trace_dir = output / "trajectories" / scenario / method
                    trace_dir.mkdir(parents=True)
                    np.savez_compressed(trace_dir / "trajectory.npz", **trace)
                    if not args.no_plots:
                        from .plot import plot_trajectory
                        plot_trajectory(trace_dir, trace)
                if (index + 1) % 25 == 0:
                    print(f"{scenario}/{method}: {index + 1}/{len(seeds)} episodes", flush=True)
            write_csv(output / "episodes.csv", rows)
            recent = rows[-len(seeds):]
            success = sum(row["team_success"] for row in recent) / len(seeds)
            print(f"{scenario}/{method}: success={success:.1%}, "
                  f"wall={perf_counter() - start:.1f}s", flush=True)
    summaries = summarize(rows)
    paired = paired_comparisons(rows, args.reference)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired.csv", paired)
    write_json(output / "summary.json", summaries)
    write_json(output / "paired.json", paired)
    if not args.no_plots:
        from .benchmark_plots import plot_benchmark
        plot_benchmark(output, summaries)
    metadata.update(status="complete", total_episodes=len(rows))
    write_json(output / "manifest.json", metadata)
    from .benchmark_report import write_report
    write_report(output, metadata, summaries, paired)
    print(f"Benchmark complete: {len(rows)} episodes; output={output}", flush=True)
    return rows, summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Compare frozen MAPPO and navigation baselines.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Use checkpoint environment/reward as the base")
    source.add_argument("--config", type=Path, help="Base JSON for experiments without a checkpoint")
    parser.add_argument("--suite", type=Path, default=PROJECT_DIR / "benchmark_config.json")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per scenario and method")
    parser.add_argument("--eval-seeds", "--eval-seed", type=int, nargs="+")
    parser.add_argument("--reference", choices=METHODS, default="goal")
    parser.add_argument("--device", default="cpu", help="Device for frozen policy inference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
