"""对比实验的场景、策略构造和可追溯配置；不修改训练环境。"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from .annealing import SimulatedAnnealingController
from .artifacts import config_dict, load_policy
from .baselines import PDController, PotentialFieldController, RandomController
from .config import EnvConfig, PPOConfig, RewardConfig
from .evaluate import GoalController
from .mpc import MPCController


CONTROLLERS = {
    "random": RandomController, "goal": GoalController, "pd": PDController,
    "apf": PotentialFieldController, "mpc": MPCController, "sa": SimulatedAnnealingController,
}
METHODS = (*CONTROLLERS, "mappo")
PROJECT_DIR = Path(__file__).resolve().parent


def load_experiment(args):
    """有 checkpoint 时以其配置为基准，压力场景仅覆盖明确列出的环境字段。"""
    checkpoint = None
    policy = None
    if args.checkpoint:
        policy, env, reward, ppo = load_policy(args.checkpoint, args.device)
        checkpoint = {
            "filename": args.checkpoint.name,
            "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "training_seed": ppo.seed,
            "control_mode": ppo.control_mode,
            "training_distribution": ppo.training_distribution,
            "training_environment_steps": ppo.epochs * ppo.steps_per_epoch,
            "note": "One frozen checkpoint; no training-seed uncertainty estimate.",
        }
    else:
        settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        env = EnvConfig(**settings.get("environment", {}))
        reward = RewardConfig(**settings.get("reward", {}))
        ppo = PPOConfig(**settings.get("ppo", {}))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    selected = args.scenarios or list(suite["scenarios"])
    scenarios = {}
    for name in selected:
        if not name.replace("_", "").isalnum():
            raise ValueError("Scenario names must contain only letters, digits and underscores")
        case = suite["scenarios"][name]
        scenarios[name] = replace(env, **case["environment"])
    methods = args.methods or list(CONTROLLERS) + (["mappo"] if policy is not None else [])
    if "mappo" in methods and policy is None:
        raise ValueError("The mappo method requires --checkpoint")
    if len(methods) != len(set(methods)) or len(selected) != len(set(selected)):
        raise ValueError("Methods and scenarios must not contain duplicates")
    seeds = args.eval_seeds or list(range(args.seed_start, args.seed_start + args.episodes))
    if not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds)):
        raise ValueError("Use a nonempty list of distinct nonnegative evaluation seeds")
    prefix = int(suite.get("prefix_steps", 10))
    if prefix < 1:
        raise ValueError("prefix_steps must be positive")
    metadata = {
        "status": "running", "base_config": config_dict(env, reward, ppo),
        "checkpoint": checkpoint, "suite": suite, "evaluation_seeds": seeds,
        "methods": methods, "prefix_steps": prefix,
        "scenarios": {name: asdict(cfg) for name, cfg in scenarios.items()},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "evaluation_device": args.device,
                    "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
        "statistics": {"confidence": .95, "success": "Wilson", "mean": "percentile bootstrap",
                       "bootstrap_samples": 2000, "paired_reference": args.reference},
        "protocol": [
            "Same scenarios, constraints, team reward and paired evaluation seeds for all methods.",
            "Environment RNG consumption can diverge after different active-source histories.",
            "No tuning on evaluation outcomes; non-nominal cases are specified stress tests.",
            "MPC and SA are centralized navigation references with known dynamics, not full ISAC optimizers.",
            "Fixed-prefix rho excludes t=0; episodes shorter than the prefix remain missing.",
            "Timing covers warmed policy.act wall time only; environment and plotting are excluded.",
            "Figures use the first listed evaluation seed, without selecting a favorable trajectory.",
        ],
    }
    root = PROJECT_DIR.parent
    metadata["source_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    metadata["source_tree_dirty"] = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip())
    metadata["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(PROJECT_DIR.glob("*.py"))}
    return policy, reward, scenarios, methods, seeds, metadata


def make_controller(method, config, suite, learned_policy):
    if method == "mappo":
        if hasattr(learned_policy, "set_environment"):
            learned_policy.set_environment(config)
        return learned_policy
    return CONTROLLERS[method](config, **suite.get("methods", {}).get(method, {}))
