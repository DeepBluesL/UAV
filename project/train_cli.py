"""Command-line parsing for training, evaluation, and baselines."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-UAV MAPPO training, frozen evaluation, and navigation baseline.")
    parser.add_argument("--mode", choices=("train", "evaluate", "baseline"), default="train")
    parser.add_argument("--config", type=Path, help="JSON with environment/reward/ppo")
    parser.add_argument("--checkpoint", type=Path, help="policy.pt for frozen evaluation")
    parser.add_argument("--output", type=Path, help="Output directory inside UAV/project")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--rollout-device")
    parser.add_argument("--control-mode", choices=("pure", "residual"))
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--training-distribution", choices=(
        "fixed", "randomized", "domain_randomized", "curriculum"))
    parser.add_argument("--initialization", choices=("random", "goal_bc"))
    parser.add_argument("--demonstration-steps", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--bc-batch-size", type=int)
    parser.add_argument("--bc-lr", type=float)
    parser.add_argument("--checkpoint-epochs", type=int, nargs="+")
    parser.add_argument("--eval-seeds", "--eval-seed", type=int, nargs="+",
                        default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("--mode evaluate requires --checkpoint")
    if args.mode == "evaluate" and args.config is not None:
        parser.error("Evaluation reads configuration from --checkpoint; omit --config")
    training = (
        args.hidden_size, args.epochs, args.steps_per_epoch, args.seed,
        args.rollout_device, args.control_mode, args.residual_scale,
        args.training_distribution, args.initialization, args.demonstration_steps,
        args.bc_epochs, args.bc_batch_size, args.bc_lr, args.checkpoint_epochs)
    if args.mode == "evaluate" and any(value is not None for value in training):
        parser.error("Evaluation uses saved training settings; use --eval-seeds for RNG seeds")
    return args
