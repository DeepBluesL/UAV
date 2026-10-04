"""Small artifact helpers for uninterrupted, fixed-budget training."""

import csv
from dataclasses import replace
from pathlib import Path

from .artifacts import save_policy


def append_csv(path, rows):
    """Append rows with one stable header instead of rewriting the episode history."""
    if not rows:
        return
    path = Path(path)
    fields = list(rows[0])
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            existing = next(csv.reader(stream))
        if existing != fields or any(list(row) != fields for row in rows):
            raise ValueError("CSV append rows must preserve the existing field order")
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_epoch_checkpoint(output, ac, environment, reward, config, epoch):
    """Save an inference-only snapshot without restarting PPO optimizers."""
    filename = f"policy_epoch{epoch}.pt"
    ac.set_environment(environment)
    save_policy(
        output / filename, ac, environment, reward,
        replace(config, epochs=epoch, checkpoint_epochs=()))
    total = epoch * config.steps_per_epoch
    return {
        "filename": filename, "epoch": epoch, "environment_steps": total,
        "demonstration_steps": config.demonstration_steps,
        "ppo_environment_steps": total - config.demonstration_steps,
    }


def print_epoch(stats, total_epochs):
    print(
        f"epoch={stats['epoch']}/{total_epochs} steps={stats['environment_steps']} "
        f"completed={stats['completed_episodes']} "
        f"reward/step={stats['mean_step_reward']:.3f} "
        f"Vloss={stats['value_loss']:.3f} "
        f"KL=({stats['kl_0']:.4f},{stats['kl_1']:.4f}) "
        f"rollout={stats['rollout_seconds']:.2f}s "
        f"update={stats['update_seconds']:.2f}s "
        f"steps/s={stats['rollout_steps_per_second']:.0f}", flush=True)
