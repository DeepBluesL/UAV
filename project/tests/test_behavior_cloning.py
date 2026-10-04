import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from project.artifacts import load_policy
from project.behavior_cloning import (
    _teacher_action,
    collect_goal_demonstrations,
    executed_normalized_action,
    fit_goal_behavior,
)
from project.config import EnvConfig, PPOConfig, RewardConfig
from project.core import MAPPOActorCritic
from project.env import DualUAVEnv
from project.train import train
from project.training_scenarios import TrainingScenarioSampler


def short_environment(**changes):
    values = dict(
        max_steps=4, goal_tolerance=.2, min_uav_distance=1.,
        uav_initial=np.array([[0., 0., 10.], [0., 4., 10.]]),
        uav_goal_positions=np.array([[5., 0., 10.], [8., 4., 10.]]),
        world_low=np.array([-10., -10., 1.]), world_high=np.array([20., 20., 30.]),
        target_initial_state=np.array([6., 8., 12., 0., 0., 0.]),
        target_acceleration=np.zeros(3), antenna_x=2, antenna_y=2,
        imperfect_csi_beta=0., csi_beta_jitter=0.)
    values.update(changes)
    return EnvConfig(**values)


class BehaviorCloningTests(unittest.TestCase):
    def test_teacher_target_matches_physical_execution_under_saturation(self):
        cfg = short_environment()
        env = DualUAVEnv(cfg, RewardConfig(), seed=2)
        obs, _, _ = env.reset(seed=2)
        raw = _teacher_action(obs, env.active, cfg)
        normalized = executed_normalized_action(torch.as_tensor(raw)).numpy()
        acceleration = np.tanh(raw) * cfg.max_uav_acceleration
        norms = np.linalg.norm(acceleration, axis=1, keepdims=True)
        acceleration /= np.maximum(1., norms / cfg.max_uav_acceleration)
        np.testing.assert_allclose(
            normalized, acceleration / cfg.max_uav_acceleration, atol=1e-7)
        saturated = torch.tensor([[20., 20., 20.]])
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(executed_normalized_action(saturated))),
            1., places=6)

    def test_goal_bc_reduces_loss_and_masks_inactive_agent(self):
        cfg = short_environment(
            uav_goal_positions=np.array([[0., 0., 10.], [8., 4., 10.]]))
        sampler = TrainingScenarioSampler(cfg, "fixed", 3)
        env = DualUAVEnv(sampler.sample(), RewardConfig(), seed=3)
        demonstrations = collect_goal_demonstrations(env, sampler, 8)
        self.assertEqual(len(demonstrations[0][0]), 0)
        self.assertGreater(len(demonstrations[1][0]), 0)
        self.assertEqual(sampler.interactions, 8)
        model = MAPPOActorCritic(31, 58, hidden_sizes=(8,), environment=cfg)
        config = PPOConfig(
            hidden_sizes=(8,), steps_per_epoch=4, epochs=3,
            initialization="goal_bc", demonstration_steps=4,
            bc_epochs=10, bc_batch_size=4, bc_lr=1e-2)
        critic_before = [value.detach().clone() for value in model.v.parameters()]
        log_std_before = [actor.log_std.detach().clone() for actor in model.pi]
        stats = fit_goal_behavior(model, demonstrations, config)
        self.assertEqual(stats["bc_samples_0"], 0)
        self.assertLess(stats["bc_loss_after_1"], stats["bc_loss_before_1"])
        for before, after in zip(critic_before, model.v.parameters()):
            torch.testing.assert_close(before, after)
        for before, actor in zip(log_std_before, model.pi):
            torch.testing.assert_close(before, actor.log_std)

    def test_total_budget_and_snapshots_include_demonstrations(self):
        cfg = replace(short_environment(), observation_version="v2")
        ppo = PPOConfig(
            hidden_sizes=(8,), steps_per_epoch=4, epochs=3,
            train_pi_iters=1, train_v_iters=1, seed=5,
            initialization="goal_bc", demonstration_steps=4,
            bc_epochs=2, bc_batch_size=4, checkpoint_epochs=(2, 3))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory)
            _, epochs, _ = train(cfg, RewardConfig(), ppo, output)
            summary = json.loads((output / "training_summary.json").read_text())
            initialization = json.loads((output / "initialization.json").read_text())
            with (output / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
                episode_rows = list(csv.DictReader(stream))
            snapshot, _, _, saved_config = load_policy(output / "policy_epoch2.pt")
            self.assertTrue((output / "policy_epoch3.pt").exists())
        self.assertEqual([row["epoch"] for row in epochs], [2, 3])
        self.assertEqual(summary["environment_steps"], 12)
        self.assertEqual(summary["demonstration_steps"], 4)
        self.assertEqual(summary["ppo_environment_steps"], 8)
        self.assertEqual(summary["sampler_interactions"], 12)
        self.assertEqual(initialization["loss_space"], "executed_normalized_acceleration")
        self.assertEqual(len(episode_rows), len({row["episode"] for row in episode_rows}))
        self.assertEqual(saved_config.epochs, 2)
        self.assertEqual(snapshot.pi[0].mu_net[0].in_features, 67)

    def test_goal_bc_validation_is_explicit(self):
        with self.assertRaises(ValueError):
            PPOConfig(initialization="goal_bc", demonstration_steps=0)
        with self.assertRaises(ValueError):
            PPOConfig(
                initialization="goal_bc", demonstration_steps=2048,
                control_mode="residual", epochs=2)
        with self.assertRaises(ValueError):
            PPOConfig(demonstration_steps=1)


if __name__ == "__main__":
    unittest.main()
