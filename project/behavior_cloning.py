"""Goal-controller demonstrations for pure-policy initialization."""

import numpy as np
import torch
from torch.optim import Adam

from .control import executable_acceleration, goal_raw_action


def _teacher_action(obs, active, config):
    """Return the raw action matching the teacher's executed, norm-clipped acceleration."""
    requested = goal_raw_action(obs, active, config)
    acceleration = executable_acceleration(requested, config)
    scaled = np.clip(
        acceleration / config.max_uav_acceleration, -1.0 + 1e-7, 1.0 - 1e-7)
    action = np.arctanh(scaled).astype(np.float32)
    action[~np.asarray(active, dtype=bool)] = 0.0
    return action


def executed_normalized_action(raw_action):
    """Differentiable environment action: tanh, vector clip, divided by max acceleration."""
    normalized = torch.tanh(raw_action)
    norms = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
    return normalized / torch.clamp(norms, min=1.0)


def _execution_loss(actor, observation, action):
    predicted = executed_normalized_action(actor.mu_net(observation))
    target = executed_normalized_action(action)
    return torch.mean((predicted - target) ** 2)


def collect_goal_demonstrations(env, sampler, steps):
    """Collect exactly ``steps`` environment interactions; retain active actor rows."""
    observations = [[], []]
    actions = [[], []]
    obs, _, _ = env.reset()
    for _ in range(steps):
        active = env.active.copy()
        teacher = _teacher_action(obs, active, env.config)
        for agent in range(2):
            if active[agent]:
                observations[agent].append(obs[agent].copy())
                actions[agent].append(teacher[agent].copy())
        obs, _, _, terminated, truncated, _ = env.step(teacher)
        if hasattr(sampler, "advance"):
            sampler.advance(1)
        if terminated or truncated:
            obs, _, _ = sampler.reset_env(env)
    return tuple(
        (np.asarray(observations[i], dtype=np.float32),
         np.asarray(actions[i], dtype=np.float32))
        for i in range(2)
    )


def _mean_loss(actor, observation, action, device):
    if not len(observation):
        return 0.0
    with torch.no_grad():
        obs = torch.as_tensor(observation, device=device)
        target = torch.as_tensor(action, device=device)
        return float(_execution_loss(actor, obs, target).cpu())


def fit_goal_behavior(ac, demonstrations, config):
    """Fit executed acceleration; leave critic and actor log standard deviations unchanged."""
    device = torch.device(config.device)
    ac.to(device).train()
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 0x4243)
    stats = {}
    for agent, (observation, action) in enumerate(demonstrations):
        actor = ac.pi[agent]
        stats[f"bc_samples_{agent}"] = len(observation)
        stats[f"bc_loss_before_{agent}"] = _mean_loss(
            actor, observation, action, device)
        optimizer = Adam(actor.mu_net.parameters(), lr=config.bc_lr)
        count = len(observation)
        for _ in range(config.bc_epochs):
            order = torch.randperm(count, generator=generator).numpy()
            for start in range(0, count, config.bc_batch_size):
                indices = order[start:start + config.bc_batch_size]
                obs = torch.as_tensor(observation[indices], device=device)
                target = torch.as_tensor(action[indices], device=device)
                loss = _execution_loss(actor, obs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        stats[f"bc_loss_after_{agent}"] = _mean_loss(
            actor, observation, action, device)
    return stats
