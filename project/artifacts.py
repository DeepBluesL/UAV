"""保存配置、日志与策略；checkpoint 只含张量和普通 Python 数据。"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig, PPOConfig, RewardConfig
from .core import MAPPOActorCritic
from .observation_specs import observation_spec


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=json_value, allow_nan=False),
        encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_dict(environment, reward, ppo):
    # 转为普通 list，保证 torch.load(weights_only=True) 可以读回配置。
    return json.loads(json.dumps(
        {"environment": asdict(environment), "reward": asdict(reward), "ppo": asdict(ppo)},
        default=json_value))


def save_policy(path, ac, environment, reward, ppo):
    obs_dim = ac.pi[0].mu_net[0].in_features
    state_dim = ac.v.v_net[0].in_features
    spec = observation_spec(environment)
    if (obs_dim, state_dim) != (spec.obs_dim, spec.state_dim):
        raise ValueError("Policy dimensions do not match the environment observation schema")
    torch.save({
        "model": ac.state_dict(),
        "config": config_dict(environment, reward, ppo),
        "obs_dim": obs_dim, "state_dim": state_dim,
        "observation_version": spec.version,
    }, path)


def load_policy(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = saved["config"]
    environment = EnvConfig(**config["environment"])
    reward = RewardConfig(**config["reward"])
    ppo = PPOConfig(**{**config["ppo"], "device": device})
    spec = observation_spec(environment)
    saved_version = saved.get("observation_version", "v1")
    if saved_version != spec.version:
        raise ValueError("Checkpoint observation version disagrees with its config")
    if (saved["obs_dim"], saved["state_dim"]) != (spec.obs_dim, spec.state_dim):
        raise ValueError("Checkpoint dimensions disagree with its observation schema")
    ac = MAPPOActorCritic(
        saved["obs_dim"], saved["state_dim"], hidden_sizes=ppo.hidden_sizes,
        environment=environment, control_mode=ppo.control_mode,
        residual_scale=ppo.residual_scale)
    ac.load_state_dict(saved["model"])
    ac.to(device).eval()
    return ac, environment, reward, ppo
