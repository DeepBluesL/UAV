"""保存配置、日志与策略；checkpoint 只含张量和普通 Python 数据。"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig, PPOConfig, RewardConfig
from .core import MAPPOActorCritic
from .observations import OBS_DIM, STATE_DIM


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
    torch.save({
        "model": ac.state_dict(),
        "config": config_dict(environment, reward, ppo),
        "obs_dim": OBS_DIM, "state_dim": STATE_DIM,
    }, path)


def load_policy(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = saved["config"]
    environment = EnvConfig(**config["environment"])
    reward = RewardConfig(**config["reward"])
    ppo = PPOConfig(**{**config["ppo"], "device": device})
    ac = MAPPOActorCritic(saved["obs_dim"], saved["state_dim"], hidden_sizes=ppo.hidden_sizes)
    ac.load_state_dict(saved["model"])
    ac.to(device).eval()
    return ac, environment, reward, ppo