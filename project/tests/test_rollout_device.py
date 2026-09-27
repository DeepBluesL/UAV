"""CPU rollout and CUDA update preserve PPO's raw-action information flow."""

import io
import unittest

import numpy as np
import torch

from project.config import EnvConfig, PPOConfig, RewardConfig
from project.core import MAPPOActorCritic
from project.env import DualUAVEnv
from project.metrics import EpisodeRecorder
from project.ppo import MAPPOBuffer, PPOTrainer
from project.rollout import collect_epoch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device alternation")
class RolloutDeviceTests(unittest.TestCase):
    def test_two_cpu_rollouts_and_cuda_updates_keep_optimizer_and_logprob(self):
        torch.manual_seed(31)
        environment, reward = EnvConfig(), RewardConfig()
        config = PPOConfig(
            hidden_sizes=(8,), steps_per_epoch=8, device="cuda",
            rollout_device="cpu", train_pi_iters=1, train_v_iters=1)
        ac = MAPPOActorCritic(
            31, 58, hidden_sizes=config.hidden_sizes, environment=environment)
        trainer = PPOTrainer(ac, config)
        parameter_ids = [id(parameter) for parameter in ac.parameters()]
        env = DualUAVEnv(environment, reward, seed=31)
        obs, state, info = env.reset(seed=31)
        recorder = EpisodeRecorder(info)

        for _ in range(2):
            ac.to("cpu")
            buffer = MAPPOBuffer(31, 58, 3, 8, config.gamma, config.lam)
            obs, state, recorder, _, _ = collect_epoch(
                env, ac, buffer, obs, state, recorder)
            data = buffer.get()
            self.assertTrue(torch.isfinite(data["act"][data["active"]]).all())
            ac.to("cuda")
            with torch.no_grad():
                mask = data["active"][:, 0]
                distribution = ac.pi[0]._distribution(data["obs"][mask, 0].cuda())
                logp = ac.pi[0]._log_prob_from_distribution(
                    distribution, data["act"][mask, 0].cuda()).cpu()
            torch.testing.assert_close(logp, data["logp"][mask, 0], atol=2e-6, rtol=0)
            before = [parameter.detach().cpu().clone() for parameter in ac.parameters()]
            trainer.update(data)
            self.assertTrue(any(
                not torch.equal(old, new.detach().cpu())
                for old, new in zip(before, ac.parameters())))
            self.assertEqual(parameter_ids, [id(parameter) for parameter in ac.parameters()])

        momentum_devices = {
            value.device.type
            for optimizer in trainer.pi_optimizers + [trainer.vf_optimizer]
            for state_values in optimizer.state.values()
            for name, value in state_values.items()
            if name in {"exp_avg", "exp_avg_sq"}
        }
        self.assertEqual(momentum_devices, {"cuda"})
        stream = io.BytesIO()
        torch.save(ac.state_dict(), stream)
        stream.seek(0)
        clone = MAPPOActorCritic(31, 58, hidden_sizes=(8,), environment=environment)
        clone.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
        for expected, actual in zip(ac.state_dict().values(), clone.state_dict().values()):
            torch.testing.assert_close(expected.cpu(), actual)


if __name__ == "__main__":
    unittest.main()
