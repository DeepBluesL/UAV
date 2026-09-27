"""Training rollout collection, kept separate from the command-line entry point."""

from .metrics import EpisodeRecorder


def collect_epoch(env, ac, buffer, obs, state, recorder, scenario_sampler=None):
    """Collect one batch without treating an epoch boundary as episode end."""
    episodes, components, reward_sum = [], {}, 0.
    for step in range(buffer.max_size):
        active_before = env.active.copy()
        if hasattr(ac, "step"):
            raw_actions, logp = ac.step(obs, active_before)
            actions = (ac.action_adapter.policy_action(raw_actions, obs, active_before)
                       if ac.action_adapter is not None else raw_actions)
        else:
            raw_actions, logp = ac.act(obs, active_before)
            actions = raw_actions
        value = ac.value(state)
        next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
        # PPO must score the original Gaussian sample, not the adapted env action.
        buffer.store(obs, state, raw_actions, reward, value, logp, active_before)
        recorder.add(reward, info)
        reward_sum += reward
        for key, contribution in info["reward_parts"].items():
            components[key] = components.get(key, 0.) + contribution

        obs, state = next_obs, next_state
        cutoff = step == buffer.max_size - 1
        if terminated or truncated or cutoff:
            buffer.finish_path(0. if terminated else ac.value(next_state))
        if terminated or truncated:
            if terminated:
                row = recorder.summary()
                if scenario_sampler is not None:
                    row["training_scenario"] = scenario_sampler.current_kind
                episodes.append(row)
            if scenario_sampler is None:
                obs, state, initial_info = env.reset()
            else:
                obs, state, initial_info = scenario_sampler.reset_env(env)
                if hasattr(ac, "set_environment"):
                    ac.set_environment(env.config)
            recorder = EpisodeRecorder(initial_info)
    stats = {"mean_step_reward": reward_sum / buffer.max_size}
    stats.update({f"mean_reward_{key}": value / buffer.max_size
                  for key, value in components.items()})
    return obs, state, recorder, episodes, stats
