"""Hidden target truth dynamics; policy observations contain estimates only."""

import numpy as np


def advance_target_state(state, config, step_count):
    """Advance one CA step, switching acceleration only at an optional turn step."""
    previous = np.asarray(state, dtype=float).copy()
    acceleration = config.target_acceleration
    if config.target_turn_step is not None and step_count >= config.target_turn_step:
        acceleration = config.target_acceleration_after_turn
    dt = config.slot_duration
    result = previous.copy()
    # Keep the legacy operation order for the default constant-acceleration path.
    result[:3] += previous[3:] * dt + .5 * acceleration * dt ** 2
    result[3:] += acceleration * dt
    return result
