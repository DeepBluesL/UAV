"""Validation kept separate so configuration declarations stay compact."""


def validate_ppo(config):
    if min(config.steps_per_epoch, config.epochs,
           config.train_pi_iters, config.train_v_iters) < 1:
        raise ValueError("Rollout size, epochs and update counts must be positive")
    if not (0 <= config.gamma <= 1 and 0 <= config.lam <= 1):
        raise ValueError("gamma and lam must be in [0, 1]")
    if config.control_mode not in {"pure", "residual"}:
        raise ValueError("control_mode must be 'pure' or 'residual'")
    if not 0 <= config.residual_scale <= 1:
        raise ValueError("residual_scale must be in [0, 1]")
    distributions = {"fixed", "randomized", "domain_randomized", "curriculum"}
    if config.training_distribution not in distributions:
        raise ValueError(f"training_distribution must be one of {sorted(distributions)}")
    if config.initialization not in {"random", "goal_bc"}:
        raise ValueError("initialization must be 'random' or 'goal_bc'")
    if min(config.bc_epochs, config.bc_batch_size) < 1 or config.bc_lr <= 0:
        raise ValueError("Behavioral-cloning settings must be positive")
    invalid_demo = config.demonstration_steps < 0 or (
        config.demonstration_steps % config.steps_per_epoch)
    if invalid_demo:
        raise ValueError("demonstration_steps must be a nonnegative epoch multiple")
    total_steps = config.epochs * config.steps_per_epoch
    if config.demonstration_steps >= total_steps and config.demonstration_steps:
        raise ValueError("demonstration_steps must be below the total interaction budget")
    if (config.initialization == "goal_bc"
            and (config.control_mode != "pure" or not config.demonstration_steps)):
        raise ValueError("goal_bc requires pure control and demonstration_steps > 0")
    if config.initialization == "random" and config.demonstration_steps:
        raise ValueError("demonstration_steps requires initialization='goal_bc'")
    config.checkpoint_epochs = tuple(int(epoch) for epoch in config.checkpoint_epochs)
    valid_checkpoints = tuple(sorted(set(config.checkpoint_epochs)))
    if (valid_checkpoints != config.checkpoint_epochs
            or any(epoch < 1 or epoch > config.epochs for epoch in config.checkpoint_epochs)):
        raise ValueError("checkpoint_epochs must be unique, sorted, and within training epochs")
    if any(epoch * config.steps_per_epoch <= config.demonstration_steps
           for epoch in config.checkpoint_epochs):
        raise ValueError("checkpoint_epochs must occur after behavioral cloning")
