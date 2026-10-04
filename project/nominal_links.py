"""Deterministic ratios of expected link powers for planning forecasts."""

import numpy as np

from .channels import make_matched_beams, steering_vector_upa


TINY = np.finfo(float).tiny


def _power(vector, beam):
    return float(np.abs(vector.conj().T @ beam).reshape(-1)[0] ** 2)


def nominal_link_metrics(positions, target, active, config):
    """Return ratios of expected powers, not mathematical expectations of SINR.

    Small-scale Rician power is replaced by ``E[|h|^2] = 1``. Beams use the
    public forecast positions and target mean and contain no sampled CSI error.
    """
    positions = np.asarray(positions, dtype=float)
    target = np.asarray(target, dtype=float)
    active = np.asarray(active, dtype=bool)
    cfg = config
    sense, comm_active = make_matched_beams(
        cfg.bs_position, positions[active], target,
        Mx=cfg.antenna_x, My=cfg.antenna_y, P=cfg.total_power)
    comm = [np.zeros_like(sense) for _ in range(2)]
    for index, beam in zip(np.flatnonzero(active), comm_active):
        comm[index] = beam
    beams = [sense, *comm]
    target_vector = steering_vector_upa(
        cfg.bs_position, target, Mx=cfg.antenna_x, My=cfg.antenna_y)
    target_distance = max(np.linalg.norm(target - cfg.bs_position), 1e-12)

    sensing = np.zeros(3)
    bs_gain = cfg.beta0_s / target_distance ** 4
    bs_signal = cfg.alpha_target ** 2 * bs_gain * _power(target_vector, sense)
    bs_interference = sum(
        cfg.alpha_target ** 2 * bs_gain * _power(target_vector, beam)
        for beam in comm)
    sensing[0] = bs_signal / max(bs_interference + cfg.residual_noise + cfg.awgn_power,
                                 TINY)

    communication = np.zeros(2)
    for index in np.flatnonzero(active):
        uav = positions[index]
        uav_vector = steering_vector_upa(
            cfg.bs_position, uav, Mx=cfg.antenna_x, My=cfg.antenna_y)
        direct_gain = cfg.beta0_c / max(np.linalg.norm(uav - cfg.bs_position), 1e-12) ** 2
        desired = direct_gain * _power(uav_vector, comm[index])
        direct_interference = sum(
            direct_gain * _power(uav_vector, comm[other])
            for other in range(2) if other != index)
        target_gain = cfg.beta0_s / (
            target_distance ** 2 * max(np.linalg.norm(uav - target), 1e-12) ** 2)
        target_echo = sum(
            cfg.alpha_target ** 2 * target_gain * _power(target_vector, beam)
            for beam in beams)
        uav_echo = 0.0
        for other in range(2):
            if other == index:
                continue
            reflector = positions[other]
            reflector_vector = steering_vector_upa(
                cfg.bs_position, reflector, Mx=cfg.antenna_x, My=cfg.antenna_y)
            echo_gain = cfg.beta0_s / (
                max(np.linalg.norm(reflector - cfg.bs_position), 1e-12) ** 2
                * max(np.linalg.norm(uav - reflector), 1e-12) ** 2)
            uav_echo += sum(
                cfg.alpha_uav ** 2 * echo_gain * _power(reflector_vector, beam)
                for beam in beams)
        communication[index] = desired / max(
            direct_interference + target_echo + uav_echo + cfg.awgn_power, TINY)
        sense_signal = cfg.alpha_target ** 2 * target_gain * _power(target_vector, sense)
        sense_interference = sum(
            cfg.alpha_target ** 2 * target_gain * _power(target_vector, beam)
            for beam in comm)
        sensing[index + 1] = sense_signal / max(
            sense_interference + cfg.residual_noise + cfg.awgn_power, TINY)
    return communication, np.maximum(sensing, TINY)
