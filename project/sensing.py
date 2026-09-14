"""BS 与 UAV 感知 SINR 公式。"""

import numpy as np

from .channels import (
    FLOAT_TINY,
    abs2_scalar,
    beta_s_bs_target,
    beta_s_two_hop,
    steering_vector_upa,
)


def bs_sensing_sinr_eq6_parts(
        u_bs,
        target,
        W_comm,
        w0,
        Mx=8,
        My=8,
        beta0_s=1e-5,
        K=10.0,
        alpha0=0.9,
        sigma_ubs=1e-10,
        sigma0=1e-11,
        rng=None
):
    """式 (6)：返回 BS 感知 SINR 及各组成项。"""
    if rng is None:
        rng = np.random.default_rng()

    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)

    beta_bs_target = beta_s_bs_target(
        u_bs=u_bs,
        target=target,
        beta0_s=beta0_s,
        K=K,
        rng=rng
    )

    signal = abs2_scalar(
        alpha0 * beta_bs_target * (a_target.conj().T @ w0)
    )

    comm_beam_interference = 0.0
    for wk in W_comm:
        comm_beam_interference += abs2_scalar(
            alpha0 * beta_bs_target * (a_target.conj().T @ wk)
        )

    denominator = comm_beam_interference + sigma_ubs + sigma0
    sinr = signal / max(denominator, FLOAT_TINY)

    return {
        "signal": signal,
        "comm_beam_interference": comm_beam_interference,
        "sigma_ubs": sigma_ubs,
        "sigma0": sigma0,
        "denominator": denominator,
        "sinr": sinr,
        "sinr_db": 10 * np.log10(max(sinr, FLOAT_TINY))
    }


def uav_sensing_sinr_eq12_parts(
        u_bs,
        uav,
        target,
        W_comm,
        w0,
        Mx=8,
        My=8,
        beta0_s=1e-5,
        K=10.0,
        alpha0=0.9,
        sigma_un=1e-10,
        sigma0=1e-11,
        rng=None
):
    """式 (12)：返回 UAV 感知 SINR 及各组成项。"""
    if rng is None:
        rng = np.random.default_rng()

    # 感知波束指向目标，因此此处使用目标导向向量。
    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)

    beta_s_target_to_uav = beta_s_two_hop(
        u_bs=u_bs,
        reflector=target,
        receiver=uav,
        beta0_s=beta0_s,
        K=K,
        rng=rng
    )

    signal = abs2_scalar(
        alpha0 * beta_s_target_to_uav * (a_target.conj().T @ w0)
    )

    comm_beam_interference = 0.0
    for wk in W_comm:
        comm_beam_interference += abs2_scalar(
            alpha0 * beta_s_target_to_uav * (a_target.conj().T @ wk)
        )

    denominator = comm_beam_interference + sigma_un + sigma0
    sinr = signal / max(denominator, FLOAT_TINY)

    return {
        "signal": signal,
        "comm_beam_interference": comm_beam_interference,
        "sigma_un": sigma_un,
        "sigma0": sigma0,
        "denominator": denominator,
        "sinr": sinr,
        "sinr_db": 10 * np.log10(max(sinr, FLOAT_TINY))
    }