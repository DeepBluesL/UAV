"""UAV 通信 SINR 公式。"""

import numpy as np

from .channels import (
    FLOAT_TINY,
    abs2_scalar,
    beta_c_bs_uav,
    beta_s_two_hop,
    steering_vector_upa,
)


def uav_comm_sinr_eq10_parts(
        u_bs,
        uav_positions,
        target,
        n_idx,
        W_comm,
        w0,
        Mx=8,
        My=8,
        beta0_c=1e-5,
        beta0_s=1e-5,
        K=10.0,
        alpha0=0.9,
        alpha1=0.9,
        sigma0=1e-11,
        rng=None,
        include_direct_sensing_beam=False
):
    """式 (10)：返回指定 UAV 的通信 SINR 及各组成项。"""
    if rng is None:
        rng = np.random.default_rng()

    N = len(uav_positions)

    if len(W_comm) != N:
        raise ValueError("len(W_comm) must equal number of UAVs.")

    if not (0 <= n_idx < N):
        raise ValueError("n_idx out of range.")

    uav_n = uav_positions[n_idx]
    all_beams = [w0] + list(W_comm)

    a_uav_n = steering_vector_upa(u_bs, uav_n, Mx=Mx, My=My)
    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)

    beta_c_n = beta_c_bs_uav(
        u_bs=u_bs,
        uav=uav_n,
        beta0_c=beta0_c,
        K=K,
        rng=rng
    )

    beta_s_target_to_n = beta_s_two_hop(
        u_bs=u_bs,
        reflector=target,
        receiver=uav_n,
        beta0_s=beta0_s,
        K=K,
        rng=rng
    )

    w_n = W_comm[n_idx]
    desired_signal = abs2_scalar(
        beta_c_n * (a_uav_n.conj().T @ w_n)
    )

    direct_comm_interference = 0.0
    for k_idx, wk in enumerate(W_comm):
        if k_idx == n_idx:
            continue

        direct_comm_interference += abs2_scalar(
            beta_c_n * (a_uav_n.conj().T @ wk)
        )

    # 式 (10) 默认不计 w0 的直达干扰；保留原有可选扩展。
    if include_direct_sensing_beam:
        direct_comm_interference += abs2_scalar(
            beta_c_n * (a_uav_n.conj().T @ w0)
        )

    target_echo_interference = 0.0
    for wk in all_beams:
        target_echo_interference += abs2_scalar(
            alpha0 * beta_s_target_to_n * (a_target.conj().T @ wk)
        )

    uav_echo_interference = 0.0
    for i_idx, uav_i in enumerate(uav_positions):
        if i_idx == n_idx:
            continue

        a_uav_i = steering_vector_upa(u_bs, uav_i, Mx=Mx, My=My)

        beta_s_uav_i_to_n = beta_s_two_hop(
            u_bs=u_bs,
            reflector=uav_i,
            receiver=uav_n,
            beta0_s=beta0_s,
            K=K,
            rng=rng
        )

        for wk in all_beams:
            uav_echo_interference += abs2_scalar(
                alpha1 * beta_s_uav_i_to_n * (a_uav_i.conj().T @ wk)
            )

    denominator = (
            direct_comm_interference
            + target_echo_interference
            + uav_echo_interference
            + sigma0
    )

    sinr = desired_signal / max(denominator, FLOAT_TINY)

    return {
        "desired_signal": desired_signal,
        "direct_comm_interference": direct_comm_interference,
        "target_echo_interference": target_echo_interference,
        "uav_echo_interference": uav_echo_interference,
        "sigma0": sigma0,
        "denominator": denominator,
        "sinr": sinr,
        "sinr_db": 10 * np.log10(max(sinr, FLOAT_TINY))
    }