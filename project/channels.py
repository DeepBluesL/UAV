"""ISAC 信道、阵列方向与匹配波束基础函数。"""

import numpy as np


DIST_EPS = 1e-12
FLOAT_TINY = np.finfo(float).tiny


def norm(p, q):
    """计算两个三维点的欧氏距离。"""
    return np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))


def db_to_linear(db):
    """将 dB 信道增益转换为线性值。"""
    return 10 ** (db / 10)


def dbm_to_watt(dbm):
    """将 dBm 功率转换为瓦特。"""
    return 10 ** ((dbm - 30) / 10)


def abs2(x):
    """返回复数信号的功率。"""
    return np.abs(x) ** 2


def abs2_scalar(x):
    arr = np.asarray(x)
    return float(np.abs(arr.reshape(-1)[0]) ** 2)


def normalize(v):
    v = np.asarray(v)
    v_norm = np.linalg.norm(v)
    if v_norm <= DIST_EPS:
        raise ValueError("Cannot normalize a zero-length vector.")
    return v / v_norm


def rician_factor(K=10, rng=None):
    """生成一个莱斯衰落复系数。"""
    if rng is None:
        rng = np.random.default_rng()

    g = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    return np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * g


def beta_c_bs_uav(u_bs, uav, beta0_c, K=10.0, rng=None):
    """式 (8)：BS 到 UAV 的通信信道幅度增益。"""
    d = norm(u_bs, uav)
    if d <= DIST_EPS:
        raise ValueError("BS and UAV positions must be different.")
    h = rician_factor(K=K, rng=rng)
    return h * np.sqrt(beta0_c / d ** 2)


def beta_s_bs_target(u_bs, target, beta0_s=1e-5, K=10, rng=None):
    """BS 经目标反射回 BS 的感知信道幅度增益。"""
    d = norm(u_bs, target)
    if d <= DIST_EPS:
        raise ValueError("BS and target positions must be different.")
    h = rician_factor(K, rng)
    return h * np.sqrt(beta0_s / (d ** 4))


def beta_s_two_hop(u_bs, reflector, receiver, beta0_s=1e-5, K=10.0, rng=None):
    """BS 经反射体到接收端的双跳感知信道。"""
    d1 = norm(u_bs, reflector)
    d2 = norm(receiver, reflector)
    if d1 <= DIST_EPS or d2 <= DIST_EPS:
        raise ValueError("Two-hop sensing path contains a zero-length hop.")
    h = rician_factor(K=K, rng=rng)
    return h * np.sqrt(beta0_s / (d1 ** 2 * d2 ** 2))


def direction_cosines(tx_pos, rx_pos):
    """计算发射端指向接收端的 x/y 方向余弦。"""
    tx_pos = np.asarray(tx_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)

    diff = rx_pos - tx_pos
    d = np.linalg.norm(diff)

    if d < DIST_EPS:
        raise ValueError("tx_pos and rx_pos are identical or too close.")

    psi_x = diff[0] / d
    psi_y = diff[1] / d

    return psi_x, psi_y


def steering_vector_upa(
        tx_pos, rx_pos, Mx=8, My=8, dx=None, dy=None, wavelength=1.0
):
    """式 (4)：返回形状为 ``(Mx*My, 1)`` 的 UPA 导向向量。"""
    if dx is None:
        dx = wavelength / 2
    if dy is None:
        dy = wavelength / 2

    psi_x, psi_y = direction_cosines(tx_pos, rx_pos)

    mx = np.arange(Mx)
    my = np.arange(My)

    # 式 (4) 给出负相位的 a^H，因此列向量 a 使用正相位。
    ax = np.exp(1j * 2 * np.pi * dx * mx * psi_x / wavelength)
    ay = np.exp(1j * 2 * np.pi * dy * my * psi_y / wavelength)

    a = np.kron(ax, ay)
    return a.reshape(-1, 1)


def make_matched_beams(u_bs, uav_positions, target, Mx=8, My=8, P=5.0):
    """生成目标感知波束和各 UAV 通信匹配波束，并均分总功率。"""
    N = len(uav_positions)
    power_per_beam = P / (N + 1)

    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)
    w0 = normalize(a_target) * np.sqrt(power_per_beam)

    W_comm = []
    for uav in uav_positions:
        a_uav = steering_vector_upa(u_bs, uav, Mx=Mx, My=My)
        wk = normalize(a_uav) * np.sqrt(power_per_beam)
        W_comm.append(wk)

    total_power = np.linalg.norm(w0) ** 2
    total_power += sum(np.linalg.norm(wk) ** 2 for wk in W_comm)

    if not np.isclose(total_power, P, atol=1e-8):
        raise RuntimeError(f"Power mismatch: total_power = {total_power}, P = {P}")

    return w0, W_comm