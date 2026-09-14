"""式 (14)～(16)、(23)：SINR 到 CRB、协方差交集及测量噪声矩阵。"""

import numpy as np


def crb_from_sensing_sinr(
    sensing_sinr, bandwidth=100e6, kappa_d=1.0,
    kappa_theta=1e-6, kappa_phi=1e-6, kappa_v=1e-6,
):
    """输出 [距离、极角、方位角、径向速度] 方差代理；原式使用 SINR**2。"""
    sinr_abs2 = float(np.abs(sensing_sinr) ** 2)
    if not np.isfinite(sinr_abs2) or sinr_abs2 <= 0.0:
        raise ValueError("sensing_sinr must be finite and non-zero.")
    sigma2_d = kappa_d / (sinr_abs2 * bandwidth)
    sigma2_theta = kappa_theta / sinr_abs2
    sigma2_phi = kappa_phi / sinr_abs2
    sigma2_v = kappa_v / sinr_abs2
    return np.array([sigma2_d, sigma2_theta, sigma2_phi, sigma2_v], dtype=float)


def ci_fusion_crb(crb_list, weights=None):
    """逐分量 CI：1 / sum(weights / crb)，输入形状 [有效源数,4]。

    默认对当前源等权；加入较差的源并不保证融合方差下降。
    """
    crb_array = np.asarray(crb_list, dtype=float)
    if crb_array.ndim != 2 or crb_array.shape[1] != 4:
        raise ValueError("crb_list must have shape (num_nodes, 4,).")
    num_nodes = crb_array.shape[0]
    if weights is None:
        weights = np.ones(num_nodes) / num_nodes
    else:
        weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != num_nodes:
        raise ValueError("weights must equal number of CRB vectors.")
    if np.any(weights < 0):
        raise ValueError("CI weights must be non-negative.")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-8):
        raise ValueError("CI weights must sum to 1.")
    if np.any(~np.isfinite(crb_array)) or np.any(crb_array <= 0.0):
        raise ValueError("All CRB entries must be finite and positive.")
    information_array = 1.0 / crb_array
    fused_information = np.sum(weights[:, None] * information_array, axis=0)
    return 1.0 / fused_information


def measurement_noise_cov_from_fused_crb(fused_crb):
    """式 (23)：将四项融合方差构成 [4,4] 对角测量噪声矩阵 Psi。"""
    fused_crb = np.asarray(fused_crb, dtype=float)
    if fused_crb.shape != (4,):
        raise ValueError("fused_crb must have shape (4,).")
    return np.diag(fused_crb)