"""式 (18)～(28)：匀速预测、信息矩阵和后验 CRB 递推。"""

import numpy as np

from .measurements import measurement_function_h, numerical_jacobian


def state_transition_matrix(dt=1.0):
    """式 (19)：[x,y,z,vx,vy,vz] 的匀速状态转移矩阵。"""
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    return np.block([[I3, dt * I3], [Z3, I3]])


def process_noise_covariance(dt=1.0, sigma2_zeta=0.1):
    """式 (20)：连续白噪声加速度模型对应的离散过程协方差。"""
    I3 = np.eye(3)
    Q_pos_pos = (dt ** 3 / 3.0) * sigma2_zeta * I3
    Q_pos_vel = (dt ** 2 / 2.0) * sigma2_zeta * I3
    Q_vel_pos = (dt ** 2 / 2.0) * sigma2_zeta * I3
    Q_vel_vel = dt * sigma2_zeta * I3
    return np.block([[Q_pos_pos, Q_pos_vel], [Q_vel_pos, Q_vel_vel]])


def predict_cartesian_state(zeta_prev, dt=1.0):
    """式 (18) 的确定性预测；过程噪声在协方差递推中体现。"""
    F = state_transition_matrix(dt)
    return F @ zeta_prev


def symmetrize(A):
    return 0.5 * (A + A.T)


def safe_inverse(A):
    """正常求逆，奇异时用伪逆；不添加会淹没小角度方差的固定 jitter。"""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if np.any(~np.isfinite(A)):
        raise ValueError("A must contain only finite values.")
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def prior_fim_eq25(J_prev, F, Phi):
    """式 (25)：J_prior = inv(Phi + F @ inv(J_prev) @ F.T)。"""
    P_prev = safe_inverse(J_prev)
    P_prior = Phi + F @ P_prev @ F.T
    J_prior = safe_inverse(P_prior)
    return symmetrize(J_prior), symmetrize(P_prior)


def data_fim_eq26(H, Psi, use_inverse_covariance=True):
    """式 (26)：默认 H.T @ inv(Psi) @ H；False 保留原公式对照选项。"""
    if use_inverse_covariance:
        Psi_inv = safe_inverse(Psi)
        J_data = H.T @ Psi_inv @ H
    else:
        J_data = H.T @ Psi @ H
    return symmetrize(J_data)


def pcrb_eq24_to_eq28_parts(
    zeta_prev, J_prev, Psi, dt=1.0, sigma2_zeta=0.1, use_inverse_covariance=True,
):
    """输入前态 [6]、前信息矩阵 [6,6]、测量噪声 [4,4]，返回所有中间量。

    此函数不生成测量或更新状态估计；rho 为六维 trace，位置指标应另取 [:3,:3]。
    """
    F = state_transition_matrix(dt)
    Phi = process_noise_covariance(dt=dt, sigma2_zeta=sigma2_zeta)
    zeta_pred = F @ zeta_prev
    y_pred = measurement_function_h(zeta_pred)
    H = numerical_jacobian(measurement_function_h, zeta_pred)
    J_prior, P_prior = prior_fim_eq25(J_prev=J_prev, F=F, Phi=Phi)
    J_data = data_fim_eq26(H=H, Psi=Psi, use_inverse_covariance=use_inverse_covariance)
    J = symmetrize(J_prior + J_data)
    PCRB = symmetrize(safe_inverse(J))
    rho = float(np.trace(PCRB))
    return {
        "F": F, "Phi": Phi, "zeta_pred": zeta_pred, "y_pred": y_pred,
        "H": H, "Psi": Psi, "P_prior": P_prior, "J_prior": J_prior,
        "J_data": J_data, "J": J, "PCRB": PCRB, "rho": rho,
    }