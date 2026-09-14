"""式 (13)、(17)、(21)～(22)：状态、球坐标测量及数值 Jacobian。"""

import numpy as np

from .channels import DIST_EPS


def make_measurement_vector(d, theta, phi, radial_velocity):
    """测量顺序 [距离 m, 极角 rad, 方位角 rad, 径向速度 m/s]。"""
    return np.array([d, theta, phi, radial_velocity])


def make_cartesian_state(x, y, z, vx, vy, vz):
    """目标状态顺序固定为 [x, y, z, vx, vy, vz]。"""
    return np.array([x, y, z, vx, vy, vz], dtype=float)


def cartesian_state_to_measurement(zeta):
    """原式 (13) 的测量转换；保留原始 arccos 行为。"""
    x, y, z, vx, vy, vz = zeta
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if d <= DIST_EPS:
        raise ValueError("Target position cannot coincide with the BS.")
    theta = np.arccos(z / d)
    phi = np.arctan2(y, x)
    radial_velocity = (vx * x + vy * y + vz * z) / d
    return np.array([d, theta, phi, radial_velocity])


def measurement_function_h(zeta):
    """式 (21)～(22) 的 h(zeta)；与原实现一致，对 arccos 输入做截断。"""
    x, y, z, vx, vy, vz = zeta
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if d <= DIST_EPS:
        raise ValueError("Target position cannot coincide with the BS.")
    theta = np.arccos(np.clip(z / d, -1.0, 1.0))
    phi = np.arctan2(y, x)
    radial_velocity = (vx * x + vy * y + vz * z) / d
    return np.array([d, theta, phi, radial_velocity], dtype=float)


def numerical_jacobian(func, x, eps=1e-5):
    """球坐标测量的中心差分 Jacobian，典型形状为 [4,6]。

    第 2 行按方位角处理跨 ±pi 的差分，不是任意输出函数的通用 Jacobian。
    """
    x = np.asarray(x, dtype=float)
    y0 = func(x)
    H = np.zeros((len(y0), len(x)), dtype=float)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        yp, ym = func(xp), func(xm)
        diff = yp - ym
        diff[2] = (diff[2] + np.pi) % (2 * np.pi) - np.pi
        H[:, i] = diff / (2 * eps)
    return H