"""确定性波束与有效感知源；只调用 project 包内的物理公式。"""

import numpy as np

from .channels import make_matched_beams
from .communication import uav_comm_sinr_eq10_parts
from .crb import ci_fusion_crb, crb_from_sensing_sinr, measurement_noise_cov_from_fused_crb
from .pcrb import pcrb_eq24_to_eq28_parts
from .sensing import bs_sensing_sinr_eq6_parts, uav_sensing_sinr_eq12_parts
from .tracking import TrackingEKF

EPS = 1.e-12


class ISACPhysics:
    """固定物理难度。输入形状由 EnvConfig 和环境接口约定。"""

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.num_antennas = config.antenna_x * config.antenna_y
        # 跟踪量测使用独立子流，避免active改变信道抽样次数后量测噪声错位。
        tracker_rng = (np.random.default_rng(rng.integers(0, 2 ** 63))
                       if config.sensing_mode == "ekf" else None)
        self.tracker = TrackingEKF(config, tracker_rng) if tracker_rng is not None else None

    def initialize_belief(self, target_state):
        """每回合只从真值采样一次有误差的EKF先验。"""
        if self.tracker is None:
            estimate = self.noisy_estimate(target_state)
            covariance = self.config.initial_covariance * np.eye(6)
        else:
            estimate, covariance = self.tracker.initialize(target_state)
        return estimate, covariance

    def noisy_estimate(self, target_state):
        """模拟 BS 广播消息；truth + noise 尚未和 PCRB 形成滤波闭环。"""
        cfg = self.config
        beta = cfg.imperfect_csi_beta
        if beta > 0 and cfg.csi_beta_jitter > 0:
            beta = float(np.clip(beta + self.rng.normal(0., cfg.csi_beta_jitter), 0., 1.))
        std = np.array([cfg.csi_position_std] * 3 + [cfg.csi_velocity_std] * 3)
        return target_state + self.rng.normal(0., beta * std)

    def link_metrics(self, positions, target_state, estimate, active):
        cfg = self.config
        intended = self._matched_beams(positions, estimate[:3], active)
        actual = self._perturb_beams(intended, active)
        common = dict(
            u_bs=cfg.bs_position, target=target_state[:3],
            W_comm=[actual[:, i:i + 1] for i in range(2)], w0=actual[:, 2:3],
            Mx=cfg.antenna_x, My=cfg.antenna_y, beta0_s=cfg.beta0_s,
            K=cfg.rician_k, alpha0=cfg.alpha_target, sigma0=cfg.awgn_power, rng=self.rng,
        )
        communication = np.zeros(2)
        sensing = np.zeros(3)
        sensing[0] = max(float(bs_sensing_sinr_eq6_parts(
            sigma_ubs=cfg.residual_noise, **common)["sinr"]), EPS)
        for i in np.flatnonzero(active):
            # 保留所有实体位置：退出者仍可能作为被动散射体。
            comm = uav_comm_sinr_eq10_parts(
                uav_positions=positions, n_idx=int(i), beta0_c=cfg.beta0_c,
                alpha1=cfg.alpha_uav, include_direct_sensing_beam=False, **common)
            sense = uav_sensing_sinr_eq12_parts(
                uav=positions[i], sigma_un=cfg.residual_noise, **common)
            communication[i] = max(float(comm["sinr"]), EPS)
            sensing[i + 1] = max(float(sense["sinr"]), EPS)
        return {
            "communication_sinrs": communication,
            "sensing_sinrs": sensing,
            "sensing_source_mask": np.r_[True, active],
            "intended_beams": intended,
            "actual_beams": actual,
            "power": float(np.sum(np.abs(actual) ** 2)),
        }

    def update(self, positions, target_state, estimate, active, previous_target_state, j_prev,
               source_velocities=None):
        prior = None
        if self.tracker is not None:
            prior, _ = self.tracker.predict()
            estimate = prior
        metrics = self.link_metrics(positions, target_state, estimate, active)
        cfg = self.config
        source_mask = metrics["sensing_source_mask"]
        source_crbs = np.zeros((3, 4))
        for i in np.flatnonzero(source_mask):
            # 原物理函数用 SINR**2 转 CRB；这里只保留原数值代理。
            source_crbs[i] = crb_from_sensing_sinr(
                sensing_sinr=float(metrics["sensing_sinrs"][i]), bandwidth=cfg.bandwidth,
                kappa_d=cfg.kappa_d, kappa_theta=cfg.kappa_theta,
                kappa_phi=cfg.kappa_phi, kappa_v=cfg.kappa_v)
        if self.tracker is not None:
            return self._ekf_update(metrics, positions, source_velocities, target_state,
                                    active, source_crbs, prior)
        # CI 对当前有效源重新等权；增加一个较差源未必改善结果。
        fused_crb = ci_fusion_crb(source_crbs[source_mask])
        psi = measurement_noise_cov_from_fused_crb(fused_crb)
        # 原公式以真实前态构造 CV prior，与环境 CA 真值存在模型失配。
        # 历史信息仅在 j_prev 中传播；退出源的上一步 CRB 不会再次融合。
        parts = pcrb_eq24_to_eq28_parts(
            zeta_prev=previous_target_state, J_prev=j_prev, Psi=psi,
            dt=cfg.slot_duration, sigma2_zeta=cfg.process_noise_intensity,
            use_inverse_covariance=True)
        pcrb = parts["PCRB"]
        metrics.update(
            j=parts["J"], pcrb=pcrb, rho_pos=float(np.trace(pcrb[:3, :3])),
            rho_all=float(np.trace(pcrb)), fused_crb=fused_crb, source_crbs=source_crbs,
            estimated_target_state=estimate.copy(),
            prior_estimated_target_state=estimate.copy(),
            measurement_source_mask=np.zeros(3, dtype=bool),
            measurement_counts=np.zeros(3, dtype=int), measurement_count=0,
            uncertainty_kind="proxy_pcrb")
        return metrics

    def _ekf_update(self, metrics, positions, velocities, target_state, active, source_crbs, prior):
        """先验定波束后生成量测；UAV仅在活动且通信达标时上传。"""
        # 固定抽取三源噪声，随后再mask；不同活动历史下量测噪声仍按源对齐。
        standard_noise = self.tracker.rng.standard_normal((3, 4))
        upload = np.r_[True, active & (metrics["communication_sinrs"] >= self.config.gamma_min)]
        if not self.config.tracking_use_uav_measurements:
            upload[1:] = False
        if not self.config.collect_measurements:
            upload[:] = False
        velocities = np.zeros_like(positions) if velocities is None else velocities
        estimate, covariance = self.tracker.update(
            target_state, positions, velocities, source_crbs, upload, standard_noise)
        metrics.update(
            j=np.linalg.pinv(covariance), pcrb=covariance,
            rho_pos=float(np.trace(covariance[:3, :3])), rho_all=float(np.trace(covariance)),
            fused_crb=ci_fusion_crb(source_crbs[metrics["sensing_source_mask"]]),
            source_crbs=source_crbs, estimated_target_state=estimate,
            prior_estimated_target_state=prior, measurement_source_mask=upload,
            measurement_counts=self.tracker.measurement_counts.copy(),
            measurement_count=int(self.tracker.measurement_counts.sum()),
            uncertainty_kind="ekf_covariance")
        return metrics

    def _matched_beams(self, positions, target, active):
        cfg = self.config
        w_sense, w_comm = make_matched_beams(
            u_bs=cfg.bs_position, uav_positions=positions[active], target=target,
            Mx=cfg.antenna_x, My=cfg.antenna_y, P=cfg.total_power)
        beams = np.zeros((self.num_antennas, 3), dtype=np.complex128)
        beams[:, 2] = w_sense.ravel()
        for i, beam in zip(np.flatnonzero(active), w_comm):
            beams[:, i] = beam.ravel()
        return beams

    def _perturb_beams(self, beams, active):
        valid = np.r_[active, True]
        actual = np.zeros_like(beams)
        beta = self.config.imperfect_csi_beta
        power_per_beam = self.config.total_power / valid.sum()
        # 复数扰动是 CSI/射频误差代理，不是严格 RF 硬件模型。
        for i in np.flatnonzero(valid):
            noise = (self.rng.standard_normal(self.num_antennas)
                     + 1j * self.rng.standard_normal(self.num_antennas)) / np.sqrt(2.)
            noise /= np.linalg.norm(noise)
            direction = (
                np.sqrt(1. - beta ** 2) * beams[:, i] / np.linalg.norm(beams[:, i])
                + beta * noise)
            actual[:, i] = direction / np.linalg.norm(direction) * np.sqrt(power_per_beam)
        # 有效列各自等功率，非活动列保持严格为零。
        actual *= np.sqrt(self.config.total_power / np.sum(np.abs(actual) ** 2))
        return actual
