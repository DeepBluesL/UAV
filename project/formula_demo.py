"""原物理公式演示的独立入口：python -m project.formula_demo。"""

import numpy as np

from .channels import make_matched_beams
from .communication import uav_comm_sinr_eq10_parts
from .config import EnvConfig
from .crb import ci_fusion_crb, crb_from_sensing_sinr, measurement_noise_cov_from_fused_crb
from .measurements import cartesian_state_to_measurement
from .pcrb import pcrb_eq24_to_eq28_parts, safe_inverse
from .sensing import bs_sensing_sinr_eq6_parts, uav_sensing_sinr_eq12_parts


def run_demo(seed=42):
    """保持原演示 BS 感知→两路通信→两路 UAV 感知的随机数调用顺序。"""
    cfg = EnvConfig()
    rng = np.random.default_rng(seed)
    w0, w_comm = make_matched_beams(
        cfg.bs_position, cfg.uav_initial, cfg.target_initial_state[:3],
        Mx=cfg.antenna_x, My=cfg.antenna_y, P=cfg.total_power)
    common = dict(
        u_bs=cfg.bs_position, target=cfg.target_initial_state[:3], W_comm=w_comm, w0=w0,
        Mx=cfg.antenna_x, My=cfg.antenna_y, beta0_s=cfg.beta0_s, K=cfg.rician_k,
        alpha0=cfg.alpha_target, sigma0=cfg.awgn_power, rng=rng)
    bs_sensing = bs_sensing_sinr_eq6_parts(sigma_ubs=cfg.residual_noise, **common)
    communication = [
        uav_comm_sinr_eq10_parts(
            uav_positions=cfg.uav_initial, n_idx=i, beta0_c=cfg.beta0_c,
            alpha1=cfg.alpha_uav, include_direct_sensing_beam=False, **common)
        for i in range(2)
    ]
    uav_sensing = [
        uav_sensing_sinr_eq12_parts(uav=position, sigma_un=cfg.residual_noise, **common)
        for position in cfg.uav_initial
    ]
    crbs = [
        crb_from_sensing_sinr(
            source["sinr"], bandwidth=cfg.bandwidth, kappa_d=cfg.kappa_d,
            kappa_theta=cfg.kappa_theta, kappa_phi=cfg.kappa_phi, kappa_v=cfg.kappa_v)
        for source in [bs_sensing, *uav_sensing]
    ]
    fused_crb = ci_fusion_crb(crbs, weights=np.ones(len(crbs)) / len(crbs))
    psi = measurement_noise_cov_from_fused_crb(fused_crb)
    pcrb = pcrb_eq24_to_eq28_parts(
        cfg.target_initial_state, safe_inverse(np.eye(6)), psi,
        dt=cfg.slot_duration, sigma2_zeta=cfg.process_noise_intensity,
        use_inverse_covariance=True)
    return {
        "beams": np.hstack([*w_comm, w0]), "bs_sensing": bs_sensing,
        "communication": communication, "uav_sensing": uav_sensing,
        "measurement": cartesian_state_to_measurement(cfg.target_initial_state),
        "crbs": np.asarray(crbs), "fused_crb": fused_crb, "pcrb": pcrb,
    }


def print_parts(title, parts):
    """逐项打印公式的中间结果，便于单独检查每项干扰或方差。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, value in parts.items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.6e}")
        else:
            print(f"{key:32s}: {value}")


def main():
    result = run_demo()
    print_parts("Matched beams", {
        "shape": result["beams"].shape, "power": float(np.sum(np.abs(result["beams"]) ** 2))})
    print_parts("[Eq. 6] BS sensing", result["bs_sensing"])
    for i in range(2):
        print_parts(f"[Eq. 10] UAV {i} communication", result["communication"][i])
        print_parts(f"[Eq. 12] UAV {i} sensing", result["uav_sensing"][i])
    print_parts("[Eq. 13-16] Measurement and CRB fusion", {
        "measurement": result["measurement"], "source_crbs": result["crbs"], "fused_crb": result["fused_crb"]})
    print_parts("[Eq. 24-28] PCRB", {
        "zeta_pred": result["pcrb"]["zeta_pred"], "PCRB": result["pcrb"]["PCRB"],
        "rho_all": result["pcrb"]["rho"], "rho_pos": float(np.trace(result["pcrb"]["PCRB"][:3, :3]))})


if __name__ == "__main__":
    main()