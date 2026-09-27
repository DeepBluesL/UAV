import argparse
import os
import sys

# Conda builds of NumPy/MKL and PyTorch can load two Intel OpenMP runtimes on
# Windows. Configure the runtime before importing numeric libraries.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from uav_isac_basic import BasicUAVISACEnv


def run_basic_demo():
    np.set_printoptions(precision=3, suppress=True)

    env = BasicUAVISACEnv(seed=7)
    obs = env.reset()
    print("Initial observation")
    print(obs)
    print()

    print(
        "t | uav_position          | target_est_position   | "
        "SINR_c | comm_ok | SINR_s_bs | SINR_s_uav | rho"
    )
    print("-" * 92)

    for _ in range(8):
        direction = env.estimated_target_state[:3] - env.uav_position
        obs, info = env.step(
            uav_delta=direction,
            sensing_power_fraction=0.55,
            add_measurement_noise=False,
        )

        print(
            f"{info.t:1d} | "
            f"{info.uav_position} | "
            f"{info.estimated_target_state[:3]} | "
            f"{info.sinr_comm:6.2f} | "
            f"{str(info.constraints['communication_sinr']):7s} | "
            f"{info.sinr_sensing_bs:9.2f} | "
            f"{info.sinr_sensing_uav:11.2f} | "
            f"{info.rho:8.4f}"
        )

    print()
    print("Last paper-style observation:")
    print(obs)


def main():
    parser = argparse.ArgumentParser(description="UAV ISAC demos")
    parser.add_argument(
        "--mode",
        choices=("basic", "c-happo"),
        default="basic",
        help="Run the original single-UAV demo or the C-HAPPO RL trainer.",
    )
    args, remaining = parser.parse_known_args()

    if args.mode == "c-happo":
        from c_happo_hybrid_beamforming import main as c_happo_main

        sys.argv = [sys.argv[0], *remaining]
        c_happo_main()
        return

    run_basic_demo()


if __name__ == "__main__":
    main()

