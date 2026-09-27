"""同一导航轨迹上的滤波消融：真值只用于离线绘图与评分。"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_tracking(trace, output):
    times = trace['times']
    truth = trace['target_positions']
    error = np.linalg.norm(trace['estimated_target_positions'] - truth, axis=1)
    prior = np.linalg.norm(trace['prior_target_positions'] - truth, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(times, prior, label='Prediction error', alpha=.7)
    axes[0].plot(times, error, label='Posterior error')
    axes[0].set(title='Realized 3D tracking error', ylabel='Position error (m)')
    for i, name in enumerate(('BS', 'UAV1', 'UAV2')):
        axes[1].step(times, trace['measurement_source_mask'][:, i].cumsum(),
                     where='post', label=name)
    axes[1].set(title='Measurements delivered to the filter', ylabel='Cumulative count')
    for ax in axes:
        ax.set_xlabel('Time (s)'); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / 'tracking.png', dpi=170)
    plt.close(fig)


def plot_tracking_ablation(output):
    """同Goal导航路径、同首个评估种子，展示测量带来的误差修正。"""
    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cases = (('prediction_only', 'Prediction only'), ('bs_only', 'BS only'),
             ('nominal', 'BS + UAV fusion'))
    for scenario, label in cases:
        path = output / 'eval' / 'rules' / 'trajectories' / scenario / 'goal' / 'trajectory.npz'
        with np.load(path) as trace:
            error = np.linalg.norm(trace['estimated_target_positions'] - trace['target_positions'], axis=1)
            axes[0].semilogy(trace['times'], error, label=label)
            axes[1].semilogy(trace['times'], trace['rho_pos'], label=label)
    axes[0].set(title='Realized position error', ylabel='3D position error (m)')
    axes[1].set(title='EKF position uncertainty', ylabel='trace(P position), m²')
    for ax in axes:
        ax.set_xlabel('Time (s)'); ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Same Goal-controller path; first evaluation seed; truth used only for scoring')
    fig.tight_layout()
    fig.savefig(output / 'tracking_ablation.png', dpi=170)
    plt.close(fig)
