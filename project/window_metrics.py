"""Comparable fixed-window tracking/link metrics; truth is used offline only."""

import numpy as np


WINDOW_METRICS = (
    "communication_prefix_rate", "tracking_prefix_early_rmse",
    "tracking_prefix_late_rmse", "tracking_prefix_p95_error",
    "position_nees_prefix_mean", "position_coverage_prefix_rate",
)


def window_metrics(trace, config, prefix_steps):
    """Arrival-step sources count as active; shorter episodes remain missing.

    Early means steps 1..min(3, prefix); late means steps 4..prefix. Neither
    replaces the prespecified full prefix, which retains initialization errors.
    """
    result = dict.fromkeys(WINDOW_METRICS)
    if len(trace["rho_pos"]) - 1 < prefix_steps:
        return result
    if "communication_sinrs" in trace and "source_mask" in trace:
        active = np.asarray(trace["source_mask"][:prefix_steps, 1:], dtype=bool)
        communication = np.asarray(trace["communication_sinrs"][:prefix_steps])
        if active.any():
            result["communication_prefix_rate"] = float(
                np.mean(communication[active] >= config.gamma_min))
    if "estimated_target_positions" not in trace:
        return result
    error = (trace["estimated_target_positions"][1:prefix_steps + 1]
             - trace["target_positions"][1:prefix_steps + 1])
    squared = np.sum(error ** 2, axis=1)
    result["tracking_prefix_early_rmse"] = float(np.sqrt(np.mean(squared[:3])))
    if prefix_steps > 3:
        result["tracking_prefix_late_rmse"] = float(np.sqrt(np.mean(squared[3:])))
    result["tracking_prefix_p95_error"] = float(np.quantile(np.sqrt(squared), .95))
    if (str(trace.get("uncertainty_kind", "")) == "ekf_covariance"
            and "tracking_covariances" in trace):
        covariance = trace["tracking_covariances"][1:prefix_steps + 1, :3, :3]
        nees = np.einsum("bi,bi->b", error, np.linalg.solve(covariance, error[..., None])[..., 0])
        result["position_nees_prefix_mean"] = float(np.mean(nees))
        # Nominal 95% chi-square(3) ellipsoid; correlated steps are not iid tests.
        result["position_coverage_prefix_rate"] = float(np.mean(nees <= 7.814727903251179))
    return result
