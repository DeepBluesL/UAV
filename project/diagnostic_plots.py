"""Small plots for sensing diagnostic output."""

from pathlib import Path

import numpy as np


def plot_diagnostics(output, episodes, source_rows):
    import matplotlib.pyplot as plt

    output = Path(output)
    groups = sorted({(row["scenario"], row["path"]) for row in episodes})
    labels = [f"{scenario}\n{path}" for scenario, path in groups]
    rmse = [_nanmean([r["rmse_first10"] for r in episodes
                      if (r["scenario"], r["path"]) == group]) for group in groups]
    time = [_nanmean([r["episode_time"] for r in episodes
                      if (r["scenario"], r["path"]) == group]) for group in groups]
    success = [np.mean([r["team_success"] for r in episodes
                        if (r["scenario"], r["path"]) == group]) for group in groups]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(max(10, len(groups) * .85), 9), constrained_layout=True)
    axes[0].bar(x, rmse); axes[0].set_ylabel("First-10\nposition RMSE (m)")
    axes[1].bar(x, time); axes[1].set_ylabel("Episode time (s)")
    axes[2].bar(x, success); axes[2].set_ylabel("Success rate"); axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(x, labels, rotation=45, ha="right")
    fig.savefig(output / "rmse_time.png", dpi=160); plt.close(fig)

    if not source_rows:
        return
    components = ("range", "azimuth", "elevation", "range_rate")
    groups = sorted({(r["scenario"], r["path"], r["source"]) for r in source_rows})
    lookup = {(r["scenario"], r["path"], r["source"], r["component"]):
              r["floor_hit_fraction"] for r in source_rows}
    matrix = np.asarray([[lookup.get((*group, component), np.nan) for component in components]
                         for group in groups], dtype=float)
    fig, ax = plt.subplots(figsize=(7, max(4, len(groups) * .24)), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(4), components)
    ax.set_yticks(range(len(groups)), ["/".join(group) for group in groups], fontsize=7)
    ax.set_title("Delivered-measurement CRB floor-hit fraction")
    fig.colorbar(image, ax=ax); fig.savefig(output / "floor_hits.png", dpi=160); plt.close(fig)


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
