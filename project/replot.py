"""从已有输出重绘：python -m project.replot --output project/output/run_seed7。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .plot import plot_results


def _csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def load_trace(output):
    with np.load(output / "trajectory.npz") as saved:
        trace = {key: saved[key] for key in saved.files}
    if "bs_position" not in trace:
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        trace["bs_position"] = np.asarray(config["environment"]["bs_position"], dtype=float)
    # 旧输出没有估计轨迹时保持缺失，绘图不会虚构数据。
    return trace


def main():
    parser = argparse.ArgumentParser(description="Redraw plots from an existing run directory.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    output = args.output.resolve()
    if not output.is_relative_to(project_dir):
        raise ValueError("Keep generated files under UAV/project")
    plot_results(output, _csv_rows(output / "training.csv"),
                 _csv_rows(output / "episodes.csv"), load_trace(output))


if __name__ == "__main__":
    main()
