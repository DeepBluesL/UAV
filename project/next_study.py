"""Sequential reproducible study: training, frozen evaluation and reporting."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .artifacts import write_json
from .next_study_jobs import PROJECT, prepare_study, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "next_study_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "summarize"), default="all")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or args.jobs < 1:
        parser.error("Use an output inside project and positive --jobs")
    if args.stage in {"all", "train"}:
        protocol = json.loads(args.config.read_text(encoding="utf-8"))
        manifest, tasks = prepare_study(protocol, output, args.jobs)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = [pool.submit(run_training, task, protocol["validation_seeds"]) for task in tasks]
            for future in as_completed(pending):
                future.result()
        manifest["status"] = "trained"
        write_json(output / "study_manifest.json", manifest)
    if args.stage in {"all", "evaluate"}:
        from .next_study_evaluate import evaluate_next_study
        evaluate_next_study(output, args.jobs)
    if args.stage in {"all", "evaluate", "summarize"}:
        from .next_study_summary import summarize_next_study
        from .next_study_plots import plot_next_study
        summarize_next_study(output)
        plot_next_study(output)
        manifest_path = output / "study_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
    print(f"Study {args.stage} complete: {output}", flush=True)


if __name__ == "__main__":
    main()
