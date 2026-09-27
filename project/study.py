"""批量训练与比较入口；每个训练种子使用独立进程，结果目录只对应一次实验。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import write_json

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / 'project'


def training_jobs(spec, output):
    """先写出每个实验的完整配置，便于独立复跑和人工修改。"""
    base = json.loads((ROOT / spec['base_config']).read_text(encoding='utf-8'))
    (output / 'configs').mkdir(exist_ok=True)
    write_json(output / 'configs' / 'base.json', base)
    suite = json.loads((ROOT / spec['suite']).read_text(encoding='utf-8'))
    write_json(output / 'configs' / 'suite.json', suite)
    jobs = []
    for variant, overrides in spec['variants'].items():
        for seed in spec['training_seeds']:
            name = f'{variant}_seed{seed}'
            settings = {**base, 'ppo': {**base.get('ppo', {}), **spec['training'],
                                       **overrides, 'seed': seed}}
            config_path = output / 'configs' / f'{name}.json'
            write_json(config_path, settings)
            jobs.append((name, config_path, output / 'train' / name))
    return jobs


def run_training(job, validation_seeds):
    name, config_path, directory = job
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-m', 'project.train', '--config', str(config_path),
               '--output', str(directory), '--no-plots', '--eval-seeds',
               *map(str, validation_seeds)]
    # 小网络和 6x6 滤波矩阵不适合开很多 BLAS 线程；CUDA 仍由配置指定。
    child_env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
    with (directory / 'console.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, env=child_env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed; see {directory / "console.log"}')
    print(f'Training complete: {name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Reproducible SA, EKF, and pure/residual RL study.')
    parser.add_argument('--config', type=Path, default=PROJECT / 'study_config.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('all', 'train', 'evaluate', 'summarize'), default='all')
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or args.jobs < 1:
        parser.error('Use an output directory inside project and --jobs >= 1')
    spec = json.loads(args.config.read_text(encoding='utf-8'))
    if args.stage in ('all', 'train'):
        output.mkdir(parents=True, exist_ok=False)
        manifest = {**spec, 'status': 'training', 'parallel_jobs': args.jobs,
                    'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(PROJECT.glob('*.py'))},
                    'protocol': [
                        'Equal joint-environment-step budgets and three independent training seeds.',
                        'No checkpoint selection on evaluation seeds; use final saved checkpoints.',
                        'Compare fixed vs randomized pure RL, then pure vs residual with randomized training.',
                        'Bounds/speed tests change only declared limits, not starts/goals or deadlines.',
                        'Rules are centralized navigation references where applicable; RL executes local actors.',
                        'Report training-seed variation separately from evaluation noise.',
                    ]}
        write_json(output / 'study_manifest.json', manifest)
        jobs = training_jobs(spec, output)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = [pool.submit(run_training, job, spec['validation_seeds']) for job in jobs]
            for job in as_completed(pending):
                job.result()
        manifest['status'] = 'trained'
        write_json(output / 'study_manifest.json', manifest)
    if args.stage in ('all', 'evaluate'):
        from .study_evaluate import evaluate_study
        evaluate_study(output)
    if args.stage in ('all', 'evaluate', 'summarize'):
        from .study_summary import summarize_study
        from .study_plots import make_study_plots
        summarize_study(output)
        make_study_plots(output)
        from .plot_tracking import plot_tracking_ablation
        plot_tracking_ablation(output)
    print(f'Study {args.stage} complete: {output}', flush=True)


if __name__ == '__main__':
    main()
