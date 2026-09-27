"""固定测试场景和种子评估整组模型；不会根据评估结果修改训练参数。"""

from argparse import Namespace
import json
from pathlib import Path

from .artifacts import write_json
from .benchmark import run_benchmark

ROOT = Path(__file__).resolve().parent.parent


def evaluate_study(output):
    output = Path(output)
    manifest = json.loads((output / 'study_manifest.json').read_text(encoding='utf-8'))
    seeds = manifest['evaluation_seeds']
    common = dict(suite=output / 'configs' / 'suite.json', scenarios=None, seed_start=seeds[0],
                  episodes=len(seeds), eval_seeds=seeds, reference='goal', device='cpu',
                  no_plots=False)
    run_benchmark(Namespace(**common, config=output / 'configs' / 'base.json', checkpoint=None,
                            methods=manifest['rules'], output=output / 'eval' / 'rules'))
    for variant in manifest['variants']:
        for seed in manifest['training_seeds']:
            name = f'{variant}_seed{seed}'
            run_benchmark(Namespace(**{**common, 'reference': 'mappo'}, config=None,
                                    checkpoint=output / 'train' / name / 'policy.pt',
                                    methods=['mappo'], output=output / 'eval' / name))
    manifest['status'] = 'complete'
    write_json(output / 'study_manifest.json', manifest)
