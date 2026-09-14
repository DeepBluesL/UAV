"""只复制 project 源码到空目录，验证运行时不需要任何旧项目模块。"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class StandaloneProjectTests(unittest.TestCase):
    def test_copied_project_can_train_without_legacy_files(self):
        project_dir = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project_dir / "tests") as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(project_dir))
            copied = root / "project"
            # 不复制输出、测试和缓存，避免目录递归包含自己的临时副本。
            shutil.copytree(project_dir, copied, ignore=shutil.ignore_patterns(
                "output", "tests", "__pycache__", ".pytest_cache"))
            script = """
import importlib.abc
import json
from pathlib import Path
import sys

class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'test', 'c_happo_hybrid_beamforming', 'dual_uav_rl', 'uav_isac_basic'
        }:
            raise ImportError('Unexpected legacy dependency: ' + fullname)
        return None

sys.meta_path.insert(0, RejectLegacy())
from project.formula_demo import run_demo
assert run_demo()['beams'].shape == (64, 3)
from project.train import main
sys.argv = ['project.train', '--epochs', '1', '--steps-per-epoch', '8',
            '--max-steps', '3', '--hidden-size', '8', '--eval-seeds', '31',
            '--output', 'project/output/standalone', '--no-plots']
main()
output = Path('project/output/standalone')
assert (output / 'policy.pt').is_file()
assert json.loads((output / 'training_summary.json').read_text())['environment_steps'] == 8
assert json.loads((output / 'evaluation.json').read_text())['summary']['episodes'] == 1
for name, module in list(sys.modules.items()):
    if name == 'project' or name.startswith('project.'):
        assert Path(module.__file__).resolve().is_relative_to(Path('project').resolve())
print('STANDALONE_OK')
"""
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-B", "-s", "-c", script], cwd=root, env=environment,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("STANDALONE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()