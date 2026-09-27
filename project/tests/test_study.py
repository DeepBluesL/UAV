"""前缀RMSE单位与study的训练种子统计边界。"""

import unittest

import numpy as np

from project.benchmark_metrics import enrich_episode
from project.config import EnvConfig
from project.study_summary import _across_seeds, _seed_row
from project.tests.test_benchmark_metrics import base_row


class StudyMetricsTests(unittest.TestCase):
    def test_position_rmse_is_three_dimensional_and_excludes_initial(self):
        trace = {'rho_pos': np.ones(4), 'target_positions': np.zeros((4, 3)),
                 'estimated_target_positions': np.array([[100, 100, 100], [3, 4, 0],
                                                        [0, 0, 5], [0, 0, 5]])}
        row = enrich_episode(base_row(), trace, EnvConfig(), prefix_steps=3)
        self.assertAlmostEqual(row['tracking_prefix_rmse'], 5.)
        self.assertIsNone(enrich_episode(base_row(), trace, EnvConfig(), 4)['tracking_prefix_rmse'])

    def test_training_seeds_receive_equal_weight_despite_different_episode_counts(self):
        rows = []
        for seed, success, count in ((7, 1, 2), (17, 0, 8)):
            episodes = [{'team_success': success, 'episode_return': 10 * success}
                        for _ in range(count)]
            rows.append(_seed_row('pure', seed, 'nominal', episodes))
        result = _across_seeds(rows)[0]
        self.assertEqual(result['training_seeds'], 2)
        self.assertEqual(result['evaluation_episodes'], 10)
        self.assertEqual(result['success_rate_mean'], .5)
        self.assertAlmostEqual(result['success_rate_std'], np.sqrt(.5))


if __name__ == '__main__':
    unittest.main()
