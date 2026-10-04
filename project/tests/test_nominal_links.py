import unittest
from unittest.mock import patch

import numpy as np

from project.config import EnvConfig
from project.nominal_links import nominal_link_metrics
from project.physics import ISACPhysics


class NominalLinksTests(unittest.TestCase):
    def test_matches_physical_formulas_for_unit_power_fading_and_no_csi_error(self):
        cfg = EnvConfig(imperfect_csi_beta=0.)
        physics = ISACPhysics(cfg, np.random.default_rng(17))
        target = cfg.target_initial_state.copy()
        # Test the physical special case of unit fading; do not equate ratios
        # of expected powers with the expectation of sampled SINR in general.
        with patch("project.channels.rician_factor", return_value=1. + 0j):
            for active in (np.array([True, True]), np.array([False, True])):
                comm, sensing = nominal_link_metrics(cfg.uav_initial, target[:3], active, cfg)
                actual = physics.link_metrics(cfg.uav_initial, target, target, active)
                np.testing.assert_allclose(comm, actual["communication_sinrs"], rtol=1e-12)
                sources = np.r_[True, active]
                np.testing.assert_allclose(sensing[sources], actual["sensing_sinrs"][sources], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
