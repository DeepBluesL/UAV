import unittest

import numpy as np

from project import channels, communication, crb, measurements, pcrb, sensing


class TestChannelFormulas(unittest.TestCase):
    def setUp(self):
        self.bs = np.zeros(3)
        self.target = np.array([60.0, 5.0, 65.0])
        self.uavs = [
            np.array([30.0, 10.0, 70.0]),
            np.array([30.0, -10.0, 70.0]),
            np.array([45.0, 20.0, 55.0]),
        ]

    def test_units_power_and_rician_sample(self):
        self.assertEqual(channels.norm([0, 0, 0], [3, 4, 0]), 5.0)
        self.assertAlmostEqual(channels.db_to_linear(-50), 1e-5)
        self.assertAlmostEqual(channels.dbm_to_watt(-80), 1e-11)
        np.testing.assert_allclose(channels.abs2([1 + 2j, -3j]), [5.0, 9.0])
        self.assertAlmostEqual(channels.abs2_scalar([[2 - 3j]]), 13.0)
        np.testing.assert_allclose(channels.normalize([3.0, 4.0]), [.6, .8])

        expected_rng = np.random.default_rng(17)
        expected = (expected_rng.standard_normal() + 1j * expected_rng.standard_normal()) / np.sqrt(2)
        actual = channels.rician_factor(K=0, rng=np.random.default_rng(17))
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            channels.normalize(np.zeros(3))

    def test_channel_distance_scaling_and_zero_hops(self):
        beta0 = 2e-5
        expected_rng = np.random.default_rng(23)
        h = channels.rician_factor(K=4, rng=expected_rng)
        actual = channels.beta_c_bs_uav(
            self.bs, self.uavs[0], beta0, K=4, rng=np.random.default_rng(23))
        self.assertAlmostEqual(actual, h * np.sqrt(beta0) / channels.norm(self.bs, self.uavs[0]))

        expected_rng = np.random.default_rng(24)
        h = channels.rician_factor(K=4, rng=expected_rng)
        d1 = channels.norm(self.bs, self.target)
        d2 = channels.norm(self.target, self.uavs[0])
        two_hop = channels.beta_s_two_hop(
            self.bs, self.target, self.uavs[0], beta0, K=4,
            rng=np.random.default_rng(24))
        self.assertAlmostEqual(two_hop, h * np.sqrt(beta0) / (d1 * d2))
        with self.assertRaises(ValueError):
            channels.beta_c_bs_uav(self.bs, self.bs, beta0)
        with self.assertRaises(ValueError):
            channels.beta_s_bs_target(self.bs, self.bs)
        with self.assertRaises(ValueError):
            channels.beta_s_two_hop(self.bs, self.bs, self.target)

    def test_steering_phase_and_matched_beam_power(self):
        psi_x, psi_y = channels.direction_cosines(self.bs, [1.0, 0.0, 0.0])
        self.assertEqual((psi_x, psi_y), (1.0, 0.0))
        steering = channels.steering_vector_upa(self.bs, [1.0, 0.0, 0.0], Mx=2, My=2)
        np.testing.assert_allclose(steering[:, 0], [1.0, 1.0, -1.0, -1.0], atol=1e-15)

        w0, comm_beams = channels.make_matched_beams(
            self.bs, self.uavs, self.target, Mx=3, My=2, P=4.2)
        self.assertEqual(w0.shape, (6, 1))
        self.assertEqual(len(comm_beams), 3)
        powers = [np.linalg.norm(w) ** 2 for w in [w0] + comm_beams]
        np.testing.assert_allclose(powers, np.full(4, 4.2 / 4), atol=1e-14)

    def test_sinr_parts_and_optional_direct_interference(self):
        w0, beams = channels.make_matched_beams(
            self.bs, self.uavs, self.target, Mx=3, My=2, P=4.2)
        common = dict(Mx=3, My=2, beta0_s=2e-5, K=4.0, alpha0=.7, sigma0=2e-11)
        bs_parts = sensing.bs_sensing_sinr_eq6_parts(
            self.bs, self.target, beams, w0, sigma_ubs=3e-10,
            rng=np.random.default_rng(31), **common)
        self.assertAlmostEqual(
            bs_parts["denominator"],
            bs_parts["comm_beam_interference"] + bs_parts["sigma_ubs"] + bs_parts["sigma0"])
        self.assertAlmostEqual(bs_parts["sinr"], bs_parts["signal"] / bs_parts["denominator"])

        uav_parts = sensing.uav_sensing_sinr_eq12_parts(
            self.bs, self.uavs[0], self.target, beams, w0, sigma_un=3e-10,
            rng=np.random.default_rng(32), **common)
        self.assertAlmostEqual(
            uav_parts["denominator"],
            uav_parts["comm_beam_interference"] + uav_parts["sigma_un"] + uav_parts["sigma0"])

        comm_args = dict(
            u_bs=self.bs, uav_positions=self.uavs, target=self.target, n_idx=1,
            W_comm=beams, w0=w0, Mx=3, My=2, beta0_c=3e-5,
            beta0_s=2e-5, K=4.0, alpha0=.7, alpha1=.8, sigma0=2e-11)
        standard = communication.uav_comm_sinr_eq10_parts(
            rng=np.random.default_rng(33), include_direct_sensing_beam=False, **comm_args)
        extended = communication.uav_comm_sinr_eq10_parts(
            rng=np.random.default_rng(33), include_direct_sensing_beam=True, **comm_args)
        self.assertGreater(extended["direct_comm_interference"], standard["direct_comm_interference"])
        self.assertLess(extended["sinr"], standard["sinr"])
        expected_denominator = sum(standard[key] for key in (
            "direct_comm_interference", "target_echo_interference",
            "uav_echo_interference", "sigma0"))
        self.assertAlmostEqual(standard["denominator"], expected_denominator)


class TestEstimationFormulas(unittest.TestCase):
    def test_measurements_and_analytic_jacobian(self):
        zeta = measurements.make_cartesian_state(3, 4, 5, 1, -2, .5)
        measured = measurements.cartesian_state_to_measurement(zeta)
        d = np.sqrt(50.0)
        np.testing.assert_allclose(
            measured, [d, np.arccos(5 / d), np.arctan2(4, 3), -2.5 / d])

        p, velocity = zeta[:3], zeta[3:]
        xy = np.linalg.norm(p[:2])
        dot = velocity @ p
        expected = np.zeros((4, 6))
        expected[0, :3] = p / d
        expected[1, :3] = [p[0] * p[2] / (d * d * xy),
                           p[1] * p[2] / (d * d * xy), -xy / d ** 2]
        expected[2, :3] = [-p[1] / xy ** 2, p[0] / xy ** 2, 0.0]
        expected[3, :3] = velocity / d - dot * p / d ** 3
        expected[3, 3:] = p / d
        actual = measurements.numerical_jacobian(measurements.measurement_function_h, zeta, eps=1e-6)
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)
        with self.assertRaises(ValueError):
            measurements.measurement_function_h(np.zeros(6))

    def test_crb_information_fusion_and_validation(self):
        variances = crb.crb_from_sensing_sinr(
            2.0, bandwidth=10.0, kappa_d=1.0,
            kappa_theta=2.0, kappa_phi=3.0, kappa_v=4.0)
        np.testing.assert_allclose(variances, [.025, .5, .75, 1.0])
        sources = np.array([[1., 2., 4., 8.], [2., 4., 8., 16.]])
        weights = np.array([.25, .75])
        expected = 1.0 / (weights[0] / sources[0] + weights[1] / sources[1])
        fused = crb.ci_fusion_crb(sources, weights)
        np.testing.assert_allclose(fused, expected)
        np.testing.assert_array_equal(crb.measurement_noise_cov_from_fused_crb(fused), np.diag(fused))

        for invalid in (np.ones((2, 3)), np.array([[1., 2., 3., 0.]])):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    crb.ci_fusion_crb(invalid)
        with self.assertRaises(ValueError):
            crb.ci_fusion_crb(sources, [.2, .2])
        with self.assertRaises(ValueError):
            crb.crb_from_sensing_sinr(0.0)

    def test_motion_information_and_pcrb_identities(self):
        dt, intensity = .5, .3
        zeta = np.array([3., 4., 5., 1., -2., .5])
        F = pcrb.state_transition_matrix(dt)
        np.testing.assert_allclose(pcrb.predict_cartesian_state(zeta, dt), F @ zeta)
        np.testing.assert_allclose(F @ zeta, [3.5, 3., 5.25, 1., -2., .5])
        Q = pcrb.process_noise_covariance(dt, intensity)
        np.testing.assert_allclose(np.diag(Q)[:3], dt ** 3 * intensity / 3)
        np.testing.assert_allclose(np.diag(Q)[3:], dt * intensity)

        singular = np.diag([2.0, 0.0, 4.0])
        np.testing.assert_allclose(pcrb.safe_inverse(singular), np.diag([.5, 0.0, .25]))
        H = np.arange(24.0).reshape(4, 6) / 10
        Psi = np.diag([.4, .03, .02, .5])
        np.testing.assert_allclose(pcrb.data_fim_eq26(H, Psi), H.T @ np.linalg.inv(Psi) @ H)
        np.testing.assert_allclose(pcrb.data_fim_eq26(H, Psi, False), H.T @ Psi @ H)

        J_prev = np.diag([2., 3., 4., 5., 6., 7.])
        expected_keys = {"F", "Phi", "zeta_pred", "y_pred", "H", "Psi", "P_prior",
                         "J_prior", "J_data", "J", "PCRB", "rho"}
        for inverse in (False, True):
            with self.subTest(use_inverse_covariance=inverse):
                parts = pcrb.pcrb_eq24_to_eq28_parts(
                    zeta, J_prev, Psi, dt=dt, sigma2_zeta=intensity,
                    use_inverse_covariance=inverse)
                self.assertEqual(set(parts), expected_keys)
                np.testing.assert_allclose(parts["J"], parts["J_prior"] + parts["J_data"])
                np.testing.assert_allclose(parts["PCRB"] @ parts["J"], np.eye(6), atol=2e-7)
                self.assertAlmostEqual(parts["rho"], np.trace(parts["PCRB"]))
                np.testing.assert_allclose(parts["PCRB"], parts["PCRB"].T)


if __name__ == "__main__":
    unittest.main()
