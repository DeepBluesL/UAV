import numpy as np

DIST_EPS = 1e-12
FLOAT_TINY = np.finfo(float).tiny


def norm(p, q):
    """Eclidean distance between two 3D points"""
    return np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))


def db_to_linear(db):
    """convert db channel gain to linear scale
        Example: -50db -> 1e-5
    """
    return 10 ** (db / 10)


def dbm_to_watt(dbm):
    """convert dbm power to watt
        Example: -80 dBm -> 1e-11 W
    """
    return 10 ** ((dbm - 30) / 10)


# 复数信号幅度:x, 对应功率|x|^2
def abs2(x):
    return np.abs(x) ** 2


def abs2_scalar(x):
    arr = np.asarray(x)
    return float(np.abs(arr.reshape(-1)[0]) ** 2)


# 向量归一化成单位长度
def normalize(v):
    v = np.asarray(v)
    v_norm = np.linalg.norm(v)
    if v_norm <= DIST_EPS:
        raise ValueError("Cannot normalize a zero-length vector.")
    return v / v_norm


u_bs = np.array([0.0, 0.0, 0.0])
uav_pos = [
    np.array([30.0, 10.0, 70.0]),
    np.array([30.0, -10.0, 70.0])
]
target = np.array([60.0, 0.0, 60.0])

N = len(uav_pos)


# Rician channel gain
def rician_factor(K=10, rng=None):
    """
        Rician fading factor:
            sqrt(K/(K+1)) + sqrt(1/(K+1)) * g
        where g ~ CN(0, 1).
    """

    if rng is None:
        rng = np.random.default_rng()

    g = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    return np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * g


# BS-UAV communication channel
def beta_c_bs_uav(u_bs, uav, beta0_c, K=10.0, rng=None):
    """
        Communication channel gain β^c_{u_bs, u_n}.
        Corresponds to Eq. (8).

        Path:
            BS -> UAV

        Power gain:
            beta0_c / d^2
        Amplitude gain:
            sqrt(beta0_c / d^2)
    """

    d = norm(u_bs, uav)
    if d <= DIST_EPS:
        raise ValueError("BS and UAV positions must be different.")
    h = rician_factor(K=K, rng=rng)
    return h * np.sqrt(beta0_c / d ** 2)


# The sensing channel gain between the BS and the target
def beta_s_bs_target(u_bs, target, beta0_s=1e-5, K=10, rng=None):
    d = norm(u_bs, target)
    if d <= DIST_EPS:
        raise ValueError("BS and target positions must be different.")
    h = rician_factor(K, rng)
    return h * np.sqrt(beta0_s / (d ** 4))


def beta_s_two_hop(u_bs, reflector, receiver, beta0_s=1e-5, K=10.0, rng=None):
    """
        Two-hop sensing/reflection channel.

        General form:
            BS -> reflector -> receiver

        For target echo:
            reflector = target
            receiver = UAV n
            corresponds to β^s_{u_n, ξ_t} in Eq. (7)

        For UAV-reflected interference:
            reflector = UAV i
            receiver = UAV n
            corresponds to β^s_{u_n, u_i} in Eq. (9)/(10)

        Power gain:
            beta0_s / (d_BS_reflector^2 * d_receiver_reflector^2)
    """
    d1 = norm(u_bs, reflector)
    d2 = norm(receiver, reflector)
    if d1 <= DIST_EPS or d2 <= DIST_EPS:
        raise ValueError("Two-hop sensing path contains a zero-length hop.")
    h = rician_factor(K=K, rng=rng)
    return h * np.sqrt(beta0_s / (d1 ** 2 * d2 ** 2))


# steering vector
def direction_cosines(tx_pos, rx_pos):
    """
        Compute x/y direction cosines from BS to a target/UAV.

        tx_pos: BS position, shape (3,)
        rx_pos: target or UAV position, shape (3,)
        """
    tx_pos = np.asarray(tx_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)

    diff = rx_pos - tx_pos
    d = np.linalg.norm(diff)

    if d < DIST_EPS:
        raise ValueError("tx_pos and rx_pos are identical or too close.")

    psi_x = diff[0] / d
    psi_y = diff[1] / d

    return psi_x, psi_y


def steering_vector_upa(
        tx_pos, rx_pos, Mx=8, My=8, dx=None, dy=None, wavelength=1.0
):
    """
        UPA steering vector corresponding to Eq. (4).

        Return:
            a: shape (Mx*My, 1)
        """
    if dx is None:
        dx = wavelength / 2
    if dy is None:
        dy = wavelength / 2

    psi_x, psi_y = direction_cosines(tx_pos, rx_pos)

    mx = np.arange(Mx)
    my = np.arange(My)

    # The paper defines a^H with a negative phase in Eq. (4), so the
    # column vector a uses the corresponding positive phase.
    ax = np.exp(1j * 2 * np.pi * dx * mx * psi_x / wavelength)
    ay = np.exp(1j * 2 * np.pi * dy * my * psi_y / wavelength)

    a = np.kron(ax, ay)
    return a.reshape(-1, 1)


def make_matched_beams(u_bs, uav_positions, target, Mx=8, My=8, P=5.0):
    """
    Create simple matched beams for sanity testing.

    This is not the paper's final C-HAPPO beamforming.
    This is only for understanding and debugging SINR formulas.

    Beam definition:
        w0:
            sensing beam aligned with target

        W_comm[n]:
            communication beam aligned with UAV n

    Power allocation:
        total beams = N communication beams + 1 sensing beam = N + 1
        each beam gets P / (N + 1)

    Paper power constraint Eq. (34d):
        sum ||w_n,t||^2 = P
    """
    N = len(uav_positions)
    power_per_beam = P / (N + 1)

    # Sensing beam w0 aligned with target
    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)
    w0 = normalize(a_target) * np.sqrt(power_per_beam)

    # Communication beams w1...wN aligned with each UAV
    W_comm = []
    for uav in uav_positions:
        a_uav = steering_vector_upa(u_bs, uav, Mx=Mx, My=My)
        wk = normalize(a_uav) * np.sqrt(power_per_beam)
        W_comm.append(wk)

    # Sanity check: total transmit power should be P
    total_power = np.linalg.norm(w0) ** 2
    total_power += sum(np.linalg.norm(wk) ** 2 for wk in W_comm)

    if not np.isclose(total_power, P, atol=1e-8):
        raise RuntimeError(f"Power mismatch: total_power = {total_power}, P = {P}")

    return w0, W_comm


def bs_sensing_sinr_eq6_parts(
        u_bs,
        target,
        W_comm,
        w0,
        Mx=8,
        My=8,
        beta0_s=1e-5,
        K=10.0,
        alpha0=0.9,
        sigma_ubs=1e-10,
        sigma0=1e-11,
        rng=None
):
    """
    Compute BS sensing SINR.

    Corresponds to paper Eq. (6):

        SINR^s_{u_bs}(t)
        =
        |alpha0 beta^s_{u_bs,xi} a_xi^H w0|^2
        /
        (sum_{k=1}^{N} |alpha0 beta^s_{u_bs,xi} a_xi^H wk|^2
         + sigma_ubs^2 + sigma0^2)

    Parts:
        signal:
            useful target echo power from sensing beam w0

        comm_beam_interference:
            communication beams reflected by target and received at BS

        sigma_ubs:
            residual cancellation noise / unresolved multipath at BS

        sigma0:
            AWGN
    """
    if rng is None:
        rng = np.random.default_rng()

    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)

    beta_bs_target = beta_s_bs_target(
        u_bs=u_bs,
        target=target,
        beta0_s=beta0_s,
        K=K,
        rng=rng
    )

    # Numerator: useful sensing echo from w0
    signal = abs2_scalar(
        alpha0 * beta_bs_target * (a_target.conj().T @ w0)
    )

    # Denominator part: interference from communication beams
    comm_beam_interference = 0.0
    for wk in W_comm:
        comm_beam_interference += abs2_scalar(
            alpha0 * beta_bs_target * (a_target.conj().T @ wk)
        )

    denominator = comm_beam_interference + sigma_ubs + sigma0

    sinr = signal / max(denominator, FLOAT_TINY)

    return {
        "signal": signal,
        "comm_beam_interference": comm_beam_interference,
        "sigma_ubs": sigma_ubs,
        "sigma0": sigma0,
        "denominator": denominator,
        "sinr": sinr,
        "sinr_db": 10 * np.log10(max(sinr, FLOAT_TINY))
    }


# ============================================================
# 5. Eq. (10): UAV communication SINR
#    uav_comm_sinr_eq10_parts
# ============================================================

def uav_comm_sinr_eq10_parts(
        u_bs,
        uav_positions,
        target,
        n_idx,
        W_comm,
        w0,
        Mx=8,
        My=8,
        beta0_c=1e-5,
        beta0_s=1e-5,
        K=10.0,
        alpha0=0.9,
        alpha1=0.9,
        sigma0=1e-11,
        rng=None,
        include_direct_sensing_beam=False
):
    """
    Compute UAV n communication SINR.

    Corresponds closely to paper Eq. (10).

    Python index:
        n_idx = 0 means UAV 1 in the paper
        n_idx = 1 means UAV 2 in the paper

    Paper Eq. (10) structure:

        SINR^c_{u_n}(t)
        =
        desired_signal
        /
        (
            direct_comm_interference
            + target_echo_interference
            + uav_echo_interference
            + sigma0
        )

    ------------------------------------------------------------
    Numerator:
        | beta^c_{bs,n} a_{u_n}^H w_n |^2

    Denominator part 1:
        sum_{k=1,k!=n}^{N}
        | beta^c_{bs,n} a_{u_n}^H w_k |^2

    Denominator part 2:
        sum_{k=0}^{N}
        | alpha0 beta^s_{u_n,xi} a_xi^H w_k |^2

    Denominator part 3:
        sum_{i=1,i!=n}^{N} sum_{k=0}^{N}
        | alpha1 beta^s_{u_n,u_i} a_{u_i}^H w_k |^2

    Denominator part 4:
        sigma0
    ------------------------------------------------------------
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(uav_positions)

    if len(W_comm) != N:
        raise ValueError("len(W_comm) must equal number of UAVs.")

    if not (0 <= n_idx < N):
        raise ValueError("n_idx out of range.")

    # Current UAV position: u_{n,t}
    uav_n = uav_positions[n_idx]

    # All BS beams indexed like the paper:
    #   all_beams[0] = w0
    #   all_beams[1] = w1
    #   all_beams[2] = w2
    all_beams = [w0] + list(W_comm)

    # Steering vector to current UAV n
    a_uav_n = steering_vector_upa(u_bs, uav_n, Mx=Mx, My=My)

    # Steering vector to target
    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)

    # Communication channel BS -> UAV n
    beta_c_n = beta_c_bs_uav(
        u_bs=u_bs,
        uav=uav_n,
        beta0_c=beta0_c,
        K=K,
        rng=rng
    )

    # Sensing/reflection channel BS -> target -> UAV n
    beta_s_target_to_n = beta_s_two_hop(
        u_bs=u_bs,
        reflector=target,
        receiver=uav_n,
        beta0_s=beta0_s,
        K=K,
        rng=rng
    )

    # ---------------------------------------------------------
    # Numerator:
    # desired_signal = | beta_c_n * a_uav_n^H * w_n |^2
    # ---------------------------------------------------------
    w_n = W_comm[n_idx]

    desired_signal = abs2_scalar(
        beta_c_n * (a_uav_n.conj().T @ w_n)
    )

    # ---------------------------------------------------------
    # Denominator part 1:
    # direct communication interference from other communication beams
    #
    # sum_{k=1,k!=n}^{N}
    # | beta_c_n * a_uav_n^H * w_k |^2
    # ---------------------------------------------------------
    direct_comm_interference = 0.0

    for k_idx, wk in enumerate(W_comm):
        if k_idx == n_idx:
            continue

        direct_comm_interference += abs2_scalar(
            beta_c_n * (a_uav_n.conj().T @ wk)
        )

    # Optional physical extension:
    # Eq. (10) does not include direct w0 interference in this term.
    # But Eq. (9) has sum_{k=0}^{N} in the direct received signal.
    if include_direct_sensing_beam:
        direct_comm_interference += abs2_scalar(
            beta_c_n * (a_uav_n.conj().T @ w0)
        )

    # ---------------------------------------------------------
    # Denominator part 2:
    # target-reflected interference from all beams
    #
    # sum_{k=0}^{N}
    # | alpha0 * beta_s_target_to_n * a_target^H * w_k |^2
    # ---------------------------------------------------------
    target_echo_interference = 0.0

    for wk in all_beams:
        target_echo_interference += abs2_scalar(
            alpha0 * beta_s_target_to_n * (a_target.conj().T @ wk)
        )

    # ---------------------------------------------------------
    # Denominator part 3:
    # other UAV-reflected interference
    #
    # sum_{i=1,i!=n}^{N} sum_{k=0}^{N}
    # | alpha1 * beta_s_{u_n,u_i} * a_{u_i}^H * w_k |^2
    # ---------------------------------------------------------
    uav_echo_interference = 0.0

    for i_idx, uav_i in enumerate(uav_positions):
        if i_idx == n_idx:
            continue

        a_uav_i = steering_vector_upa(u_bs, uav_i, Mx=Mx, My=My)

        beta_s_uav_i_to_n = beta_s_two_hop(
            u_bs=u_bs,
            reflector=uav_i,
            receiver=uav_n,
            beta0_s=beta0_s,
            K=K,
            rng=rng
        )

        for wk in all_beams:
            uav_echo_interference += abs2_scalar(
                alpha1 * beta_s_uav_i_to_n * (a_uav_i.conj().T @ wk)
            )

    # ---------------------------------------------------------
    # Denominator and SINR
    # ---------------------------------------------------------
    denominator = (
            direct_comm_interference
            + target_echo_interference
            + uav_echo_interference
            + sigma0
    )

    sinr = desired_signal / max(denominator, FLOAT_TINY)

    return {
        "desired_signal": desired_signal,
        "direct_comm_interference": direct_comm_interference,
        "target_echo_interference": target_echo_interference,
        "uav_echo_interference": uav_echo_interference,
        "sigma0": sigma0,
        "denominator": denominator,
        "sinr": sinr,
        "sinr_db": 10 * np.log10(max(sinr, FLOAT_TINY))
    }


# ============================================================
# 6. Eq. (12): UAV sensing SINR
#    uav_sensing_sinr_eq12_parts
# ============================================================

def uav_sensing_sinr_eq12_parts(
        u_bs,
        uav,
        target,
        W_comm,
        w0,
        Mx=8,
        My=8,
        beta0_s=1e-5,
        K=10.0,
        alpha0=0.9,
        sigma_un=1e-10,
        sigma0=1e-11,
        rng=None
):
    """
    Compute UAV sensing SINR.

    Corresponds to paper Eq. (12):

        SINR^s_{u_n}(t)
        =
        |alpha0 beta^s_{u_n,xi} a_xi^H w0|^2
        /
        (
            sum_{k=1}^{N}
            |alpha0 beta^s_{u_n,xi} a_xi^H wk|^2
            + sigma_un^2
            + sigma0^2
        )

    Important:
        The steering vector here is a_target, not a_uav.

        Because the BS sensing beam is directed toward the target,
        and the echo path is BS -> target -> UAV.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Steering vector to target
    a_target = steering_vector_upa(u_bs, target, Mx=Mx, My=My)

    # Sensing/reflection channel BS -> target -> UAV
    beta_s_target_to_uav = beta_s_two_hop(
        u_bs=u_bs,
        reflector=target,
        receiver=uav,
        beta0_s=beta0_s,
        K=K,
        rng=rng
    )

    # Numerator: useful sensing echo from w0
    signal = abs2_scalar(
        alpha0 * beta_s_target_to_uav * (a_target.conj().T @ w0)
    )

    # Denominator part: communication beam interference through target echo
    comm_beam_interference = 0.0

    for wk in W_comm:
        comm_beam_interference += abs2_scalar(
            alpha0 * beta_s_target_to_uav * (a_target.conj().T @ wk)
        )

    denominator = comm_beam_interference + sigma_un + sigma0

    sinr = signal / max(denominator, FLOAT_TINY)

    return {
        "signal": signal,
        "comm_beam_interference": comm_beam_interference,
        "sigma_un": sigma_un,
        "sigma0": sigma0,
        "denominator": denominator,
        "sinr": sinr,
        "sinr_db": 10 * np.log10(max(sinr, FLOAT_TINY))
    }


# ============================================================
# 8. Eq. (13)-(16): Sensing Information Fusion
#    make_measurement_vector
#    cartesian_state_to_measurement
#    crb_from_sensing_sinr
#    ci_fusion_crb
# ============================================================

def make_measurement_vector(d, theta, phi, radial_velocity):
    return np.array([d, theta, phi, radial_velocity])


# Eq.22
def cartesian_state_to_measurement(zeta):
    """
       Convert Cartesian target state to spherical measurement.

       zeta:
           [x, y, z, vx, vy, vz]

       Output:
           c = [d, theta, phi, radial_velocity]

       This corresponds to Eq. (13), and the detailed transformation
       is later written in Eq. (22) of the paper.
    """
    x, y, z, vx, vy, vz = zeta

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if d <= DIST_EPS:
        raise ValueError("Target position cannot coincide with the BS.")

    theta = np.arccos(z / d)

    phi = np.arctan2(y, x)

    radial_velocity = (vx * x + vy * y + vz * z) / d

    return np.array([d, theta, phi, radial_velocity])


# Eq.14
def crb_from_sensing_sinr(
        sensing_sinr,
        bandwidth=100e6,
        kappa_d=1.0,
        kappa_theta=1e-6,
        kappa_phi=1e-6,
        kappa_v=1e-6
):
    """
    convert sensing SINR to CRB for [d,theta,phi,v]
    """

    sinr_abs2 = float(np.abs(sensing_sinr) ** 2)
    if not np.isfinite(sinr_abs2) or sinr_abs2 <= 0.0:
        raise ValueError("sensing_sinr must be finite and non-zero.")

    sigma2_d = kappa_d / (sinr_abs2 * bandwidth)
    sigma2_theta = kappa_theta / sinr_abs2
    sigma2_phi = kappa_phi / sinr_abs2
    sigma2_v = kappa_v / sinr_abs2

    return np.array([sigma2_d, sigma2_theta, sigma2_phi, sigma2_v], dtype=float)


# Eq.15-16
def ci_fusion_crb(crb_list, weights=None):
    """
    Eq.15-16

    Input:
    crb_list:
            list of CRB vectors.

            crb_list[0]:
                CRB from BS sensing SINR

            crb_list[1]:
                CRB from UAV 1 sensing SINR

            crb_list[2]:
                CRB from UAV 2 sensing SINR

            Each CRB vector has shape (4,):
                [sigma2_d, sigma2_theta, sigma2_phi, sigma2_v]

        weights:
            CI weights omega_n.
            If None, use equal weights 1/(N+1), consistent with paper setup.

    Output:
        fused_crb:
            [bar_sigma2_d, bar_sigma2_theta, bar_sigma2_phi, bar_sigma2_v]

    """
    crb_array = np.asarray(crb_list, dtype=float)

    if crb_array.ndim != 2 or crb_array.shape[1] != 4:
        raise ValueError("crb_list must have shape (num_nodes, 4,).")

    num_nodes = crb_array.shape[0]

    if weights is None:
        weights = np.ones(num_nodes) / num_nodes
    else:
        weights = np.asarray(weights, dtype=float)

    if weights.shape[0] != num_nodes:
        raise ValueError("weights must equal number of CRB vectors.")

    if np.any(weights < 0):
        raise ValueError("CI weights must be non-negative.")

    if not np.isclose(np.sum(weights), 1.0, atol=1e-8):
        raise ValueError("CI weights must sum to 1.")

    if np.any(~np.isfinite(crb_array)) or np.any(crb_array <= 0.0):
        raise ValueError("All CRB entries must be finite and positive.")

    information_array = 1.0 / crb_array

    fused_information = np.sum(weights[:, None] * information_array, axis=0)

    fused_crb = 1.0 / fused_information

    return fused_crb


def measurement_noise_cov_from_fused_crb(fused_crb):
    """
    Eq. (23), will need after Eq. (15).

    Psi_t = diag(
        bar_sigma2_d,
        bar_sigma2_theta,
        bar_sigma2_phi,
        bar_sigma2_v
    )
    """
    fused_crb = np.asarray(fused_crb, dtype=float)

    if fused_crb.shape != (4,):
        raise ValueError("fused_crb must have shape (4,).")

    return np.diag(fused_crb)


# ============================================================
# 9. Eq. (17)-(28): Target motion, measurement model, PCRB
#    make_cartesian_state
#    state_transition_matrix
#    process_noise_covariance
#    predict_cartesian_state
#    measurement_function_h
#    numerical_jacobian
#    prior_fim_eq25
#    data_fim_eq26
#    pcrb_eq24_to_eq28_parts
# ============================================================

# Eq.17
def make_cartesian_state(x, y, z, vx, vy, vz):
    return np.array([x, y, z, vx, vy, vz], dtype=float)


# Eq.19
def state_transition_matrix(dt=1.0):
    """
       Eq. (19):
           F_zeta = [[1, T],
                     [0, 1]] ⊗ I_3

       For state order:
           zeta = [x, y, z, vx, vy, vz]^T

       The expanded matrix is:
           x_new  = x  + vx * dt
           y_new  = y  + vy * dt
           z_new  = z  + vz * dt
           vx_new = vx
           vy_new = vy
           vz_new = vz
       """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    F = np.block([
        [I3, dt * I3],
        [Z3, I3]
    ])

    return F


def process_noise_covariance(dt=1.0, sigma2_zeta=0.1):
    """
        Eq. (20):
            Phi_zeta =
                [[T^3/3, T^2/2],
                 [T^2/2, T    ]] ⊗ sigma2_zeta * I_3

        For state order:
            [x, y, z, vx, vy, vz]

        Interpretation:
            This models uncertainty from acceleration errors and maneuvers.
        """
    I3 = np.eye(3)

    Q_pos_pos = (dt ** 3 / 3.0) * sigma2_zeta * I3
    Q_pos_vel = (dt ** 2 / 2.0) * sigma2_zeta * I3
    Q_vel_pos = (dt ** 2 / 2.0) * sigma2_zeta * I3
    Q_vel_vel = dt * sigma2_zeta * I3

    Q = np.block([
        [Q_pos_pos, Q_pos_vel],
        [Q_vel_pos, Q_vel_vel]
    ])

    return Q


def predict_cartesian_state(zeta_prev, dt=1.0):
    """
    Eq. (18) without random process noise:
        zeta_pred = F_zeta @ zeta_prev

    In PCRB computation, the process noise is represented by Phi_zeta,
    so here we usually only return the deterministic prediction.
    """
    F = state_transition_matrix(dt)
    return F @ zeta_prev


def measurement_function_h(zeta):
    """
    Eq. (21)-(22):
        y_t = h(zeta_t) + measurement_noise

    h(zeta) converts Cartesian state:
        zeta = [x, y, z, vx, vy, vz]

    into spherical measurement:
        [d, theta, phi, radial_velocity]

    Paper writes:
        phi = arctan(y/x)

    Code uses:
        phi = atan2(y, x)

    because atan2 is numerically safer and handles quadrant correctly.
    """
    x, y, z, vx, vy, vz = zeta

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if d <= DIST_EPS:
        raise ValueError("Target position cannot coincide with the BS.")

    theta = np.arccos(np.clip(z / d, -1.0, 1.0))

    phi = np.arctan2(y, x)

    radial_velocity = (vx * x + vy * y + vz * z) / d

    return np.array([d, theta, phi, radial_velocity], dtype=float)


def numerical_jacobian(func, x, eps=1e-5):
    """
    Numerically compute Jacobian matrix.

    For Eq. (26):
        H_zeta = partial h(zeta) / partial zeta

    If:
        h: R^6 -> R^4

    then:
        H has shape (4, 6)

    This is easier for beginners than manually deriving the analytic Jacobian.
    """
    x = np.asarray(x, dtype=float)
    y0 = func(x)

    H = np.zeros((len(y0), len(x)), dtype=float)

    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()

        xp[i] += eps
        xm[i] -= eps

        yp = func(xp)
        ym = func(xm)

        diff = yp - ym

        # Optional: handle azimuth angle discontinuity for phi component.
        # phi is index 2 in [d, theta, phi, radial_velocity].
        diff[2] = (diff[2] + np.pi) % (2 * np.pi) - np.pi

        H[:, i] = diff / (2 * eps)

    return H


def symmetrize(A):
    """
    Numerical helper:
        make a matrix symmetric.

    FIM/PCRB should be symmetric theoretically,
    but numerical inversion may introduce tiny asymmetry.
    """
    return 0.5 * (A + A.T)


def safe_inverse(A):
    """
    Matrix inverse with a pseudo-inverse fallback for singular inputs.

    A fixed absolute jitter is deliberately avoided because the covariance
    entries in this model span many orders of magnitude; adding 1e-9, for
    example, can dominate a valid small angular variance.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if np.any(~np.isfinite(A)):
        raise ValueError("A must contain only finite values.")

    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def prior_fim_eq25(J_prev, F, Phi):
    """
    Eq. (25):
        J_prior =
            ( Phi + F @ inv(J_prev) @ F.T )^{-1}

    J_prev:
        posterior FIM from previous time slot

    inv(J_prev):
        previous PCRB / covariance lower bound
    """
    P_prev = safe_inverse(J_prev)
    P_prior = Phi + F @ P_prev @ F.T
    J_prior = safe_inverse(P_prior)

    return symmetrize(J_prior), symmetrize(P_prior)


def data_fim_eq26(H, Psi, use_inverse_covariance=True):
    """
    Eq. (26): data FIM.

    Standard Gaussian measurement FIM:
        J_data = H.T @ inv(Psi) @ H

    Note:
        The PDF text may visually appear as H.T @ Psi @ H.
        However, since Psi is a measurement covariance matrix,
        the standard FIM uses inverse covariance.

    use_inverse_covariance=True:
        recommended implementation.

    use_inverse_covariance=False:
        literal form if you want to test the PDF's displayed formula.
    """
    if use_inverse_covariance:
        Psi_inv = safe_inverse(Psi)
        J_data = H.T @ Psi_inv @ H
    else:
        J_data = H.T @ Psi @ H

    return symmetrize(J_data)


def pcrb_eq24_to_eq28_parts(
        zeta_prev,
        J_prev,
        Psi,
        dt=1.0,
        sigma2_zeta=0.1,
        use_inverse_covariance=True
):
    """
    Compute Eq. (17)-(28) in one clear block.

    Inputs:
        zeta_prev:
            previous Cartesian state estimate or true state for testing
            shape (6,)

        J_prev:
            previous posterior FIM
            shape (6, 6)

        Psi:
            measurement noise covariance from fused CRB
            shape (4, 4)

        dt:
            time slot duration

        sigma2_zeta:
            process noise intensity

    Outputs:
        dictionary containing all intermediate variables:
            F
            Phi
            zeta_pred
            y_pred
            H
            J_prior
            J_data
            J
            PCRB
            rho
    """
    # Eq. (19): state transition matrix
    F = state_transition_matrix(dt)

    # Eq. (20): process noise covariance
    Phi = process_noise_covariance(dt=dt, sigma2_zeta=sigma2_zeta)

    # Eq. (18): predicted Cartesian state
    zeta_pred = F @ zeta_prev

    # Eq. (21)-(22): predicted measurement
    y_pred = measurement_function_h(zeta_pred)

    # Jacobian H evaluated at predicted state
    H = numerical_jacobian(measurement_function_h, zeta_pred)

    # Eq. (25): prior FIM
    J_prior, P_prior = prior_fim_eq25(J_prev=J_prev, F=F, Phi=Phi)

    # Eq. (26): data FIM
    J_data = data_fim_eq26(
        H=H,
        Psi=Psi,
        use_inverse_covariance=use_inverse_covariance
    )

    # Eq. (24): posterior FIM
    J = J_prior + J_data
    J = symmetrize(J)

    # Eq. (27): PCRB
    PCRB = safe_inverse(J)
    PCRB = symmetrize(PCRB)

    # Eq. (28): trace(PCRB)
    rho = float(np.trace(PCRB))

    return {
        "F": F,
        "Phi": Phi,
        "zeta_pred": zeta_pred,
        "y_pred": y_pred,
        "H": H,
        "Psi": Psi,
        "P_prior": P_prior,
        "J_prior": J_prior,
        "J_data": J_data,
        "J": J,
        "PCRB": PCRB,
        "rho": rho
    }


# ============================================================
# 7. main sanity test
# ============================================================

def print_parts(title, parts):
    """
    Pretty print SINR parts.
    """
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    for key, value in parts.items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.6e}")
        else:
            print(f"{key:32s}: {value}")


def main():
    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # 1. Parameters from the paper simulation setup
    # --------------------------------------------------------
    Mx = 8
    My = 8
    P = 5.0
    bandwidth = 100e6
    antenna_spacing_ratio = 0.5  # dx = dy = wavelength / 2

    beta0_c = db_to_linear(-50)  # communication reference gain, -50 dB
    beta0_s = db_to_linear(-50)  # sensing reference gain, -50 dB

    K = 10.0  # Rician K-factor

    alpha0 = 0.9  # RCS of sensed target
    alpha1 = 0.9  # RCS of UAVs

    sigma0 = dbm_to_watt(-80)  # AWGN, -80 dBm
    sigma_ubs = dbm_to_watt(-70)  # BS residual interference, -70 dBm
    sigma_un = dbm_to_watt(-70)  # UAV residual interference, -70 dBm

    gamma_min_db = 5.0
    gamma_min_linear = 10 ** (gamma_min_db / 10)

    kappa_d = 1.0
    kappa_theta = 1e-6
    kappa_phi = 1e-6
    kappa_v = 1e-6

    max_uav_speed = 5.0
    min_uav_distance = 5.0
    process_noise_intensity = 0.1
    target_acceleration = np.array([0.02, -0.02, 0.02])
    total_time_slots = 100
    slot_duration = 1.0

    # --------------------------------------------------------
    # 2. Positions
    # --------------------------------------------------------
    u_bs = np.array([0.0, 0.0, 0.0])

    uav_positions = [
        np.array([30.0, 10.0, 70.0]),  # UAV 1
        np.array([30.0, -10.0, 70.0])  # UAV 2
    ]

    target = np.array([60.0, 0.0, 60.0])
    target_velocity = np.array([-1.0, 1.0, -1.0])

    N = len(uav_positions)
    if N != 2:
        raise RuntimeError("This baseline is intentionally fixed to two UAVs.")

    # --------------------------------------------------------
    # 3. Create initial matched beams
    # --------------------------------------------------------
    w0, W_comm = make_matched_beams(
        u_bs=u_bs,
        uav_positions=uav_positions,
        target=target,
        Mx=Mx,
        My=My,
        P=P
    )

    total_power = np.linalg.norm(w0) ** 2
    total_power += sum(np.linalg.norm(wk) ** 2 for wk in W_comm)

    # Paper Eq. (49) column order: W = [w1, w2, w0].
    W = np.hstack([*W_comm, w0])
    print("\nInitial setup")
    print("-" * 72)
    print(f"Number of UAVs N              : {N}")
    print(f"BS antenna array Mx x My       : {Mx} x {My}")
    print(f"Each beam shape                : {w0.shape}")
    print(f"Beam matrix W=[w1,w2,w0]       : {W.shape}")
    print(f"Adjacent antenna spacing       : {antenna_spacing_ratio:.1f} wavelength")
    print(f"Total transmit power P         : {P:.6f} W")
    print(f"Power check sum ||w||^2        : {total_power:.6f} W")
    print(f"Communication threshold gamma  : {gamma_min_db:.2f} dB")
    print(f"ISAC bandwidth b               : {bandwidth:.6e} Hz")
    print(f"Maximum UAV speed              : {max_uav_speed:.2f} m/s")
    print(f"Minimum UAV distance           : {min_uav_distance:.2f} m")
    print(f"Target acceleration            : {target_acceleration} m/s^2")
    print(f"Process noise intensity        : {process_noise_intensity:.2f} m^2/s^3")
    print(f"Total slots / slot duration    : {total_time_slots} / {slot_duration:.1f} s")
    print(f"beta0_c                        : {beta0_c:.6e}")
    print(f"beta0_s                        : {beta0_s:.6e}")
    print(f"sigma0                         : {sigma0:.6e} W")
    print(f"sigma_ubs                      : {sigma_ubs:.6e} W")
    print(f"sigma_un                       : {sigma_un:.6e} W")

    # --------------------------------------------------------
    # 4. Eq. (6): BS sensing SINR
    # --------------------------------------------------------
    bs_parts = bs_sensing_sinr_eq6_parts(
        u_bs=u_bs,
        target=target,
        W_comm=W_comm,
        w0=w0,
        Mx=Mx,
        My=My,
        beta0_s=beta0_s,
        K=K,
        alpha0=alpha0,
        sigma_ubs=sigma_ubs,
        sigma0=sigma0,
        rng=rng
    )

    print_parts("[Eq. (6)] BS sensing SINR", bs_parts)

    # --------------------------------------------------------
    # 5. Eq. (10): UAV communication SINR
    # --------------------------------------------------------
    for n_idx in range(N):
        comm_parts = uav_comm_sinr_eq10_parts(
            u_bs=u_bs,
            uav_positions=uav_positions,
            target=target,
            n_idx=n_idx,
            W_comm=W_comm,
            w0=w0,
            Mx=Mx,
            My=My,
            beta0_c=beta0_c,
            beta0_s=beta0_s,
            K=K,
            alpha0=alpha0,
            alpha1=alpha1,
            sigma0=sigma0,
            rng=rng,
            include_direct_sensing_beam=False
        )

        feasible = comm_parts["sinr"] >= gamma_min_linear

        print_parts(
            f"[Eq. (10)] UAV {n_idx + 1} communication SINR "
            f"(constraint >= {gamma_min_db:.1f} dB? {feasible})",
            comm_parts
        )

    # --------------------------------------------------------
    # 6. Eq. (12): UAV sensing SINR
    # --------------------------------------------------------
    uav_sensing_parts_list = []

    for n_idx, uav in enumerate(uav_positions):
        sensing_parts = uav_sensing_sinr_eq12_parts(
            u_bs=u_bs,
            uav=uav,
            target=target,
            W_comm=W_comm,
            w0=w0,
            Mx=Mx,
            My=My,
            beta0_s=beta0_s,
            K=K,
            alpha0=alpha0,
            sigma_un=sigma_un,
            sigma0=sigma0,
            rng=rng
        )

        uav_sensing_parts_list.append(sensing_parts)

        print_parts(
            f"[Eq. (12)] UAV {n_idx + 1} sensing SINR",
            sensing_parts
        )

    # --------------------------------------------------------
    # 7. Eq. (13)-(16): Sensing information fusion
    # --------------------------------------------------------

    # Eq. (13): example measurement vector from target state
    zeta_example = make_cartesian_state(*target, *target_velocity)
    c_t = cartesian_state_to_measurement(zeta_example)

    print("\n" + "=" * 72)
    print("[Eq. (13)] Measurement vector c_t = [d, theta, phi, v]")
    print("=" * 72)
    print(f"c_t: {c_t}")

    # Collect sensing SINRs from BS and UAVs
    bs_sensing_sinr = bs_parts["sinr"]

    uav_sensing_sinrs = [
        parts["sinr"]
        for parts in uav_sensing_parts_list
    ]

    all_sensing_sinrs = [bs_sensing_sinr] + uav_sensing_sinrs

    # Eq. (14): SINR -> CRB for each sensing node
    crb_list = []

    for node_idx, sinr_s in enumerate(all_sensing_sinrs):
        crb = crb_from_sensing_sinr(
            sensing_sinr=sinr_s,
            bandwidth=bandwidth,
            kappa_d=kappa_d,
            kappa_theta=kappa_theta,
            kappa_phi=kappa_phi,
            kappa_v=kappa_v
        )

        crb_list.append(crb)

        node_name = "BS" if node_idx == 0 else f"UAV {node_idx}"

        print("\n" + "-" * 72)
        print(f"[Eq. (14)] CRB from {node_name}")
        print("-" * 72)
        print(f"sensing SINR linear: {sinr_s:.6e}")
        print(
            f"sensing SINR dB    : "
            f"{10 * np.log10(max(sinr_s, FLOAT_TINY)):.6f} dB"
        )
        print(f"sigma2_d           : {crb[0]:.6e}")
        print(f"sigma2_theta       : {crb[1]:.6e}")
        print(f"sigma2_phi         : {crb[2]:.6e}")
        print(f"sigma2_v           : {crb[3]:.6e}")

    # Eq. (15)-(16): CI fusion
    weights = np.ones(len(crb_list)) / len(crb_list)

    fused_crb = ci_fusion_crb(
        crb_list=crb_list,
        weights=weights
    )

    Psi_t = measurement_noise_cov_from_fused_crb(fused_crb)

    print("\n" + "=" * 72)
    print("[Eq. (15)-(16)] CI fused CRB")
    print("=" * 72)
    print(f"weights             : {weights}")
    print(f"fused sigma2_d      : {fused_crb[0]:.6e}")
    print(f"fused sigma2_theta  : {fused_crb[1]:.6e}")
    print(f"fused sigma2_phi    : {fused_crb[2]:.6e}")
    print(f"fused sigma2_v      : {fused_crb[3]:.6e}")

    print("\nMeasurement noise covariance Psi_t for later PCRB/EKF:")
    print(Psi_t)

    # --------------------------------------------------------
    # 8. Eq. (17)-(28): one-slot PCRB baseline
    # --------------------------------------------------------
    # The paper does not state an initial posterior covariance.  For a
    # reproducible sanity test, use 1 m^2 position variance and
    # 1 (m/s)^2 velocity variance at t=0.
    P0 = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    J0 = safe_inverse(P0)

    pcrb_parts = pcrb_eq24_to_eq28_parts(
        zeta_prev=zeta_example,
        J_prev=J0,
        Psi=Psi_t,
        dt=slot_duration,
        sigma2_zeta=process_noise_intensity,
        # Eq. (26) in the PDF omits the inverse on Psi.  Since Psi is a
        # covariance matrix, the physically consistent FIM uses Psi^{-1}.
        use_inverse_covariance=True
    )

    print("\n" + "=" * 72)
    print("[Eq. (17)-(28)] One-slot PCRB baseline")
    print("=" * 72)
    print(f"zeta_pred shape / value        : {pcrb_parts['zeta_pred'].shape} / "
          f"{pcrb_parts['zeta_pred']}")
    print(f"H shape                        : {pcrb_parts['H'].shape}")
    print(f"Psi shape                      : {pcrb_parts['Psi'].shape}")
    print(f"J shape                        : {pcrb_parts['J'].shape}")
    print(f"PCRB shape                     : {pcrb_parts['PCRB'].shape}")
    print(f"rho = trace(PCRB)              : {pcrb_parts['rho']:.6e}")


if __name__ == "__main__":
    main()
