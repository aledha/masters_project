import numpy

parameter = {
    "BSLmax": 0,
    "BSRmax": 1,
    "Beta0": 2,
    "Beta1": 3,
    "Buf_c": 4,
    "Buf_sr": 5,
    "Buf_ss": 6,
    "Ca_o": 7,
    "Cm": 8,
    "EC": 9,
    "F": 10,
    "K_NaCa": 11,
    "K_buf_c": 12,
    "K_buf_sr": 13,
    "K_buf_ss": 14,
    "K_mNa": 15,
    "K_mk": 16,
    "K_o": 17,
    "K_pCa": 18,
    "K_sat": 19,
    "K_up": 20,
    "KmBSL": 21,
    "KmBSR": 22,
    "Km_Ca": 23,
    "Km_Nai": 24,
    "Na_o": 25,
    "P_NaK": 26,
    "P_kna": 27,
    "R": 28,
    "T": 29,
    "Tot_A": 30,
    "Tref": 31,
    "Trpn50": 32,
    "V_c": 33,
    "V_leak": 34,
    "V_rel": 35,
    "V_sr": 36,
    "V_ss": 37,
    "V_xfer": 38,
    "Vmax_up": 39,
    "alpha": 40,
    "calib": 41,
    "cat50_ref": 42,
    "cmdnmax": 43,
    "csqnmax": 44,
    "dLambda": 45,
    "emcoupling": 46,
    "etal": 47,
    "etas": 48,
    "g_CaL": 49,
    "g_K1": 50,
    "g_Kr": 51,
    "g_Ks": 52,
    "g_Na": 53,
    "g_bca": 54,
    "g_bna": 55,
    "g_pCa": 56,
    "g_pK": 57,
    "g_to": 58,
    "gamma": 59,
    "gammas": 60,
    "gammaw": 61,
    "isacs": 62,
    "k1_prime": 63,
    "k2_prime": 64,
    "k3": 65,
    "k4": 66,
    "kmcmdn": 67,
    "kmcsqn": 68,
    "kmtrpn": 69,
    "ktrpn": 70,
    "ku": 71,
    "kuw": 72,
    "kws": 73,
    "lmbda": 74,
    "max_sr": 75,
    "min_sr": 76,
    "mode": 77,
    "ntm": 78,
    "ntrpn": 79,
    "p_a": 80,
    "p_b": 81,
    "p_k": 82,
    "phi": 83,
    "rs": 84,
    "rw": 85,
    "scale_HF_cat50_ref": 86,
    "stim_amplitude": 87,
    "stim_duration": 88,
    "stim_period": 89,
    "stim_start": 90,
    "trpnmax": 91,
}


def parameter_index(name: str) -> int:
    """Return the index of the parameter with the given name

    Arguments
    ---------
    name : str
        The name of the parameter

    Returns
    -------
    int
        The index of the parameter

    Raises
    ------
    KeyError
        If the name is not a valid parameter
    """

    return parameter[name]


state = {
    "fCass": 0,
    "f": 1,
    "f2": 2,
    "r": 3,
    "s": 4,
    "Zetaw": 5,
    "XS": 6,
    "XW": 7,
    "TmB": 8,
    "Zetas": 9,
    "Na_i": 10,
    "h": 11,
    "j": 12,
    "m": 13,
    "Xr1": 14,
    "Xr2": 15,
    "Xs": 16,
    "d": 17,
    "R_prime": 18,
    "CaTrpn": 19,
    "Ca_i": 20,
    "K_i": 21,
    "V": 22,
    "Cd": 23,
    "Ca_SR": 24,
    "Ca_ss": 25,
}


def state_index(name: str) -> int:
    """Return the index of the state with the given name

    Arguments
    ---------
    name : str
        The name of the state

    Returns
    -------
    int
        The index of the state

    Raises
    ------
    KeyError
        If the name is not a valid state
    """

    return state[name]


monitor = {
    "Aw": 0,
    "CaTrpn_max": 1,
    "E_Ca": 2,
    "E_K": 3,
    "E_Ks": 4,
    "E_Na": 5,
    "fCass_inf": 6,
    "tau_fCass": 7,
    "XW_max": 8,
    "XS_max": 9,
    "XU": 10,
    "alpha_d": 11,
    "alpha_h": 12,
    "alpha_j": 13,
    "alpha_m": 14,
    "alpha_xr1": 15,
    "alpha_xr2": 16,
    "alpha_xs": 17,
    "beta_d": 18,
    "beta_h": 19,
    "beta_j": 20,
    "beta_m": 21,
    "beta_xr1": 22,
    "beta_xr2": 23,
    "beta_xs": 24,
    "d_inf": 25,
    "f2_inf": 26,
    "f_inf": 27,
    "gamma_d": 28,
    "h_inf": 29,
    "j_inf": 30,
    "m_inf": 31,
    "r_inf": 32,
    "s_inf": 33,
    "tau_f": 34,
    "tau_f2": 35,
    "tau_r": 36,
    "tau_s": 37,
    "xr1_inf": 38,
    "xr2_inf": 39,
    "xs_inf": 40,
    "ksu": 41,
    "cs": 42,
    "cw": 43,
    "kwu": 44,
    "f_JCa_i_free": 45,
    "f_JCa_sr_free": 46,
    "f_JCa_ss_free": 47,
    "gammasu": 48,
    "gammawu": 49,
    "i_CaL": 50,
    "i_NaCa": 51,
    "i_NaK": 52,
    "i_Stim": 53,
    "i_leak": 54,
    "i_p_Ca": 55,
    "i_up": 56,
    "i_xfer": 57,
    "kb": 58,
    "kcasr": 59,
    "lambda_min12": 60,
    "As": 61,
    "i_b_Ca": 62,
    "alpha_K1": 63,
    "beta_K1": 64,
    "i_Kr": 65,
    "i_p_K": 66,
    "i_to": 67,
    "i_Ks": 68,
    "i_Na": 69,
    "i_b_Na": 70,
    "dfCass_dt": 71,
    "tau_h": 72,
    "tau_j": 73,
    "tau_m": 74,
    "tau_xr1": 75,
    "tau_xr2": 76,
    "tau_xs": 77,
    "tau_d": 78,
    "df_dt": 79,
    "df2_dt": 80,
    "dr_dt": 81,
    "ds_dt": 82,
    "dZetaw_dt": 83,
    "dXS_dt": 84,
    "dXW_dt": 85,
    "dTmB_dt": 86,
    "k1": 87,
    "k2": 88,
    "C": 89,
    "cat50": 90,
    "lambda_min087": 91,
    "dZetas_dt": 92,
    "ddt_Ca_i_total": 93,
    "xK1_inf": 94,
    "dNa_i_dt": 95,
    "dh_dt": 96,
    "dj_dt": 97,
    "dm_dt": 98,
    "dXr1_dt": 99,
    "dXr2_dt": 100,
    "dXs_dt": 101,
    "dd_dt": 102,
    "O": 103,
    "dR_prime_dt": 104,
    "F1": 105,
    "dCd": 106,
    "dCaTrpn_dt": 107,
    "h_lambda_prima": 108,
    "dCa_i_dt": 109,
    "i_K1": 110,
    "i_rel": 111,
    "eta": 112,
    "J_TRPN": 113,
    "h_lambda": 114,
    "dK_i_dt": 115,
    "dV_dt": 116,
    "ddt_Ca_sr_total": 117,
    "ddt_Ca_ss_total": 118,
    "Fd": 119,
    "dCd_dt": 120,
    "Ta": 121,
    "dCa_SR_dt": 122,
    "dCa_ss_dt": 123,
    "Tp": 124,
    "Ttot": 125,
}


def monitor_index(name: str) -> int:
    """Return the index of the monitor with the given name

    Arguments
    ---------
    name : str
        The name of the monitor

    Returns
    -------
    int
        The index of the monitor

    Raises
    ------
    KeyError
        If the name is not a valid monitor
    """

    return monitor[name]


def init_parameter_values(**values):
    """Initialize parameter values"""
    # BSLmax=1.124, BSRmax=0.047, Beta0=2.3, Beta1=-2.4, Buf_c=0.2
    # Buf_sr=10.0, Buf_ss=0.4, Ca_o=2.0, Cm=185.0, EC=1.5, F=96.485
    # K_NaCa=1000.0, K_buf_c=0.001, K_buf_sr=0.3, K_buf_ss=0.00025
    # K_mNa=40.0, K_mk=1.0, K_o=5.4, K_pCa=0.0005, K_sat=0.1
    # K_up=0.00025, KmBSL=0.0087, KmBSR=0.00087, Km_Ca=1.38
    # Km_Nai=87.5, Na_o=140.0, P_NaK=2.724, P_kna=0.03, R=8.314
    # T=310.0, Tot_A=25, Tref=120, Trpn50=0.35, V_c=16404.0
    # V_leak=0.00036, V_rel=0.102, V_sr=1094.0, V_ss=54.68
    # V_xfer=0.0038, Vmax_up=0.006375, alpha=2.5, calib=1
    # cat50_ref=0.805, cmdnmax=0.05, csqnmax=10.0, dLambda=0
    # emcoupling=1, etal=200, etas=20, g_CaL=0.0398, g_K1=5.405
    # g_Kr=0.153, g_Ks=0.392, g_Na=14.838, g_bca=0.000592
    # g_bna=0.00029, g_pCa=0.1238, g_pK=0.0146, g_to=0.294
    # gamma=0.35, gammas=0.0085, gammaw=0.615, isacs=0
    # k1_prime=0.15, k2_prime=0.045, k3=0.06, k4=0.005
    # kmcmdn=0.00238, kmcsqn=0.8, kmtrpn=0.0005, ktrpn=0.1, ku=0.04
    # kuw=0.182, kws=0.012, lmbda=1, max_sr=2.5, min_sr=1, mode=1
    # ntm=2.4, ntrpn=2, p_a=2.1, p_b=9.1, p_k=7, phi=2.23, rs=0.25
    # rw=0.5, scale_HF_cat50_ref=1.0, stim_amplitude=-52.0
    # stim_duration=1.0, stim_period=1000.0, stim_start=10.0
    # trpnmax=0.07

    parameters = numpy.array(
        [
            1.124,
            0.047,
            2.3,
            -2.4,
            0.2,
            10.0,
            0.4,
            2.0,
            185.0,
            1.5,
            96.485,
            1000.0,
            0.001,
            0.3,
            0.00025,
            40.0,
            1.0,
            5.4,
            0.0005,
            0.1,
            0.00025,
            0.0087,
            0.00087,
            1.38,
            87.5,
            140.0,
            2.724,
            0.03,
            8.314,
            310.0,
            25,
            120,
            0.35,
            16404.0,
            0.00036,
            0.102,
            1094.0,
            54.68,
            0.0038,
            0.006375,
            2.5,
            1,
            0.805,
            0.05,
            10.0,
            0,
            1,
            200,
            20,
            0.0398,
            5.405,
            0.153,
            0.392,
            14.838,
            0.000592,
            0.00029,
            0.1238,
            0.0146,
            0.294,
            0.35,
            0.0085,
            0.615,
            0,
            0.15,
            0.045,
            0.06,
            0.005,
            0.00238,
            0.8,
            0.0005,
            0.1,
            0.04,
            0.182,
            0.012,
            1,
            2.5,
            1,
            1,
            2.4,
            2,
            2.1,
            9.1,
            7,
            2.23,
            0.25,
            0.5,
            1.0,
            -52.0,
            1.0,
            1000.0,
            10.0,
            0.07,
        ],
        dtype=numpy.float64,
    )

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # fCass=0.9953, f=0.7888, f2=0.9755, r=2.42e-08, s=0.999998
    # Zetaw=0, XS=0, XW=0, TmB=1, Zetas=0, Na_i=8.604, h=0.7444
    # j=0.7045, m=0.00172, Xr1=0.00621, Xr2=0.4712, Xs=0.0095
    # d=3.373e-05, R_prime=0.9073, CaTrpn=0.0001, Ca_i=0.000126
    # K_i=136.89, V=-85.23, Cd=0, Ca_SR=3.64, Ca_ss=0.00036

    states = numpy.array(
        [
            0.9953,
            0.7888,
            0.9755,
            2.42e-08,
            0.999998,
            0,
            0,
            0,
            1,
            0,
            8.604,
            0.7444,
            0.7045,
            0.00172,
            0.00621,
            0.4712,
            0.0095,
            3.373e-05,
            0.9073,
            0.0001,
            0.000126,
            136.89,
            -85.23,
            0,
            3.64,
            0.00036,
        ],
        dtype=numpy.float64,
    )

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters):

    # Assign states
    fCass = states[0]
    f = states[1]
    f2 = states[2]
    r = states[3]
    s = states[4]
    Zetaw = states[5]
    XS = states[6]
    XW = states[7]
    TmB = states[8]
    Zetas = states[9]
    Na_i = states[10]
    h = states[11]
    j = states[12]
    m = states[13]
    Xr1 = states[14]
    Xr2 = states[15]
    Xs = states[16]
    d = states[17]
    R_prime = states[18]
    CaTrpn = states[19]
    Ca_i = states[20]
    K_i = states[21]
    V = states[22]
    Cd = states[23]
    Ca_SR = states[24]
    Ca_ss = states[25]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    Buf_c = parameters[4]
    Buf_sr = parameters[5]
    Buf_ss = parameters[6]
    Ca_o = parameters[7]
    Cm = parameters[8]
    EC = parameters[9]
    F = parameters[10]
    K_NaCa = parameters[11]
    K_buf_c = parameters[12]
    K_buf_sr = parameters[13]
    K_buf_ss = parameters[14]
    K_mNa = parameters[15]
    K_mk = parameters[16]
    K_o = parameters[17]
    K_pCa = parameters[18]
    K_sat = parameters[19]
    K_up = parameters[20]
    KmBSL = parameters[21]
    KmBSR = parameters[22]
    Km_Ca = parameters[23]
    Km_Nai = parameters[24]
    Na_o = parameters[25]
    P_NaK = parameters[26]
    P_kna = parameters[27]
    R = parameters[28]
    T = parameters[29]
    Tot_A = parameters[30]
    Tref = parameters[31]
    Trpn50 = parameters[32]
    V_c = parameters[33]
    V_leak = parameters[34]
    V_rel = parameters[35]
    V_sr = parameters[36]
    V_ss = parameters[37]
    V_xfer = parameters[38]
    Vmax_up = parameters[39]
    alpha = parameters[40]
    calib = parameters[41]
    cat50_ref = parameters[42]
    cmdnmax = parameters[43]
    csqnmax = parameters[44]
    dLambda = parameters[45]
    emcoupling = parameters[46]
    etal = parameters[47]
    etas = parameters[48]
    g_CaL = parameters[49]
    g_K1 = parameters[50]
    g_Kr = parameters[51]
    g_Ks = parameters[52]
    g_Na = parameters[53]
    g_bca = parameters[54]
    g_bna = parameters[55]
    g_pCa = parameters[56]
    g_pK = parameters[57]
    g_to = parameters[58]
    gamma = parameters[59]
    gammas = parameters[60]
    gammaw = parameters[61]
    isacs = parameters[62]
    k1_prime = parameters[63]
    k2_prime = parameters[64]
    k3 = parameters[65]
    k4 = parameters[66]
    kmcmdn = parameters[67]
    kmcsqn = parameters[68]
    kmtrpn = parameters[69]
    ktrpn = parameters[70]
    ku = parameters[71]
    kuw = parameters[72]
    kws = parameters[73]
    lmbda = parameters[74]
    max_sr = parameters[75]
    min_sr = parameters[76]
    mode = parameters[77]
    ntm = parameters[78]
    ntrpn = parameters[79]
    p_a = parameters[80]
    p_b = parameters[81]
    p_k = parameters[82]
    phi = parameters[83]
    rs = parameters[84]
    rw = parameters[85]
    scale_HF_cat50_ref = parameters[86]
    stim_amplitude = parameters[87]
    stim_duration = parameters[88]
    stim_period = parameters[89]
    stim_start = parameters[90]
    trpnmax = parameters[91]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
    E_Ca = (((0.5 * R) * T) / F) * numpy.log(Ca_o / Ca_i)
    E_K = ((R * T) / F) * numpy.log(K_o / K_i)
    E_Ks = ((R * T) / F) * numpy.log((K_o + Na_o * P_kna) / (K_i + Na_i * P_kna))
    E_Na = ((R * T) / F) * numpy.log(Na_o / Na_i)
    fCass_inf = 0.4 + 0.6 / ((Ca_ss / 0.05) ** 2 + 1)
    tau_fCass = 2 + 80 / ((Ca_ss / 0.05) ** 2 + 1)
    XW_max = numpy.where((XW > 0), XW, 0)
    XS_max = numpy.where((XS > 0), XS, 0)
    XU = -XW + (-XS + (1 - TmB))
    alpha_d = 0.25 + 1.4 / (numpy.exp((-V - 35) / 13) + 1)
    alpha_h = numpy.where((V < -40), 0.057 * numpy.exp((-(V + 80)) / 6.8), 0)
    alpha_j = numpy.where(
        (V < -40),
        (
            (
                (V + 37.78)
                * (
                    (-25428) * numpy.exp(0.2444 * V)
                    - 6.948e-06 * numpy.exp((-0.04391) * V)
                )
            )
            / 1
        )
        / (numpy.exp(0.311 * (V + 79.23)) + 1),
        0,
    )
    alpha_m = 1 / (numpy.exp((-V - 60) / 5) + 1)
    alpha_xr1 = 450 / (numpy.exp((-V - 45) / 10) + 1)
    alpha_xr2 = 3 / (numpy.exp((-V - 60) / 20) + 1)
    alpha_xs = 1400 / numpy.sqrt(numpy.exp((5 - V) / 6) + 1)
    beta_d = 1.4 / (numpy.exp((V + 5) / 5) + 1)
    beta_h = numpy.where(
        (V < -40),
        2.7 * numpy.exp(0.079 * V) + 310000 * numpy.exp(0.3485 * V),
        0.77 / ((0.13 * (numpy.exp((V + 10.66) / ((-11.1))) + 1))),
    )
    beta_j = numpy.where(
        (V < -40),
        (0.02424 * numpy.exp((-0.01052) * V))
        / (numpy.exp((-0.1378) * (V + 40.14)) + 1),
        (0.6 * numpy.exp(0.057 * V)) / (numpy.exp((-0.1) * (V + 32)) + 1),
    )
    beta_m = 0.1 / (numpy.exp((V + 35) / 5) + 1) + 0.1 / (numpy.exp((V - 50) / 200) + 1)
    beta_xr1 = 6 / (numpy.exp((V + 30) / 11.5) + 1)
    beta_xr2 = 1.12 / (numpy.exp((V - 60) / 20) + 1)
    beta_xs = 1 / (numpy.exp((V - 35) / 15) + 1)
    d_inf = 1 / (numpy.exp((-V - 8) / 7.5) + 1)
    f2_inf = 0.33 + 0.67 / (numpy.exp((V + 35) / 7) + 1)
    f_inf = 1 / (numpy.exp((V + 20) / 7) + 1)
    gamma_d = 1 / (numpy.exp((50 - V) / 20) + 1)
    h_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    j_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    m_inf = 1 / (numpy.exp((-V - 56.86) / 9.03) + 1) ** 2
    r_inf = 1 / (numpy.exp((20 - V) / 6) + 1)
    s_inf = 1 / (numpy.exp((V + 20) / 5) + 1)
    tau_f = (
        (
            1102.5 * numpy.exp((-((V + 27) ** 2)) / 225)
            + 200 / (numpy.exp((13 - V) / 10) + 1)
        )
        + 180 / (numpy.exp((V + 30) / 10) + 1)
    ) + 20
    tau_f2 = (
        562 * numpy.exp((-((V + 27) ** 2)) / 240) + 31 / (numpy.exp((25 - V) / 10) + 1)
    ) + 80 / (numpy.exp((V + 30) / 10) + 1)
    tau_r = 9.5 * numpy.exp((-((V + 40) ** 2)) / 1800) + 0.8
    tau_s = (
        85 * numpy.exp((-((V + 45) ** 2)) / 320) + 5 / (numpy.exp((V - 20) / 5) + 1)
    ) + 3
    xr1_inf = 1 / (numpy.exp((-V - 26) / 7) + 1)
    xr2_inf = 1 / (numpy.exp((V + 88) / 24) + 1)
    xs_inf = 1 / (numpy.exp((-V - 5) / 14) + 1)
    ksu = (kws * rw) * (-1 + 1 / rs)
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    kwu = kuw * (-1 + 1 / rw) - kws
    f_JCa_i_free = 1 / ((Buf_c * K_buf_c) / (Ca_i + K_buf_c) ** 2 + 1)
    f_JCa_sr_free = 1 / ((Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr) ** 2 + 1)
    f_JCa_ss_free = 1 / ((Buf_ss * K_buf_ss) / (Ca_ss + K_buf_ss) ** 2 + 1)
    gammasu = gammas * numpy.where(
        numpy.logical_or((Zetas > 0), numpy.logical_and((Zetas > -1), (Zetas < -1))),
        numpy.where((Zetas > 0), Zetas, 0),
        numpy.where((Zetas < -1), -Zetas - 1, 0),
    )
    gammawu = gammaw * numpy.abs(Zetaw)
    i_CaL = (
        ((F**2 * ((4 * (fCass * (f2 * (f * (d * g_CaL))))) * (V - 15))) / ((R * T)))
        * (-Ca_o + (0.25 * Ca_ss) * numpy.exp((F * (2 * (V - 15))) / ((R * T))))
    ) / (numpy.exp((F * (2 * (V - 15))) / ((R * T))) - 1)
    i_NaCa = (
        K_NaCa
        * (
            Ca_o * (Na_i**3 * numpy.exp((F * (V * gamma)) / ((R * T))))
            - alpha * Ca_i * (Na_o**3 * numpy.exp((F * (V * (gamma - 1))) / ((R * T))))
        )
    ) / (
        (
            ((Ca_o + Km_Ca) * (Km_Nai**3 + Na_o**3))
            * (K_sat * numpy.exp((F * (V * (gamma - 1))) / ((R * T))) + 1)
        )
    )
    i_NaK = ((Na_i * ((K_o * P_NaK) / (K_mk + K_o))) / (K_mNa + Na_i)) / (
        (0.1245 * numpy.exp((F * ((-0.1) * V)) / ((R * T))) + 1)
        + 0.0353 * numpy.exp((F * (-V)) / ((R * T)))
    )
    i_Stim = numpy.where(
        numpy.logical_and(
            (stim_start <= -stim_period * numpy.floor(t / stim_period) + t),
            (
                stim_duration + stim_start
                >= -stim_period * numpy.floor(t / stim_period) + t
            ),
        ),
        stim_amplitude,
        0,
    )
    i_leak = V_leak * (Ca_SR - Ca_i)
    i_p_Ca = (Ca_i * g_pCa) / (Ca_i + K_pCa)
    i_up = Vmax_up / (1 + K_up**2 / Ca_i**2)
    i_xfer = V_xfer * (-Ca_i + Ca_ss)
    kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
    kcasr = max_sr - (max_sr - min_sr) / ((EC / Ca_SR) ** 2 + 1)
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    As = Aw
    i_b_Ca = g_bca * (-E_Ca + V)
    alpha_K1 = 0.1 / (numpy.exp(0.06 * ((-E_K + V) - 200)) + 1)
    beta_K1 = (
        3 * numpy.exp(0.0002 * ((-E_K + V) + 100)) + numpy.exp(0.1 * ((-E_K + V) - 10))
    ) / (numpy.exp((-0.5) * (-E_K + V)) + 1)
    i_Kr = (Xr2 * (Xr1 * ((0.4303314829119352 * numpy.sqrt(K_o)) * g_Kr))) * (-E_K + V)
    i_p_K = (g_pK * (-E_K + V)) / (numpy.exp((25 - V) / 5.98) + 1)
    i_to = (s * (g_to * r)) * (-E_K + V)
    i_Ks = (Xs**2 * g_Ks) * (-E_Ks + V)
    i_Na = (j * (h * (g_Na * m**3))) * (-E_Na + V)
    i_b_Na = g_bna * (-E_Na + V)
    dfCass_dt = (-fCass + fCass_inf) / tau_fCass
    values[0] = dfCass_dt
    tau_h = 1 / (alpha_h + beta_h)
    tau_j = 1 / (alpha_j + beta_j)
    tau_m = (1 * alpha_m) * beta_m
    tau_xr1 = (1 * alpha_xr1) * beta_xr1
    tau_xr2 = (1 * alpha_xr2) * beta_xr2
    tau_xs = (1 * alpha_xs) * beta_xs + 80
    tau_d = (1 * alpha_d) * beta_d + gamma_d
    df_dt = (-f + f_inf) / tau_f
    values[1] = df_dt
    df2_dt = (-f2 + f2_inf) / tau_f2
    values[2] = df2_dt
    dr_dt = (-r + r_inf) / tau_r
    values[3] = dr_dt
    ds_dt = (-s + s_inf) / tau_s
    values[4] = ds_dt
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    values[5] = dZetaw_dt
    dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
    values[6] = dXS_dt
    dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
    values[7] = dXW_dt
    dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
        kb
        * numpy.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )
    values[8] = dTmB_dt
    k1 = k1_prime / kcasr
    k2 = k2_prime * kcasr
    C = lambda_min12 - 1
    cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    dZetas_dt = As * dLambda - Zetas * cs
    values[9] = dZetas_dt
    ddt_Ca_i_total = i_xfer + (
        (Cm * (-(-2 * i_NaCa + (i_b_Ca + i_p_Ca)))) / ((F * (2 * V_c)))
        + (V_sr * (i_leak - i_up)) / V_c
    )
    xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    dNa_i_dt = Cm * ((-(3 * i_NaCa + (3 * i_NaK + (i_Na + i_b_Na)))) / ((F * V_c)))
    values[10] = dNa_i_dt
    dh_dt = (-h + h_inf) / tau_h
    values[11] = dh_dt
    dj_dt = (-j + j_inf) / tau_j
    values[12] = dj_dt
    dm_dt = (-m + m_inf) / tau_m
    values[13] = dm_dt
    dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1
    values[14] = dXr1_dt
    dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2
    values[15] = dXr2_dt
    dXs_dt = (-Xs + xs_inf) / tau_xs
    values[16] = dXs_dt
    dd_dt = (-d + d_inf) / tau_d
    values[17] = dd_dt
    O = (R_prime * (Ca_ss**2 * k1)) / (Ca_ss**2 * k1 + k3)
    dR_prime_dt = R_prime * (Ca_ss * (-k2)) + k4 * (1 - R_prime)
    values[18] = dR_prime_dt
    F1 = numpy.exp(C * p_b) - 1
    dCd = C - Cd
    dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * Ca_i) / cat50) ** ntrpn * (1 - CaTrpn))
    values[19] = dCaTrpn_dt
    h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
    dCa_i_dt = ddt_Ca_i_total * f_JCa_i_free
    values[20] = dCa_i_dt
    i_K1 = ((0.4303314829119352 * numpy.sqrt(K_o)) * (g_K1 * xK1_inf)) * (-E_K + V)
    i_rel = (O * V_rel) * (Ca_SR - Ca_ss)
    eta = numpy.where((dCd < 0), etas, etal)
    J_TRPN = dCaTrpn_dt * trpnmax
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    dK_i_dt = Cm * (
        (-(-2 * i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))
        / ((F * V_c))
    )
    values[21] = dK_i_dt
    dV_dt = -(
        i_Stim
        + (
            i_p_Ca
            + (
                i_p_K
                + (
                    i_b_Ca
                    + (
                        i_NaCa
                        + (
                            i_b_Na
                            + (
                                i_Na
                                + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to)))))
                            )
                        )
                    )
                )
            )
        )
    )
    values[22] = dV_dt
    ddt_Ca_sr_total = i_up - (i_leak + i_rel)
    ddt_Ca_ss_total = ((Cm * (-i_CaL)) / ((F * (2 * V_ss))) + (V_sr * i_rel) / V_ss) - (
        V_c * i_xfer
    ) / V_ss
    Fd = dCd * eta
    dCd_dt = (p_k * (C - Cd)) / eta
    values[23] = dCd_dt
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    dCa_SR_dt = ddt_Ca_sr_total * f_JCa_sr_free
    values[24] = dCa_SR_dt
    dCa_ss_dt = ddt_Ca_ss_total * f_JCa_ss_free
    values[25] = dCa_ss_dt
    Tp = p_a * (F1 + Fd)
    Ttot = Ta + Tp

    return values


def monitor_values(t, states, parameters):

    # Assign states
    fCass = states[0]
    f = states[1]
    f2 = states[2]
    r = states[3]
    s = states[4]
    Zetaw = states[5]
    XS = states[6]
    XW = states[7]
    TmB = states[8]
    Zetas = states[9]
    Na_i = states[10]
    h = states[11]
    j = states[12]
    m = states[13]
    Xr1 = states[14]
    Xr2 = states[15]
    Xs = states[16]
    d = states[17]
    R_prime = states[18]
    CaTrpn = states[19]
    Ca_i = states[20]
    K_i = states[21]
    V = states[22]
    Cd = states[23]
    Ca_SR = states[24]
    Ca_ss = states[25]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    Buf_c = parameters[4]
    Buf_sr = parameters[5]
    Buf_ss = parameters[6]
    Ca_o = parameters[7]
    Cm = parameters[8]
    EC = parameters[9]
    F = parameters[10]
    K_NaCa = parameters[11]
    K_buf_c = parameters[12]
    K_buf_sr = parameters[13]
    K_buf_ss = parameters[14]
    K_mNa = parameters[15]
    K_mk = parameters[16]
    K_o = parameters[17]
    K_pCa = parameters[18]
    K_sat = parameters[19]
    K_up = parameters[20]
    KmBSL = parameters[21]
    KmBSR = parameters[22]
    Km_Ca = parameters[23]
    Km_Nai = parameters[24]
    Na_o = parameters[25]
    P_NaK = parameters[26]
    P_kna = parameters[27]
    R = parameters[28]
    T = parameters[29]
    Tot_A = parameters[30]
    Tref = parameters[31]
    Trpn50 = parameters[32]
    V_c = parameters[33]
    V_leak = parameters[34]
    V_rel = parameters[35]
    V_sr = parameters[36]
    V_ss = parameters[37]
    V_xfer = parameters[38]
    Vmax_up = parameters[39]
    alpha = parameters[40]
    calib = parameters[41]
    cat50_ref = parameters[42]
    cmdnmax = parameters[43]
    csqnmax = parameters[44]
    dLambda = parameters[45]
    emcoupling = parameters[46]
    etal = parameters[47]
    etas = parameters[48]
    g_CaL = parameters[49]
    g_K1 = parameters[50]
    g_Kr = parameters[51]
    g_Ks = parameters[52]
    g_Na = parameters[53]
    g_bca = parameters[54]
    g_bna = parameters[55]
    g_pCa = parameters[56]
    g_pK = parameters[57]
    g_to = parameters[58]
    gamma = parameters[59]
    gammas = parameters[60]
    gammaw = parameters[61]
    isacs = parameters[62]
    k1_prime = parameters[63]
    k2_prime = parameters[64]
    k3 = parameters[65]
    k4 = parameters[66]
    kmcmdn = parameters[67]
    kmcsqn = parameters[68]
    kmtrpn = parameters[69]
    ktrpn = parameters[70]
    ku = parameters[71]
    kuw = parameters[72]
    kws = parameters[73]
    lmbda = parameters[74]
    max_sr = parameters[75]
    min_sr = parameters[76]
    mode = parameters[77]
    ntm = parameters[78]
    ntrpn = parameters[79]
    p_a = parameters[80]
    p_b = parameters[81]
    p_k = parameters[82]
    phi = parameters[83]
    rs = parameters[84]
    rw = parameters[85]
    scale_HF_cat50_ref = parameters[86]
    stim_amplitude = parameters[87]
    stim_duration = parameters[88]
    stim_period = parameters[89]
    stim_start = parameters[90]
    trpnmax = parameters[91]

    # Assign expressions
    shape = 126 if len(states.shape) == 1 else (126, states.shape[1])
    values = numpy.zeros(shape)
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    values[0] = Aw
    CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
    values[1] = CaTrpn_max
    E_Ca = (((0.5 * R) * T) / F) * numpy.log(Ca_o / Ca_i)
    values[2] = E_Ca
    E_K = ((R * T) / F) * numpy.log(K_o / K_i)
    values[3] = E_K
    E_Ks = ((R * T) / F) * numpy.log((K_o + Na_o * P_kna) / (K_i + Na_i * P_kna))
    values[4] = E_Ks
    E_Na = ((R * T) / F) * numpy.log(Na_o / Na_i)
    values[5] = E_Na
    fCass_inf = 0.4 + 0.6 / ((Ca_ss / 0.05) ** 2 + 1)
    values[6] = fCass_inf
    tau_fCass = 2 + 80 / ((Ca_ss / 0.05) ** 2 + 1)
    values[7] = tau_fCass
    XW_max = numpy.where((XW > 0), XW, 0)
    values[8] = XW_max
    XS_max = numpy.where((XS > 0), XS, 0)
    values[9] = XS_max
    XU = -XW + (-XS + (1 - TmB))
    values[10] = XU
    alpha_d = 0.25 + 1.4 / (numpy.exp((-V - 35) / 13) + 1)
    values[11] = alpha_d
    alpha_h = numpy.where((V < -40), 0.057 * numpy.exp((-(V + 80)) / 6.8), 0)
    values[12] = alpha_h
    alpha_j = numpy.where(
        (V < -40),
        (
            (
                (V + 37.78)
                * (
                    (-25428) * numpy.exp(0.2444 * V)
                    - 6.948e-06 * numpy.exp((-0.04391) * V)
                )
            )
            / 1
        )
        / (numpy.exp(0.311 * (V + 79.23)) + 1),
        0,
    )
    values[13] = alpha_j
    alpha_m = 1 / (numpy.exp((-V - 60) / 5) + 1)
    values[14] = alpha_m
    alpha_xr1 = 450 / (numpy.exp((-V - 45) / 10) + 1)
    values[15] = alpha_xr1
    alpha_xr2 = 3 / (numpy.exp((-V - 60) / 20) + 1)
    values[16] = alpha_xr2
    alpha_xs = 1400 / numpy.sqrt(numpy.exp((5 - V) / 6) + 1)
    values[17] = alpha_xs
    beta_d = 1.4 / (numpy.exp((V + 5) / 5) + 1)
    values[18] = beta_d
    beta_h = numpy.where(
        (V < -40),
        2.7 * numpy.exp(0.079 * V) + 310000 * numpy.exp(0.3485 * V),
        0.77 / ((0.13 * (numpy.exp((V + 10.66) / ((-11.1))) + 1))),
    )
    values[19] = beta_h
    beta_j = numpy.where(
        (V < -40),
        (0.02424 * numpy.exp((-0.01052) * V))
        / (numpy.exp((-0.1378) * (V + 40.14)) + 1),
        (0.6 * numpy.exp(0.057 * V)) / (numpy.exp((-0.1) * (V + 32)) + 1),
    )
    values[20] = beta_j
    beta_m = 0.1 / (numpy.exp((V + 35) / 5) + 1) + 0.1 / (numpy.exp((V - 50) / 200) + 1)
    values[21] = beta_m
    beta_xr1 = 6 / (numpy.exp((V + 30) / 11.5) + 1)
    values[22] = beta_xr1
    beta_xr2 = 1.12 / (numpy.exp((V - 60) / 20) + 1)
    values[23] = beta_xr2
    beta_xs = 1 / (numpy.exp((V - 35) / 15) + 1)
    values[24] = beta_xs
    d_inf = 1 / (numpy.exp((-V - 8) / 7.5) + 1)
    values[25] = d_inf
    f2_inf = 0.33 + 0.67 / (numpy.exp((V + 35) / 7) + 1)
    values[26] = f2_inf
    f_inf = 1 / (numpy.exp((V + 20) / 7) + 1)
    values[27] = f_inf
    gamma_d = 1 / (numpy.exp((50 - V) / 20) + 1)
    values[28] = gamma_d
    h_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    values[29] = h_inf
    j_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    values[30] = j_inf
    m_inf = 1 / (numpy.exp((-V - 56.86) / 9.03) + 1) ** 2
    values[31] = m_inf
    r_inf = 1 / (numpy.exp((20 - V) / 6) + 1)
    values[32] = r_inf
    s_inf = 1 / (numpy.exp((V + 20) / 5) + 1)
    values[33] = s_inf
    tau_f = (
        (
            1102.5 * numpy.exp((-((V + 27) ** 2)) / 225)
            + 200 / (numpy.exp((13 - V) / 10) + 1)
        )
        + 180 / (numpy.exp((V + 30) / 10) + 1)
    ) + 20
    values[34] = tau_f
    tau_f2 = (
        562 * numpy.exp((-((V + 27) ** 2)) / 240) + 31 / (numpy.exp((25 - V) / 10) + 1)
    ) + 80 / (numpy.exp((V + 30) / 10) + 1)
    values[35] = tau_f2
    tau_r = 9.5 * numpy.exp((-((V + 40) ** 2)) / 1800) + 0.8
    values[36] = tau_r
    tau_s = (
        85 * numpy.exp((-((V + 45) ** 2)) / 320) + 5 / (numpy.exp((V - 20) / 5) + 1)
    ) + 3
    values[37] = tau_s
    xr1_inf = 1 / (numpy.exp((-V - 26) / 7) + 1)
    values[38] = xr1_inf
    xr2_inf = 1 / (numpy.exp((V + 88) / 24) + 1)
    values[39] = xr2_inf
    xs_inf = 1 / (numpy.exp((-V - 5) / 14) + 1)
    values[40] = xs_inf
    ksu = (kws * rw) * (-1 + 1 / rs)
    values[41] = ksu
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    values[42] = cs
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    values[43] = cw
    kwu = kuw * (-1 + 1 / rw) - kws
    values[44] = kwu
    f_JCa_i_free = 1 / ((Buf_c * K_buf_c) / (Ca_i + K_buf_c) ** 2 + 1)
    values[45] = f_JCa_i_free
    f_JCa_sr_free = 1 / ((Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr) ** 2 + 1)
    values[46] = f_JCa_sr_free
    f_JCa_ss_free = 1 / ((Buf_ss * K_buf_ss) / (Ca_ss + K_buf_ss) ** 2 + 1)
    values[47] = f_JCa_ss_free
    gammasu = gammas * numpy.where(
        numpy.logical_or((Zetas > 0), numpy.logical_and((Zetas > -1), (Zetas < -1))),
        numpy.where((Zetas > 0), Zetas, 0),
        numpy.where((Zetas < -1), -Zetas - 1, 0),
    )
    values[48] = gammasu
    gammawu = gammaw * numpy.abs(Zetaw)
    values[49] = gammawu
    i_CaL = (
        ((F**2 * ((4 * (fCass * (f2 * (f * (d * g_CaL))))) * (V - 15))) / ((R * T)))
        * (-Ca_o + (0.25 * Ca_ss) * numpy.exp((F * (2 * (V - 15))) / ((R * T))))
    ) / (numpy.exp((F * (2 * (V - 15))) / ((R * T))) - 1)
    values[50] = i_CaL
    i_NaCa = (
        K_NaCa
        * (
            Ca_o * (Na_i**3 * numpy.exp((F * (V * gamma)) / ((R * T))))
            - alpha * Ca_i * (Na_o**3 * numpy.exp((F * (V * (gamma - 1))) / ((R * T))))
        )
    ) / (
        (
            ((Ca_o + Km_Ca) * (Km_Nai**3 + Na_o**3))
            * (K_sat * numpy.exp((F * (V * (gamma - 1))) / ((R * T))) + 1)
        )
    )
    values[51] = i_NaCa
    i_NaK = ((Na_i * ((K_o * P_NaK) / (K_mk + K_o))) / (K_mNa + Na_i)) / (
        (0.1245 * numpy.exp((F * ((-0.1) * V)) / ((R * T))) + 1)
        + 0.0353 * numpy.exp((F * (-V)) / ((R * T)))
    )
    values[52] = i_NaK
    i_Stim = numpy.where(
        numpy.logical_and(
            (stim_start <= -stim_period * numpy.floor(t / stim_period) + t),
            (
                stim_duration + stim_start
                >= -stim_period * numpy.floor(t / stim_period) + t
            ),
        ),
        stim_amplitude,
        0,
    )
    values[53] = i_Stim
    i_leak = V_leak * (Ca_SR - Ca_i)
    values[54] = i_leak
    i_p_Ca = (Ca_i * g_pCa) / (Ca_i + K_pCa)
    values[55] = i_p_Ca
    i_up = Vmax_up / (1 + K_up**2 / Ca_i**2)
    values[56] = i_up
    i_xfer = V_xfer * (-Ca_i + Ca_ss)
    values[57] = i_xfer
    kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
    values[58] = kb
    kcasr = max_sr - (max_sr - min_sr) / ((EC / Ca_SR) ** 2 + 1)
    values[59] = kcasr
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    values[60] = lambda_min12
    As = Aw
    values[61] = As
    i_b_Ca = g_bca * (-E_Ca + V)
    values[62] = i_b_Ca
    alpha_K1 = 0.1 / (numpy.exp(0.06 * ((-E_K + V) - 200)) + 1)
    values[63] = alpha_K1
    beta_K1 = (
        3 * numpy.exp(0.0002 * ((-E_K + V) + 100)) + numpy.exp(0.1 * ((-E_K + V) - 10))
    ) / (numpy.exp((-0.5) * (-E_K + V)) + 1)
    values[64] = beta_K1
    i_Kr = (Xr2 * (Xr1 * ((0.4303314829119352 * numpy.sqrt(K_o)) * g_Kr))) * (-E_K + V)
    values[65] = i_Kr
    i_p_K = (g_pK * (-E_K + V)) / (numpy.exp((25 - V) / 5.98) + 1)
    values[66] = i_p_K
    i_to = (s * (g_to * r)) * (-E_K + V)
    values[67] = i_to
    i_Ks = (Xs**2 * g_Ks) * (-E_Ks + V)
    values[68] = i_Ks
    i_Na = (j * (h * (g_Na * m**3))) * (-E_Na + V)
    values[69] = i_Na
    i_b_Na = g_bna * (-E_Na + V)
    values[70] = i_b_Na
    dfCass_dt = (-fCass + fCass_inf) / tau_fCass
    values[71] = dfCass_dt
    tau_h = 1 / (alpha_h + beta_h)
    values[72] = tau_h
    tau_j = 1 / (alpha_j + beta_j)
    values[73] = tau_j
    tau_m = (1 * alpha_m) * beta_m
    values[74] = tau_m
    tau_xr1 = (1 * alpha_xr1) * beta_xr1
    values[75] = tau_xr1
    tau_xr2 = (1 * alpha_xr2) * beta_xr2
    values[76] = tau_xr2
    tau_xs = (1 * alpha_xs) * beta_xs + 80
    values[77] = tau_xs
    tau_d = (1 * alpha_d) * beta_d + gamma_d
    values[78] = tau_d
    df_dt = (-f + f_inf) / tau_f
    values[79] = df_dt
    df2_dt = (-f2 + f2_inf) / tau_f2
    values[80] = df2_dt
    dr_dt = (-r + r_inf) / tau_r
    values[81] = dr_dt
    ds_dt = (-s + s_inf) / tau_s
    values[82] = ds_dt
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    values[83] = dZetaw_dt
    dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
    values[84] = dXS_dt
    dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
    values[85] = dXW_dt
    dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
        kb
        * numpy.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )
    values[86] = dTmB_dt
    k1 = k1_prime / kcasr
    values[87] = k1
    k2 = k2_prime * kcasr
    values[88] = k2
    C = lambda_min12 - 1
    values[89] = C
    cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
    values[90] = cat50
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    values[91] = lambda_min087
    dZetas_dt = As * dLambda - Zetas * cs
    values[92] = dZetas_dt
    ddt_Ca_i_total = i_xfer + (
        (Cm * (-(-2 * i_NaCa + (i_b_Ca + i_p_Ca)))) / ((F * (2 * V_c)))
        + (V_sr * (i_leak - i_up)) / V_c
    )
    values[93] = ddt_Ca_i_total
    xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    values[94] = xK1_inf
    dNa_i_dt = Cm * ((-(3 * i_NaCa + (3 * i_NaK + (i_Na + i_b_Na)))) / ((F * V_c)))
    values[95] = dNa_i_dt
    dh_dt = (-h + h_inf) / tau_h
    values[96] = dh_dt
    dj_dt = (-j + j_inf) / tau_j
    values[97] = dj_dt
    dm_dt = (-m + m_inf) / tau_m
    values[98] = dm_dt
    dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1
    values[99] = dXr1_dt
    dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2
    values[100] = dXr2_dt
    dXs_dt = (-Xs + xs_inf) / tau_xs
    values[101] = dXs_dt
    dd_dt = (-d + d_inf) / tau_d
    values[102] = dd_dt
    O = (R_prime * (Ca_ss**2 * k1)) / (Ca_ss**2 * k1 + k3)
    values[103] = O
    dR_prime_dt = R_prime * (Ca_ss * (-k2)) + k4 * (1 - R_prime)
    values[104] = dR_prime_dt
    F1 = numpy.exp(C * p_b) - 1
    values[105] = F1
    dCd = C - Cd
    values[106] = dCd
    dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * Ca_i) / cat50) ** ntrpn * (1 - CaTrpn))
    values[107] = dCaTrpn_dt
    h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
    values[108] = h_lambda_prima
    dCa_i_dt = ddt_Ca_i_total * f_JCa_i_free
    values[109] = dCa_i_dt
    i_K1 = ((0.4303314829119352 * numpy.sqrt(K_o)) * (g_K1 * xK1_inf)) * (-E_K + V)
    values[110] = i_K1
    i_rel = (O * V_rel) * (Ca_SR - Ca_ss)
    values[111] = i_rel
    eta = numpy.where((dCd < 0), etas, etal)
    values[112] = eta
    J_TRPN = dCaTrpn_dt * trpnmax
    values[113] = J_TRPN
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    values[114] = h_lambda
    dK_i_dt = Cm * (
        (-(-2 * i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))
        / ((F * V_c))
    )
    values[115] = dK_i_dt
    dV_dt = -(
        i_Stim
        + (
            i_p_Ca
            + (
                i_p_K
                + (
                    i_b_Ca
                    + (
                        i_NaCa
                        + (
                            i_b_Na
                            + (
                                i_Na
                                + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to)))))
                            )
                        )
                    )
                )
            )
        )
    )
    values[116] = dV_dt
    ddt_Ca_sr_total = i_up - (i_leak + i_rel)
    values[117] = ddt_Ca_sr_total
    ddt_Ca_ss_total = ((Cm * (-i_CaL)) / ((F * (2 * V_ss))) + (V_sr * i_rel) / V_ss) - (
        V_c * i_xfer
    ) / V_ss
    values[118] = ddt_Ca_ss_total
    Fd = dCd * eta
    values[119] = Fd
    dCd_dt = (p_k * (C - Cd)) / eta
    values[120] = dCd_dt
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    values[121] = Ta
    dCa_SR_dt = ddt_Ca_sr_total * f_JCa_sr_free
    values[122] = dCa_SR_dt
    dCa_ss_dt = ddt_Ca_ss_total * f_JCa_ss_free
    values[123] = dCa_ss_dt
    Tp = p_a * (F1 + Fd)
    values[124] = Tp
    Ttot = Ta + Tp
    values[125] = Ttot

    return values


def generalized_rush_larsen(states, t, dt, parameters):

    # Assign states
    fCass = states[0]
    f = states[1]
    f2 = states[2]
    r = states[3]
    s = states[4]
    Zetaw = states[5]
    XS = states[6]
    XW = states[7]
    TmB = states[8]
    Zetas = states[9]
    Na_i = states[10]
    h = states[11]
    j = states[12]
    m = states[13]
    Xr1 = states[14]
    Xr2 = states[15]
    Xs = states[16]
    d = states[17]
    R_prime = states[18]
    CaTrpn = states[19]
    Ca_i = states[20]
    K_i = states[21]
    V = states[22]
    Cd = states[23]
    Ca_SR = states[24]
    Ca_ss = states[25]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    Buf_c = parameters[4]
    Buf_sr = parameters[5]
    Buf_ss = parameters[6]
    Ca_o = parameters[7]
    Cm = parameters[8]
    EC = parameters[9]
    F = parameters[10]
    K_NaCa = parameters[11]
    K_buf_c = parameters[12]
    K_buf_sr = parameters[13]
    K_buf_ss = parameters[14]
    K_mNa = parameters[15]
    K_mk = parameters[16]
    K_o = parameters[17]
    K_pCa = parameters[18]
    K_sat = parameters[19]
    K_up = parameters[20]
    KmBSL = parameters[21]
    KmBSR = parameters[22]
    Km_Ca = parameters[23]
    Km_Nai = parameters[24]
    Na_o = parameters[25]
    P_NaK = parameters[26]
    P_kna = parameters[27]
    R = parameters[28]
    T = parameters[29]
    Tot_A = parameters[30]
    Tref = parameters[31]
    Trpn50 = parameters[32]
    V_c = parameters[33]
    V_leak = parameters[34]
    V_rel = parameters[35]
    V_sr = parameters[36]
    V_ss = parameters[37]
    V_xfer = parameters[38]
    Vmax_up = parameters[39]
    alpha = parameters[40]
    calib = parameters[41]
    cat50_ref = parameters[42]
    cmdnmax = parameters[43]
    csqnmax = parameters[44]
    dLambda = parameters[45]
    emcoupling = parameters[46]
    etal = parameters[47]
    etas = parameters[48]
    g_CaL = parameters[49]
    g_K1 = parameters[50]
    g_Kr = parameters[51]
    g_Ks = parameters[52]
    g_Na = parameters[53]
    g_bca = parameters[54]
    g_bna = parameters[55]
    g_pCa = parameters[56]
    g_pK = parameters[57]
    g_to = parameters[58]
    gamma = parameters[59]
    gammas = parameters[60]
    gammaw = parameters[61]
    isacs = parameters[62]
    k1_prime = parameters[63]
    k2_prime = parameters[64]
    k3 = parameters[65]
    k4 = parameters[66]
    kmcmdn = parameters[67]
    kmcsqn = parameters[68]
    kmtrpn = parameters[69]
    ktrpn = parameters[70]
    ku = parameters[71]
    kuw = parameters[72]
    kws = parameters[73]
    lmbda = parameters[74]
    max_sr = parameters[75]
    min_sr = parameters[76]
    mode = parameters[77]
    ntm = parameters[78]
    ntrpn = parameters[79]
    p_a = parameters[80]
    p_b = parameters[81]
    p_k = parameters[82]
    phi = parameters[83]
    rs = parameters[84]
    rw = parameters[85]
    scale_HF_cat50_ref = parameters[86]
    stim_amplitude = parameters[87]
    stim_duration = parameters[88]
    stim_period = parameters[89]
    stim_start = parameters[90]
    trpnmax = parameters[91]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
    E_Ca = (((0.5 * R) * T) / F) * numpy.log(Ca_o / Ca_i)
    E_K = ((R * T) / F) * numpy.log(K_o / K_i)
    E_Ks = ((R * T) / F) * numpy.log((K_o + Na_o * P_kna) / (K_i + Na_i * P_kna))
    E_Na = ((R * T) / F) * numpy.log(Na_o / Na_i)
    fCass_inf = 0.4 + 0.6 / ((Ca_ss / 0.05) ** 2 + 1)
    tau_fCass = 2 + 80 / ((Ca_ss / 0.05) ** 2 + 1)
    XW_max = numpy.where((XW > 0), XW, 0)
    XS_max = numpy.where((XS > 0), XS, 0)
    XU = -XW + (-XS + (1 - TmB))
    alpha_d = 0.25 + 1.4 / (numpy.exp((-V - 35) / 13) + 1)
    alpha_h = numpy.where((V < -40), 0.057 * numpy.exp((-(V + 80)) / 6.8), 0)
    alpha_j = numpy.where(
        (V < -40),
        (
            (
                (V + 37.78)
                * (
                    (-25428) * numpy.exp(0.2444 * V)
                    - 6.948e-06 * numpy.exp((-0.04391) * V)
                )
            )
            / 1
        )
        / (numpy.exp(0.311 * (V + 79.23)) + 1),
        0,
    )
    alpha_m = 1 / (numpy.exp((-V - 60) / 5) + 1)
    alpha_xr1 = 450 / (numpy.exp((-V - 45) / 10) + 1)
    alpha_xr2 = 3 / (numpy.exp((-V - 60) / 20) + 1)
    alpha_xs = 1400 / numpy.sqrt(numpy.exp((5 - V) / 6) + 1)
    beta_d = 1.4 / (numpy.exp((V + 5) / 5) + 1)
    beta_h = numpy.where(
        (V < -40),
        2.7 * numpy.exp(0.079 * V) + 310000 * numpy.exp(0.3485 * V),
        0.77 / ((0.13 * (numpy.exp((V + 10.66) / ((-11.1))) + 1))),
    )
    beta_j = numpy.where(
        (V < -40),
        (0.02424 * numpy.exp((-0.01052) * V))
        / (numpy.exp((-0.1378) * (V + 40.14)) + 1),
        (0.6 * numpy.exp(0.057 * V)) / (numpy.exp((-0.1) * (V + 32)) + 1),
    )
    beta_m = 0.1 / (numpy.exp((V + 35) / 5) + 1) + 0.1 / (numpy.exp((V - 50) / 200) + 1)
    beta_xr1 = 6 / (numpy.exp((V + 30) / 11.5) + 1)
    beta_xr2 = 1.12 / (numpy.exp((V - 60) / 20) + 1)
    beta_xs = 1 / (numpy.exp((V - 35) / 15) + 1)
    d_inf = 1 / (numpy.exp((-V - 8) / 7.5) + 1)
    f2_inf = 0.33 + 0.67 / (numpy.exp((V + 35) / 7) + 1)
    f_inf = 1 / (numpy.exp((V + 20) / 7) + 1)
    gamma_d = 1 / (numpy.exp((50 - V) / 20) + 1)
    h_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    j_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    m_inf = 1 / (numpy.exp((-V - 56.86) / 9.03) + 1) ** 2
    r_inf = 1 / (numpy.exp((20 - V) / 6) + 1)
    s_inf = 1 / (numpy.exp((V + 20) / 5) + 1)
    tau_f = (
        (
            1102.5 * numpy.exp((-((V + 27) ** 2)) / 225)
            + 200 / (numpy.exp((13 - V) / 10) + 1)
        )
        + 180 / (numpy.exp((V + 30) / 10) + 1)
    ) + 20
    tau_f2 = (
        562 * numpy.exp((-((V + 27) ** 2)) / 240) + 31 / (numpy.exp((25 - V) / 10) + 1)
    ) + 80 / (numpy.exp((V + 30) / 10) + 1)
    tau_r = 9.5 * numpy.exp((-((V + 40) ** 2)) / 1800) + 0.8
    tau_s = (
        85 * numpy.exp((-((V + 45) ** 2)) / 320) + 5 / (numpy.exp((V - 20) / 5) + 1)
    ) + 3
    xr1_inf = 1 / (numpy.exp((-V - 26) / 7) + 1)
    xr2_inf = 1 / (numpy.exp((V + 88) / 24) + 1)
    xs_inf = 1 / (numpy.exp((-V - 5) / 14) + 1)
    ksu = (kws * rw) * (-1 + 1 / rs)
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    kwu = kuw * (-1 + 1 / rw) - kws
    f_JCa_i_free = 1 / ((Buf_c * K_buf_c) / (Ca_i + K_buf_c) ** 2 + 1)
    f_JCa_sr_free = 1 / ((Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr) ** 2 + 1)
    f_JCa_ss_free = 1 / ((Buf_ss * K_buf_ss) / (Ca_ss + K_buf_ss) ** 2 + 1)
    gammasu = gammas * numpy.where(
        numpy.logical_or((Zetas > 0), numpy.logical_and((Zetas > -1), (Zetas < -1))),
        numpy.where((Zetas > 0), Zetas, 0),
        numpy.where((Zetas < -1), -Zetas - 1, 0),
    )
    gammawu = gammaw * numpy.abs(Zetaw)
    i_CaL = (
        ((F**2 * ((4 * (fCass * (f2 * (f * (d * g_CaL))))) * (V - 15))) / ((R * T)))
        * (-Ca_o + (0.25 * Ca_ss) * numpy.exp((F * (2 * (V - 15))) / ((R * T))))
    ) / (numpy.exp((F * (2 * (V - 15))) / ((R * T))) - 1)
    i_NaCa = (
        K_NaCa
        * (
            Ca_o * (Na_i**3 * numpy.exp((F * (V * gamma)) / ((R * T))))
            - alpha * Ca_i * (Na_o**3 * numpy.exp((F * (V * (gamma - 1))) / ((R * T))))
        )
    ) / (
        (
            ((Ca_o + Km_Ca) * (Km_Nai**3 + Na_o**3))
            * (K_sat * numpy.exp((F * (V * (gamma - 1))) / ((R * T))) + 1)
        )
    )
    i_NaK = ((Na_i * ((K_o * P_NaK) / (K_mk + K_o))) / (K_mNa + Na_i)) / (
        (0.1245 * numpy.exp((F * ((-0.1) * V)) / ((R * T))) + 1)
        + 0.0353 * numpy.exp((F * (-V)) / ((R * T)))
    )
    i_Stim = numpy.where(
        numpy.logical_and(
            (stim_start <= -stim_period * numpy.floor(t / stim_period) + t),
            (
                stim_duration + stim_start
                >= -stim_period * numpy.floor(t / stim_period) + t
            ),
        ),
        stim_amplitude,
        0,
    )
    i_leak = V_leak * (Ca_SR - Ca_i)
    i_p_Ca = (Ca_i * g_pCa) / (Ca_i + K_pCa)
    i_up = Vmax_up / (1 + K_up**2 / Ca_i**2)
    i_xfer = V_xfer * (-Ca_i + Ca_ss)
    kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
    kcasr = max_sr - (max_sr - min_sr) / ((EC / Ca_SR) ** 2 + 1)
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    As = Aw
    i_b_Ca = g_bca * (-E_Ca + V)
    alpha_K1 = 0.1 / (numpy.exp(0.06 * ((-E_K + V) - 200)) + 1)
    beta_K1 = (
        3 * numpy.exp(0.0002 * ((-E_K + V) + 100)) + numpy.exp(0.1 * ((-E_K + V) - 10))
    ) / (numpy.exp((-0.5) * (-E_K + V)) + 1)
    i_Kr = (Xr2 * (Xr1 * ((0.4303314829119352 * numpy.sqrt(K_o)) * g_Kr))) * (-E_K + V)
    i_p_K = (g_pK * (-E_K + V)) / (numpy.exp((25 - V) / 5.98) + 1)
    i_to = (s * (g_to * r)) * (-E_K + V)
    i_Ks = (Xs**2 * g_Ks) * (-E_Ks + V)
    i_Na = (j * (h * (g_Na * m**3))) * (-E_Na + V)
    i_b_Na = g_bna * (-E_Na + V)
    dfCass_dt = (-fCass + fCass_inf) / tau_fCass
    dfCass_dt_linearized = -1 / tau_fCass
    values[0] = (
        dfCass_dt * (numpy.exp(dfCass_dt_linearized * dt) - 1) / dfCass_dt_linearized
        + fCass
    )
    tau_h = 1 / (alpha_h + beta_h)
    tau_j = 1 / (alpha_j + beta_j)
    tau_m = (1 * alpha_m) * beta_m
    tau_xr1 = (1 * alpha_xr1) * beta_xr1
    tau_xr2 = (1 * alpha_xr2) * beta_xr2
    tau_xs = (1 * alpha_xs) * beta_xs + 80
    tau_d = (1 * alpha_d) * beta_d + gamma_d
    df_dt = (-f + f_inf) / tau_f
    df_dt_linearized = -1 / tau_f
    values[1] = df_dt * (numpy.exp(df_dt_linearized * dt) - 1) / df_dt_linearized + f
    df2_dt = (-f2 + f2_inf) / tau_f2
    df2_dt_linearized = -1 / tau_f2
    values[2] = (
        df2_dt * (numpy.exp(df2_dt_linearized * dt) - 1) / df2_dt_linearized + f2
    )
    dr_dt = (-r + r_inf) / tau_r
    dr_dt_linearized = -1 / tau_r
    values[3] = dr_dt * (numpy.exp(dr_dt_linearized * dt) - 1) / dr_dt_linearized + r
    ds_dt = (-s + s_inf) / tau_s
    ds_dt_linearized = -1 / tau_s
    values[4] = ds_dt * (numpy.exp(ds_dt_linearized * dt) - 1) / ds_dt_linearized + s
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    dZetaw_dt_linearized = -cw
    values[5] = Zetaw + numpy.where(
        numpy.logical_or(
            (dZetaw_dt_linearized > 1e-08), (dZetaw_dt_linearized < -1e-08)
        ),
        dZetaw_dt * (numpy.exp(dZetaw_dt_linearized * dt) - 1) / dZetaw_dt_linearized,
        dZetaw_dt * dt,
    )
    dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
    dXS_dt_linearized = -gammasu - ksu
    values[6] = XS + numpy.where(
        numpy.logical_or((dXS_dt_linearized > 1e-08), (dXS_dt_linearized < -1e-08)),
        dXS_dt * (numpy.exp(dXS_dt_linearized * dt) - 1) / dXS_dt_linearized,
        dXS_dt * dt,
    )
    dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
    dXW_dt_linearized = -gammawu - kws - kwu
    values[7] = XW + numpy.where(
        numpy.logical_or((dXW_dt_linearized > 1e-08), (dXW_dt_linearized < -1e-08)),
        dXW_dt * (numpy.exp(dXW_dt_linearized * dt) - 1) / dXW_dt_linearized,
        dXW_dt * dt,
    )
    dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
        kb
        * numpy.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )
    dTmB_dt_linearized = -(CaTrpn ** (ntm / 2)) * ku
    values[8] = TmB + numpy.where(
        numpy.logical_or((dTmB_dt_linearized > 1e-08), (dTmB_dt_linearized < -1e-08)),
        dTmB_dt * (numpy.exp(dTmB_dt_linearized * dt) - 1) / dTmB_dt_linearized,
        dTmB_dt * dt,
    )
    k1 = k1_prime / kcasr
    k2 = k2_prime * kcasr
    C = lambda_min12 - 1
    cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    dZetas_dt = As * dLambda - Zetas * cs
    dZetas_dt_linearized = -cs
    values[9] = Zetas + numpy.where(
        numpy.logical_or(
            (dZetas_dt_linearized > 1e-08), (dZetas_dt_linearized < -1e-08)
        ),
        dZetas_dt * (numpy.exp(dZetas_dt_linearized * dt) - 1) / dZetas_dt_linearized,
        dZetas_dt * dt,
    )
    ddt_Ca_i_total = i_xfer + (
        (Cm * (-(-2 * i_NaCa + (i_b_Ca + i_p_Ca)))) / ((F * (2 * V_c)))
        + (V_sr * (i_leak - i_up)) / V_c
    )
    xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    dNa_i_dt = Cm * ((-(3 * i_NaCa + (3 * i_NaK + (i_Na + i_b_Na)))) / ((F * V_c)))
    values[10] = Na_i + dNa_i_dt * dt
    dh_dt = (-h + h_inf) / tau_h
    dh_dt_linearized = -1 / tau_h
    values[11] = dh_dt * (numpy.exp(dh_dt_linearized * dt) - 1) / dh_dt_linearized + h
    dj_dt = (-j + j_inf) / tau_j
    dj_dt_linearized = -1 / tau_j
    values[12] = dj_dt * (numpy.exp(dj_dt_linearized * dt) - 1) / dj_dt_linearized + j
    dm_dt = (-m + m_inf) / tau_m
    dm_dt_linearized = -1 / tau_m
    values[13] = dm_dt * (numpy.exp(dm_dt_linearized * dt) - 1) / dm_dt_linearized + m
    dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1
    dXr1_dt_linearized = -1 / tau_xr1
    values[14] = (
        Xr1 + dXr1_dt * (numpy.exp(dXr1_dt_linearized * dt) - 1) / dXr1_dt_linearized
    )
    dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2
    dXr2_dt_linearized = -1 / tau_xr2
    values[15] = (
        Xr2 + dXr2_dt * (numpy.exp(dXr2_dt_linearized * dt) - 1) / dXr2_dt_linearized
    )
    dXs_dt = (-Xs + xs_inf) / tau_xs
    dXs_dt_linearized = -1 / tau_xs
    values[16] = (
        Xs + dXs_dt * (numpy.exp(dXs_dt_linearized * dt) - 1) / dXs_dt_linearized
    )
    dd_dt = (-d + d_inf) / tau_d
    dd_dt_linearized = -1 / tau_d
    values[17] = d + dd_dt * (numpy.exp(dd_dt_linearized * dt) - 1) / dd_dt_linearized
    O = (R_prime * (Ca_ss**2 * k1)) / (Ca_ss**2 * k1 + k3)
    dR_prime_dt = R_prime * (Ca_ss * (-k2)) + k4 * (1 - R_prime)
    dR_prime_dt_linearized = Ca_ss * (-k2) - k4
    values[18] = R_prime + numpy.where(
        numpy.logical_or(
            (dR_prime_dt_linearized > 1e-08), (dR_prime_dt_linearized < -1e-08)
        ),
        dR_prime_dt
        * (numpy.exp(dR_prime_dt_linearized * dt) - 1)
        / dR_prime_dt_linearized,
        dR_prime_dt * dt,
    )
    F1 = numpy.exp(C * p_b) - 1
    dCd = C - Cd
    dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * Ca_i) / cat50) ** ntrpn * (1 - CaTrpn))
    dCaTrpn_dt_linearized = ktrpn * (-(((1000 * Ca_i) / cat50) ** ntrpn) - 1)
    values[19] = CaTrpn + numpy.where(
        numpy.logical_or(
            (dCaTrpn_dt_linearized > 1e-08), (dCaTrpn_dt_linearized < -1e-08)
        ),
        dCaTrpn_dt
        * (numpy.exp(dCaTrpn_dt_linearized * dt) - 1)
        / dCaTrpn_dt_linearized,
        dCaTrpn_dt * dt,
    )
    h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
    dCa_i_dt = ddt_Ca_i_total * f_JCa_i_free
    values[20] = Ca_i + dCa_i_dt * dt
    i_K1 = ((0.4303314829119352 * numpy.sqrt(K_o)) * (g_K1 * xK1_inf)) * (-E_K + V)
    i_rel = (O * V_rel) * (Ca_SR - Ca_ss)
    eta = numpy.where((dCd < 0), etas, etal)
    J_TRPN = dCaTrpn_dt * trpnmax
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    dK_i_dt = Cm * (
        (-(-2 * i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))
        / ((F * V_c))
    )
    values[21] = K_i + dK_i_dt * dt
    dV_dt = -(
        i_Stim
        + (
            i_p_Ca
            + (
                i_p_K
                + (
                    i_b_Ca
                    + (
                        i_NaCa
                        + (
                            i_b_Na
                            + (
                                i_Na
                                + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to)))))
                            )
                        )
                    )
                )
            )
        )
    )
    values[22] = V + dV_dt * dt
    ddt_Ca_sr_total = i_up - (i_leak + i_rel)
    ddt_Ca_ss_total = ((Cm * (-i_CaL)) / ((F * (2 * V_ss))) + (V_sr * i_rel) / V_ss) - (
        V_c * i_xfer
    ) / V_ss
    Fd = dCd * eta
    dCd_dt = (p_k * (C - Cd)) / eta
    dCd_dt_linearized = -p_k / eta
    values[23] = Cd + numpy.where(
        numpy.logical_or((dCd_dt_linearized > 1e-08), (dCd_dt_linearized < -1e-08)),
        dCd_dt * (numpy.exp(dCd_dt_linearized * dt) - 1) / dCd_dt_linearized,
        dCd_dt * dt,
    )
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    dCa_SR_dt = ddt_Ca_sr_total * f_JCa_sr_free
    values[24] = Ca_SR + dCa_SR_dt * dt
    dCa_ss_dt = ddt_Ca_ss_total * f_JCa_ss_free
    values[25] = Ca_ss + dCa_ss_dt * dt
    Tp = p_a * (F1 + Fd)
    Ttot = Ta + Tp

    return values
