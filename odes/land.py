import numpy

parameter = {
    "BSLmax": 0,
    "BSRmax": 1,
    "Beta0": 2,
    "Beta1": 3,
    "KmBSL": 4,
    "KmBSR": 5,
    "Tot_A": 6,
    "Tref": 7,
    "Trpn50": 8,
    "calib": 9,
    "cat50_ref": 10,
    "cmdnmax": 11,
    "csqnmax": 12,
    "dLambda": 13,
    "emcoupling": 14,
    "etal": 15,
    "etas": 16,
    "gammas": 17,
    "gammaw": 18,
    "isacs": 19,
    "kmcmdn": 20,
    "kmcsqn": 21,
    "kmtrpn": 22,
    "ktrpn": 23,
    "ku": 24,
    "kuw": 25,
    "kws": 26,
    "lmbda": 27,
    "mode": 28,
    "ntm": 29,
    "ntrpn": 30,
    "p_a": 31,
    "p_b": 32,
    "p_k": 33,
    "phi": 34,
    "rs": 35,
    "rw": 36,
    "scale_HF_cat50_ref": 37,
    "trpnmax": 38,
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
    "cai": 0,
    "Zetaw": 1,
    "XS": 2,
    "XW": 3,
    "TmB": 4,
    "Zetas": 5,
    "CaTrpn": 6,
    "Cd": 7,
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
    "dcai_dt": 0,
    "Aw": 1,
    "CaTrpn_max": 2,
    "XW_max": 3,
    "XS_max": 4,
    "XU": 5,
    "cs": 6,
    "ksu": 7,
    "cw": 8,
    "kwu": 9,
    "gammasu": 10,
    "gammawu": 11,
    "kb": 12,
    "lambda_min12": 13,
    "As": 14,
    "dZetaw_dt": 15,
    "dXS_dt": 16,
    "dXW_dt": 17,
    "dTmB_dt": 18,
    "C": 19,
    "cat50": 20,
    "lambda_min087": 21,
    "dZetas_dt": 22,
    "F1": 23,
    "dCd": 24,
    "dCaTrpn_dt": 25,
    "h_lambda_prima": 26,
    "eta": 27,
    "J_TRPN": 28,
    "h_lambda": 29,
    "Fd": 30,
    "dCd_dt": 31,
    "Ta": 32,
    "Tp": 33,
    "Ttot": 34,
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
    # BSLmax=1.124, BSRmax=0.047, Beta0=2.3, Beta1=-2.4
    # KmBSL=0.0087, KmBSR=0.00087, Tot_A=25, Tref=120, Trpn50=0.35
    # calib=1, cat50_ref=0.805, cmdnmax=0.05, csqnmax=10.0
    # dLambda=0, emcoupling=1, etal=200, etas=20, gammas=0.0085
    # gammaw=0.615, isacs=0, kmcmdn=0.00238, kmcsqn=0.8
    # kmtrpn=0.0005, ktrpn=0.1, ku=0.04, kuw=0.182, kws=0.012
    # lmbda=1, mode=1, ntm=2.4, ntrpn=2, p_a=2.1, p_b=9.1, p_k=7
    # phi=2.23, rs=0.25, rw=0.5, scale_HF_cat50_ref=1.0
    # trpnmax=0.07

    parameters = numpy.array(
        [
            1.124,
            0.047,
            2.3,
            -2.4,
            0.0087,
            0.00087,
            25,
            120,
            0.35,
            1,
            0.805,
            0.05,
            10.0,
            0,
            1,
            200,
            20,
            0.0085,
            0.615,
            0,
            0.00238,
            0.8,
            0.0005,
            0.1,
            0.04,
            0.182,
            0.012,
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
            0.07,
        ],
        dtype=numpy.float64,
    )

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # cai=0, Zetaw=0, XS=0, XW=0, TmB=1, Zetas=0, CaTrpn=0.0001
    # Cd=0

    states = numpy.array([0, 0, 0, 0, 1, 0, 0.0001, 0], dtype=numpy.float64)

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters):

    # Assign states
    cai = states[0]
    Zetaw = states[1]
    XS = states[2]
    XW = states[3]
    TmB = states[4]
    Zetas = states[5]
    CaTrpn = states[6]
    Cd = states[7]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    KmBSL = parameters[4]
    KmBSR = parameters[5]
    Tot_A = parameters[6]
    Tref = parameters[7]
    Trpn50 = parameters[8]
    calib = parameters[9]
    cat50_ref = parameters[10]
    cmdnmax = parameters[11]
    csqnmax = parameters[12]
    dLambda = parameters[13]
    emcoupling = parameters[14]
    etal = parameters[15]
    etas = parameters[16]
    gammas = parameters[17]
    gammaw = parameters[18]
    isacs = parameters[19]
    kmcmdn = parameters[20]
    kmcsqn = parameters[21]
    kmtrpn = parameters[22]
    ktrpn = parameters[23]
    ku = parameters[24]
    kuw = parameters[25]
    kws = parameters[26]
    lmbda = parameters[27]
    mode = parameters[28]
    ntm = parameters[29]
    ntrpn = parameters[30]
    p_a = parameters[31]
    p_b = parameters[32]
    p_k = parameters[33]
    phi = parameters[34]
    rs = parameters[35]
    rw = parameters[36]
    scale_HF_cat50_ref = parameters[37]
    trpnmax = parameters[38]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    dcai_dt = 0
    values[0] = dcai_dt
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
    XW_max = numpy.where((XW > 0), XW, 0)
    XS_max = numpy.where((XS > 0), XS, 0)
    XU = -XW + (-XS + (1 - TmB))
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    ksu = (kws * rw) * (-1 + 1 / rs)
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    kwu = kuw * (-1 + 1 / rw) - kws
    gammasu = gammas * numpy.where(
        numpy.logical_or((Zetas > 0), numpy.logical_and((Zetas > -1), (Zetas < -1))),
        numpy.where((Zetas > 0), Zetas, 0),
        numpy.where((Zetas < -1), -Zetas - 1, 0),
    )
    gammawu = gammaw * numpy.abs(Zetaw)
    kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    As = Aw
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    values[1] = dZetaw_dt
    dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
    values[2] = dXS_dt
    dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
    values[3] = dXW_dt
    dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
        kb
        * numpy.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )
    values[4] = dTmB_dt
    C = lambda_min12 - 1
    cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    dZetas_dt = As * dLambda - Zetas * cs
    values[5] = dZetas_dt
    F1 = numpy.exp(C * p_b) - 1
    dCd = C - Cd
    dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * cai) / cat50) ** ntrpn * (1 - CaTrpn))
    values[6] = dCaTrpn_dt
    h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
    eta = numpy.where((dCd < 0), etas, etal)
    J_TRPN = dCaTrpn_dt * trpnmax
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    Fd = dCd * eta
    dCd_dt = (p_k * (C - Cd)) / eta
    values[7] = dCd_dt
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    Tp = p_a * (F1 + Fd)
    Ttot = Ta + Tp

    return values


def monitor_values(t, states, parameters):

    # Assign states
    cai = states[0]
    Zetaw = states[1]
    XS = states[2]
    XW = states[3]
    TmB = states[4]
    Zetas = states[5]
    CaTrpn = states[6]
    Cd = states[7]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    KmBSL = parameters[4]
    KmBSR = parameters[5]
    Tot_A = parameters[6]
    Tref = parameters[7]
    Trpn50 = parameters[8]
    calib = parameters[9]
    cat50_ref = parameters[10]
    cmdnmax = parameters[11]
    csqnmax = parameters[12]
    dLambda = parameters[13]
    emcoupling = parameters[14]
    etal = parameters[15]
    etas = parameters[16]
    gammas = parameters[17]
    gammaw = parameters[18]
    isacs = parameters[19]
    kmcmdn = parameters[20]
    kmcsqn = parameters[21]
    kmtrpn = parameters[22]
    ktrpn = parameters[23]
    ku = parameters[24]
    kuw = parameters[25]
    kws = parameters[26]
    lmbda = parameters[27]
    mode = parameters[28]
    ntm = parameters[29]
    ntrpn = parameters[30]
    p_a = parameters[31]
    p_b = parameters[32]
    p_k = parameters[33]
    phi = parameters[34]
    rs = parameters[35]
    rw = parameters[36]
    scale_HF_cat50_ref = parameters[37]
    trpnmax = parameters[38]

    # Assign expressions
    shape = 35 if len(states.shape) == 1 else (35, states.shape[1])
    values = numpy.zeros(shape)
    dcai_dt = 0
    values[0] = dcai_dt
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    values[1] = Aw
    CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
    values[2] = CaTrpn_max
    XW_max = numpy.where((XW > 0), XW, 0)
    values[3] = XW_max
    XS_max = numpy.where((XS > 0), XS, 0)
    values[4] = XS_max
    XU = -XW + (-XS + (1 - TmB))
    values[5] = XU
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    values[6] = cs
    ksu = (kws * rw) * (-1 + 1 / rs)
    values[7] = ksu
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    values[8] = cw
    kwu = kuw * (-1 + 1 / rw) - kws
    values[9] = kwu
    gammasu = gammas * numpy.where(
        numpy.logical_or((Zetas > 0), numpy.logical_and((Zetas > -1), (Zetas < -1))),
        numpy.where((Zetas > 0), Zetas, 0),
        numpy.where((Zetas < -1), -Zetas - 1, 0),
    )
    values[10] = gammasu
    gammawu = gammaw * numpy.abs(Zetaw)
    values[11] = gammawu
    kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
    values[12] = kb
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    values[13] = lambda_min12
    As = Aw
    values[14] = As
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    values[15] = dZetaw_dt
    dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
    values[16] = dXS_dt
    dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
    values[17] = dXW_dt
    dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
        kb
        * numpy.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )
    values[18] = dTmB_dt
    C = lambda_min12 - 1
    values[19] = C
    cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
    values[20] = cat50
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    values[21] = lambda_min087
    dZetas_dt = As * dLambda - Zetas * cs
    values[22] = dZetas_dt
    F1 = numpy.exp(C * p_b) - 1
    values[23] = F1
    dCd = C - Cd
    values[24] = dCd
    dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * cai) / cat50) ** ntrpn * (1 - CaTrpn))
    values[25] = dCaTrpn_dt
    h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
    values[26] = h_lambda_prima
    eta = numpy.where((dCd < 0), etas, etal)
    values[27] = eta
    J_TRPN = dCaTrpn_dt * trpnmax
    values[28] = J_TRPN
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    values[29] = h_lambda
    Fd = dCd * eta
    values[30] = Fd
    dCd_dt = (p_k * (C - Cd)) / eta
    values[31] = dCd_dt
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    values[32] = Ta
    Tp = p_a * (F1 + Fd)
    values[33] = Tp
    Ttot = Ta + Tp
    values[34] = Ttot

    return values


def generalized_rush_larsen(states, t, dt, parameters):

    # Assign states
    cai = states[0]
    Zetaw = states[1]
    XS = states[2]
    XW = states[3]
    TmB = states[4]
    Zetas = states[5]
    CaTrpn = states[6]
    Cd = states[7]

    # Assign parameters
    BSLmax = parameters[0]
    BSRmax = parameters[1]
    Beta0 = parameters[2]
    Beta1 = parameters[3]
    KmBSL = parameters[4]
    KmBSR = parameters[5]
    Tot_A = parameters[6]
    Tref = parameters[7]
    Trpn50 = parameters[8]
    calib = parameters[9]
    cat50_ref = parameters[10]
    cmdnmax = parameters[11]
    csqnmax = parameters[12]
    dLambda = parameters[13]
    emcoupling = parameters[14]
    etal = parameters[15]
    etas = parameters[16]
    gammas = parameters[17]
    gammaw = parameters[18]
    isacs = parameters[19]
    kmcmdn = parameters[20]
    kmcsqn = parameters[21]
    kmtrpn = parameters[22]
    ktrpn = parameters[23]
    ku = parameters[24]
    kuw = parameters[25]
    kws = parameters[26]
    lmbda = parameters[27]
    mode = parameters[28]
    ntm = parameters[29]
    ntrpn = parameters[30]
    p_a = parameters[31]
    p_b = parameters[32]
    p_k = parameters[33]
    phi = parameters[34]
    rs = parameters[35]
    rw = parameters[36]
    scale_HF_cat50_ref = parameters[37]
    trpnmax = parameters[38]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    dcai_dt = 0
    values[0] = cai + dcai_dt * dt
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
    XW_max = numpy.where((XW > 0), XW, 0)
    XS_max = numpy.where((XS > 0), XS, 0)
    XU = -XW + (-XS + (1 - TmB))
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    ksu = (kws * rw) * (-1 + 1 / rs)
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    kwu = kuw * (-1 + 1 / rw) - kws
    gammasu = gammas * numpy.where(
        numpy.logical_or((Zetas > 0), numpy.logical_and((Zetas > -1), (Zetas < -1))),
        numpy.where((Zetas > 0), Zetas, 0),
        numpy.where((Zetas < -1), -Zetas - 1, 0),
    )
    gammawu = gammaw * numpy.abs(Zetaw)
    kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    As = Aw
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    dZetaw_dt_linearized = -cw
    values[1] = Zetaw + numpy.where(
        numpy.logical_or(
            (dZetaw_dt_linearized > 1e-08), (dZetaw_dt_linearized < -1e-08)
        ),
        dZetaw_dt * (numpy.exp(dZetaw_dt_linearized * dt) - 1) / dZetaw_dt_linearized,
        dZetaw_dt * dt,
    )
    dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
    dXS_dt_linearized = -gammasu - ksu
    values[2] = XS + numpy.where(
        numpy.logical_or((dXS_dt_linearized > 1e-08), (dXS_dt_linearized < -1e-08)),
        dXS_dt * (numpy.exp(dXS_dt_linearized * dt) - 1) / dXS_dt_linearized,
        dXS_dt * dt,
    )
    dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
    dXW_dt_linearized = -gammawu - kws - kwu
    values[3] = XW + numpy.where(
        numpy.logical_or((dXW_dt_linearized > 1e-08), (dXW_dt_linearized < -1e-08)),
        dXW_dt * (numpy.exp(dXW_dt_linearized * dt) - 1) / dXW_dt_linearized,
        dXW_dt * dt,
    )
    dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
        kb
        * numpy.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )
    dTmB_dt_linearized = -(CaTrpn ** (ntm / 2)) * ku
    values[4] = TmB + numpy.where(
        numpy.logical_or((dTmB_dt_linearized > 1e-08), (dTmB_dt_linearized < -1e-08)),
        dTmB_dt * (numpy.exp(dTmB_dt_linearized * dt) - 1) / dTmB_dt_linearized,
        dTmB_dt * dt,
    )
    C = lambda_min12 - 1
    cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    dZetas_dt = As * dLambda - Zetas * cs
    dZetas_dt_linearized = -cs
    values[5] = Zetas + numpy.where(
        numpy.logical_or(
            (dZetas_dt_linearized > 1e-08), (dZetas_dt_linearized < -1e-08)
        ),
        dZetas_dt * (numpy.exp(dZetas_dt_linearized * dt) - 1) / dZetas_dt_linearized,
        dZetas_dt * dt,
    )
    F1 = numpy.exp(C * p_b) - 1
    dCd = C - Cd
    dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * cai) / cat50) ** ntrpn * (1 - CaTrpn))
    dCaTrpn_dt_linearized = ktrpn * (-(((1000 * cai) / cat50) ** ntrpn) - 1)
    values[6] = CaTrpn + numpy.where(
        numpy.logical_or(
            (dCaTrpn_dt_linearized > 1e-08), (dCaTrpn_dt_linearized < -1e-08)
        ),
        dCaTrpn_dt
        * (numpy.exp(dCaTrpn_dt_linearized * dt) - 1)
        / dCaTrpn_dt_linearized,
        dCaTrpn_dt * dt,
    )
    h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
    eta = numpy.where((dCd < 0), etas, etal)
    J_TRPN = dCaTrpn_dt * trpnmax
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    Fd = dCd * eta
    dCd_dt = (p_k * (C - Cd)) / eta
    dCd_dt_linearized = -p_k / eta
    values[7] = Cd + numpy.where(
        numpy.logical_or((dCd_dt_linearized > 1e-08), (dCd_dt_linearized < -1e-08)),
        dCd_dt * (numpy.exp(dCd_dt_linearized * dt) - 1) / dCd_dt_linearized,
        dCd_dt * dt,
    )
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    Tp = p_a * (F1 + Fd)
    Ttot = Ta + Tp

    return values
