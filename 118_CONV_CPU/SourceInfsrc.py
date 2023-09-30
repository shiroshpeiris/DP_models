import math
import PUdefs as pu
import numpy as np
from numba import njit


@njit
def unitmodel(x, sysparams, inputs, Ugq, Ugd, delta):
    """

    :param x:
    :param sysparams:
    :param Pref:
    :param Qref:
    :param Upcc:
    :param wref:
    :param Ugq:
    :param Ugd:
    :return:
    """
    pi = math.pi
    # ------------control params--------------------

    Lgrid = sysparams[0]
    Rgrid = sysparams[1]
    Ltf = sysparams[2]
    VratedLLrmsLV = sysparams[3]

    # -----input parameters

    Ufq = inputs[0]
    Ufd = inputs[1]
    wsys = inputs[2]

    # base parameters

    VbaseLV = (VratedLLrmsLV * np.sqrt(2)) / (np.sqrt(3))
    ZbaseLV = (VratedLLrmsLV ** 2) / pu.MVAbase
    LbaseLV = ZbaseLV / pu.Wrated
    CbaseLV = 1 / (pu.Wrated * ZbaseLV)


    Cbase = CbaseLV
    Lbase = LbaseLV
    Zbase = ZbaseLV
    Wbase = pu.Wbase



    # --LV line per unitized values

    Lg = (Lgrid / Lbase) + Ltf  # Line and Transformer impedance
    Rg = Rgrid / Zbase

    # -------Nonlinear equations-------------------


    Isq = x[0]
    Isd = x[1]

    # -------------------Equations-------------------
    # -------------------Equations-------------------

    F = np.empty((2))

    # -!!! All decoupled equations should be multiplied by Wbase when per
    # unitization


    #-------convert from common reference frame

    UfQ = (+Ufq * np.cos(delta) + Ufd * np.sin(delta))                  # source side lagging delta introduces 30 degree shift
    UfD = (-Ufq * np.sin(delta) + Ufd * np.cos(delta))

    UgQ = (+Ugq * np.cos(delta) - Ugd * np.sin(delta))
    UgD = (+Ugq * np.sin(delta) + Ugd * np.cos(delta))

    F[0] = Wbase * ((UfQ / Lg) - (1 / Lg) * Ugq - (Rg / Lg) * Isq + wsys * Isd)
    F[1] = Wbase * ((UfD / Lg) - (1 / Lg) * Ugd - (Rg / Lg) * Isd - wsys * Isq)

    v_mag = np.sqrt(Ugd ** 2 + Ugq ** 2)
    i_mag = np.sqrt(Isq ** 2 + Isd ** 2)
    VAR = np.empty((2))
    VAR[0] = v_mag
    VAR[1] = UgQ * Isq + UgD * Isd
    return F, VAR


