import math
import PUdefs as pu
import numpy as np
from numba import njit
from PUdefs import Wrated


@njit

def Lloadmodel(x, wsys, sysparams, busU):
    """

    :param x:
    :param wparams:
    :param sysparams:
    :param U1q:
    :param U1d:
    :param U2q:
    :param U2d:
    :return:
    """
    pi = math.pi
    # base parameters
    Qload = sysparams


    ZbaseHV = pu.ZbaseHV

    Wbase = pu.Wbase

    w0 = Wrated / Wbase

    Lload = (pu.VratedPhrmsHV ** 2) / (Qload * Wrated)

    # ---------load and HV line 1-2------------

    Lload12 = Lload * Wrated / ZbaseHV


    # ---line 12 states--

    Ubus_q = busU[0]
    Ubus_d = busU[1]
    # ---line 13 states--

    I_L_q = x[0]
    I_L_d = x[1]

    # -------------------Equations-------------------

    F = np.empty((2))

    # ------Inverter 1-------

    F[0] = Wbase * ((Ubus_q/Lload12) + (wsys) * I_L_d)
    F[1] = Wbase * ((Ubus_d/Lload12) - (wsys) * I_L_q)

    return F
