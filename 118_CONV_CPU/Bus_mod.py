import math
import PUdefs as pu
import numpy as np
from numba import njit
from PUdefs import Wrated
@njit


def busmodel(x, wsys, sysparams, ldparams, GenIq, GenId, LdI, Iin, Iout):
    """

    :param x:
    :param wparams:
    :param sysparams:
    :param Isq_u1_12_load:
    :param Isd_u1_12_load:
    :param Isq_u1_13_load:
    :param Isd_u1_13_load:
    :param Isq:
    :param Isd:
    :return:
    """
    pi = math.pi
    wC_bus = sysparams/(pu.ZbaseHV)
    Pload = ldparams[0]
    # wCload = ldparams[2]

    # base parameters

    CbaseHV = pu.CbaseHV
    ZbaseHV = pu.ZbaseHV

    Wbase = pu.Wbase



    # -------per unitized values--------------


    C_bus = (wC_bus / Wbase) / CbaseHV
    # C_load = (wCload / Wbase) / CbaseHV
    # C_bus = C_bus + C_load


    LdIq = LdI[0]
    LdId = LdI[1]

    Iinq = Iin[0]
    Iind = Iin[1]

    Ioutq = Iout[0]
    Ioutd = Iout[1]


    # ---states--

    Ubus_q = x[0]
    Ubus_d = x[1]

    # load calculations
    if Pload > 0:
        Rload = (pu.VratedPhrmsHV ** 2) / Pload
        Rload12 = Rload / ZbaseHV
        LdIq = LdIq + (Ubus_q / Rload12)
        LdId = LdId + (Ubus_d / Rload12)

    else:
        LdIq = LdIq
        LdId = LdId



    F = np.empty((2))

    # ---------------bus component-----------

    #----------convert to common reference frame----------

    # IsQ = (+Isq * np.cos(delta) - Isd * np.sin(delta))
    # IsD = (+Isq * np.sin(delta) + Isd * np.cos(delta))

    F[0] = Wbase * (((2 / (C_bus)) * ((GenIq + Iinq) - (LdIq + Ioutq))) + (wsys) * Ubus_d)
    F[1] = Wbase * (((2 / (C_bus)) * ((GenId + Iind) - (LdId + Ioutd))) - (wsys) * Ubus_q)



    return F
