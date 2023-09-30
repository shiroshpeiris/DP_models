import math
import PUdefs as pu
import numpy as np
from numba import njit
from PUdefs import Wrated

@njit

def linemodel(x, wsys, sysparams, U_in, U_out):
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
    R_12act = sysparams[0] * (pu.ZbaseHV)
    wL_12act = sysparams[1] * (pu.ZbaseHV)

    # base parameters

    LbaseHV = pu.LbaseHV
    ZbaseHV = pu.ZbaseHV

    Wbase = pu.Wbase

    # -------per unitized values--------------

    w0 = Wrated / Wbase

    R_12 = R_12act / ZbaseHV
    L_12 = (wL_12act / Wbase) / LbaseHV



    ##element order delta,wint,P,Q,Phid,Phiq,Gamd,Gamq,Icd,Icq,Ufd,Ufq

    U_in_q = U_in[0]
    U_in_d = U_in[1]

    U_out_q = U_out[0]
    U_out_d = U_out[1]


    # ---line 12 states--

    I_12q = x[0]
    I_12d = x[1]


    # -------------------Equations-------------------

    F = np.empty((2))

    # ---------------Line component----------

    #----------convert to common reference frame----------

    # IsQ = (+Isq * np.cos(delta) - Isd * np.sin(delta))
    # IsD = (+Isq * np.sin(delta) + Isd * np.cos(delta))


    F[0] = Wbase * (((1 / L_12) * ((U_in_q - U_out_q) - (R_12) * I_12q)) + (wsys) * I_12d)
    F[1] = Wbase * (((1 / L_12) * ((U_in_d - U_out_d) - (R_12) * I_12d)) - (wsys) * I_12q)


    return F
