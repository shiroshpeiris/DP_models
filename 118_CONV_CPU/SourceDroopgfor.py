import math
from PUdefs import *
import numpy as np
from numba import njit


@njit
def unitmodel(x, sysparams, inputs, Ugq, Ugd, delta, genscaler, unitno):
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

    # base parameters

    MVAbase = MVAbaseLV

    # ------------control params--------------------

    Wrated = sysparams[0]
    H = sysparams[1]
    wc = sysparams[2]
    Kp = sysparams[3]
    Kq = sysparams[4]
    Kd = sysparams[5]
    Kpi = sysparams[6]
    Kii = sysparams[7]
    Rvir = sysparams[8]
    Lvir = sysparams[9]
    Cfilt = sysparams[10]
    Lfilt = sysparams[11]
    Lgrid = sysparams[12]
    Rgrid = sysparams[13]
    Ltf = sysparams[14]
    Cdamp = sysparams[15]
    Ldamp = sysparams[16]
    Rdamp = sysparams[17]
    SCR = sysparams[18]
    XR = sysparams[19]



    #-----input parameters

    Pref = inputs[0]
    Qref = inputs[1]
    Upcc = inputs[2]
    wref = inputs[3]

    # base parameters

    VbaseLV = (VratedLLrmsLV * np.sqrt(2)) / (np.sqrt(3))
    ZbaseLV = (VratedLLrmsLV ** 2) / MVAbase
    LbaseLV = ZbaseLV / Wrated
    CbaseLV = 1 / (Wrated * ZbaseLV)
    Wbase = Wrated

    # -------Filter per unitized values--------------

    Cf = Cfilt / CbaseLV
    Lf = Lfilt / LbaseLV

    Cd = Cdamp / CbaseLV
    Ld = Ldamp / LbaseLV
    Rd = Rdamp / ZbaseLV

    # --LV line per unitized values




    Lg = (Lgrid / LbaseLV) + Ltf #+ L_sys
    # Line and Transformer impedance
    Rg = Rgrid / ZbaseLV #+ R_sys
    w0 = Wrated / Wbase

    Lvir = Lvir/LbaseLV
    Rvir = Rvir/ZbaseLV

    # -------Nonlinear equations-------------------

    wint = x[0]
    P = x[1]
    Q = x[2]
    Icqref = x[3]
    Icdref = x[4]
    Gamq = x[5]
    Gamd = x[6]
    Icq = x[7]
    Icd = x[8]
    Ufq = x[9]
    Ufd = x[10]
    Isq = x[11]
    Isd = x[12]
    I1q = x[13]
    I1d = x[14]
    Uc2q = x[15]
    Uc2d = x[16]

    # -------------------Equations-------------------
    # -------------------Equations-------------------

    F = np.empty((17))
    wsys = Kp * wint + wref

    # -!!! All decoupled equations should be multiplied by Wbase when per
    # unitization

    Ucqref = Upcc + Kq * Qref - Kq * Q
    Ucdref = 0

    # -----------Frequency control---------

    F[0] = -(Kd / (2 * H)) * wint + (Pref / (2 * H)) - (P / (2 * H))

    # --------Power Measurements---------

    F[1] = -wc * P + wc * (Ufq * Icq + Ufd * Icd)
    F[2] = -wc * Q + wc * (Ufd * Icq - Ufq * Icd)

    # ------Control Equations--------

    F[3] = Wbase * ((Ucqref / Lvir) - (Ufq / Lvir) - (Rvir / Lvir) * Icqref + w0 * Icdref)
    F[4] = Wbase * ((Ucdref / Lvir) - (Ufd / Lvir) - (Rvir / Lvir) * Icdref - w0 * Icqref)



    F[5] = (Icqref - Icq)
    F[6] = (Icdref - Icd)

    # ------Filter Current--------
    F[7] = Wbase * ((1 / Lf) * ((Icqref - Icq) * Kpi + Gamq * Kii + Ufq - Icd * w0 * Lf) - (Ufq / Lf) + wsys * Icd)
    F[8] = Wbase * ((1 / Lf) * ((Icdref - Icd) * Kpi + Gamd * Kii + Ufd + Icq * w0 * Lf) - (Ufd / Lf) - wsys * Icq)

    # ------------Grid current and Filter voltages------

    F[9] = Wbase * ((Icq - Isq - I1q) / Cf + wsys * Ufd)
    F[10] = Wbase * ((Icd - Isd - I1d) / Cf - wsys * Ufq)

    #-------convert from common reference frame

    UgQ = (+Ugq * np.cos(delta) + Ugd * np.sin(delta))
    UgD = (-Ugq * np.sin(delta) + Ugd * np.cos(delta))


    F[11] = Wbase * ((Ufq / Lg) - (1 / Lg) * (UgQ) - (Rg / Lg) * Isq + wsys * Isd)
    F[12] = Wbase * ((Ufd / Lg) - (1 / Lg) * (UgD) - (Rg / Lg) * Isd - wsys * Isq)

    F[13] = Wbase * ((1 / (Rd * Cf)) * (Icq - Isq - I1q) - (I1q / (Rd * Cd)) + ((Ufq - Uc2q) / Ld) + wsys * I1d)
    F[14] = Wbase * ((1 / (Rd * Cf)) * (Icd - Isd - I1d) - (I1d / (Rd * Cd)) + ((Ufd - Uc2d) / Ld) - wsys * I1q)

    F[15] = Wbase * ((I1q / Cd) + wsys * Uc2d)
    F[16] = Wbase * ((I1d / Cd) - wsys * Uc2q)


    v_mag = np.sqrt(UgD ** 2 + UgQ ** 2)
    VAR = np.empty((4))
    VAR[0] = P * genscaler
    VAR[1] = Q * genscaler
    VAR[2] = wsys
    VAR[3] = v_mag

    return F, VAR