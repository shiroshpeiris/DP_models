import math
import PUdefs as pu
import numpy as np
from numba import njit


@njit
def unitmodel(x, sysparams, inputs, Ugq, Ugd, delta, genscaler, unitno):
    """

    :param Vrefgen:
    :param wrefgen:
    :param x:
    :param Ugq:
    :param Ugd:
    :return:
    """
    pi = math.pi

    # base parameters

    Wbase = pu.Wbase



    #-----------control and machine Parameters-------------------

    # ------Inertia and Damping---------

    Wrated = sysparams[0]
    Hgen = sysparams[1]
    KD = sysparams[2]

    # Stator and rotor inductances, resistances
    Ld_subtr = sysparams[3]
    Ld_tr = sysparams[4]
    Ld = sysparams[5]
    Lq_subtr = sysparams[6]
    Lq_tr = sysparams[7]
    Lq = sysparams[8]
    Ra = sysparams[9]
    Ll = sysparams[10]

    # Transient and subtransient time constants
    Td_subtr = sysparams[11]
    Td_tr = sysparams[12]
    Td0_subtr = sysparams[13]
    Td0_tr = sysparams[14]
    Tq_subtr = sysparams[15]
    Tq_tr = sysparams[16]
    Tq0_subtr = sysparams[17]
    Tq0_tr = sysparams[18]

    # ----Mechanical Governor-----

    Tp = sysparams[19]
    Qgov = sysparams[20]
    Tg = sysparams[21]
    Rt = sysparams[22]
    Rp = sysparams[23]
    Tr = sysparams[24]


    # --------------------Exciter--------------

    Ta = sysparams[25]
    Ka = sysparams[26]
    Tc = sysparams[27]
    Tb = sysparams[28]
    Trexc = sysparams[29]

    #------------line parameters

    Rgrid = sysparams[30]
    Lgrid = sysparams[31]
    Xtf = sysparams[32]
    VratedLLrmsLV = sysparams[33]

    #------------Turbine parameters


    qNL = sysparams[34]
    h0 = sysparams[35]
    Tw = sysparams[36]
    fp = sysparams[37]
    D = sysparams[38]
    At = sysparams[39]

    # ----input paramters
    Vrefgen = inputs[0]
    Loadref = inputs[1]
    wrefgen = inputs[2]
    Eofst = inputs[3]

    # base parameters

    VbaseLV = (VratedLLrmsLV * np.sqrt(2)) / (np.sqrt(3))
    ZbaseLV = (VratedLLrmsLV ** 2) / pu.MVAbase
    LbaseLV = ZbaseLV / pu.Wrated
    CbaseLV = 1 / (pu.Wrated * ZbaseLV)


    Cbase = CbaseLV
    Lbase = LbaseLV
    Zbase = ZbaseLV
    Wbase = pu.Wbase



    #------Per unitized LV line parameters------------

    Rtfgen = Rgrid / Zbase
    Ltfgen = (Lgrid / Lbase) + Xtf

    # ----Calculated generator stator and field inductances and resistances-----

    # d-axis calculated inductances

    Lad = Ld - Ll
    Lfd = (Ld_tr - Ll) * (Lad) / (Lad - (Ld_tr - Ll))

    L1d = ((Ld_subtr - Ll) * (Lad * Lfd)) / (Lad * Lfd - (Ld_subtr - Ll) * (Lad + Lfd))
    Rfd = (Lad + Lfd) / (Wrated * Td0_tr)
    # Rfd=(Lfd+(Lad*Ll)/(Lad+Ll))/(w0*Td_tr)
    R1d = (1 / (Wrated * Td0_subtr)) * (L1d + (Lad * Lfd) / (Lad + Lfd))

    # q-axis calculated inductances

    Laq = Lq - Ll
    L1q = (Lq_tr - Ll) * (Laq) / (Laq - (Lq_tr - Ll))
    L2q = ((Lq_subtr - Ll) * (Laq * L1q)) / (Laq * L1q - (Lq_subtr - Ll) * (Laq + L1q))
    R1q = (Laq + L1q) / (Wrated * Tq0_tr)
    R2q = (1 / (Wrated * Tq0_subtr)) * (L2q + (Laq * L1q) / (Laq + L1q))

    # Intermediaty constants-------------------
    Lpd = Ld * Lfd * L1d + Lad * Ll * L1d + Lad * Ll * Lfd
    Lpq = Lq * L1q * L2q + Laq * Ll * L2q + Laq * Ll * L1q

    # ----Coefficents of stator currrent DQ components

    Kid1 = -(Lfd * L1d + Lad * (L1d + Lfd)) / Lpd
    Kid2 = (Lad * L1d) / Lpd
    Kid3 = (Lad * Lfd) / Lpd

    Kiq1 = -(L1q * L2q + Laq * (L2q + L1q)) / Lpq
    Kiq2 = (Laq * L2q) / Lpq
    Kiq3 = (Laq * L1q) / Lpq

    # ------Coefficents of Field and damper winding fluxes

    KPhifd_Efd = Wrated * Rfd / Lad
    KPhifd_Phifd = -(Wrated * Rfd / Lfd) * (1 + (Kid2 / Kid1))
    KPhifd_id = (Wrated * Rfd / Lfd) * (Ll + (1 / Kid1))
    KPhifd_Phi1d = -(Wrated * Rfd / Lfd) * (Kid3 / Kid1)

    KPhi1d_Phi1d = -(Wrated * R1d / L1d) * (1 + (Kid3 / Kid1))
    KPhi1d_id = (Wrated * R1d / L1d) * (Ll + (1 / Kid1))
    KPhi1d_Phifd = -(Wrated * R1d / L1d) * (Kid2 / Kid1)

    KPhi1q_Phi1q = -(Wrated * R1q / L1q) * (1 + (Kiq2 / Kiq1))
    KPhi1q_iq = (Wrated * R1q / L1q) * (Ll + (1 / Kiq1))
    KPhi1q_Phi2q = -(Wrated * R1q / L1q) * (Kiq3 / Kiq1)

    KPhi2q_Phi2q = -(Wrated * R2q / L2q) * (1 + (Kiq3 / Kiq1))
    KPhi2q_iq = (Wrated * R2q / L2q) * (Ll + (1 / Kiq1))
    KPhi2q_Phi1q = -(Wrated * R2q / L2q) * (Kiq2 / Kiq1)

    # ---------------------Coefficents of Te----------

    Kw_id_Phiq = -(1 / (2 * Hgen)) * (1 / Kiq1)
    Kw_id_Phi1q = (1 / (2 * Hgen)) * (Kiq2 / Kiq1)
    Kw_id_Phi2q = (1 / (2 * Hgen)) * (Kiq3 / Kiq1)

    Kw_iq_Phid = (1 / (2 * Hgen)) * (1 / Kid1)
    Kw_iq_Phifd = -(1 / (2 * Hgen)) * (Kid2 / Kid1)
    Kw_iq_Phi1d = -(1 / (2 * Hgen)) * (Kid3 / Kid1)

    # -----Coefficents of derivatives of Stator flux DQ coefficents

    Kdcom = Wrated / (1 - (Ltfgen * Kid1))
    KdEfd = (Ltfgen * Kid2 * Rfd) / Lad
    KdPhifd = (-Ltfgen * Kid2 * Rfd) / Lfd
    KdPhi1d = (-Ltfgen * Kid3 * R1d) / L1d
    KdPhid = Ltfgen * (((Kid2 * Rfd) / Lfd) + ((Kid3 * R1d) / L1d))
    Kdid = KdPhid * Ll + (Rtfgen + Ra)

    Kqcom = Wrated / (1 - (Ltfgen * Kiq1))
    KqPhi1q = (-Ltfgen * Kiq2 * R1q) / L1q
    KqPhi2q = (-Ltfgen * Kiq3 * R2q) / L2q
    KqPhiq = Ltfgen * (((Kiq2 * R1q) / L1q) + ((Kiq3 * R2q) / L2q))
    Kqiq = KqPhiq * Ll + (Rtfgen + Ra)

    # Coefficents of derivatives Stator current DQ components

    KiPhid = (Kid1 * Kdcom * KdPhid + (Wrated * ((Kid2 * Rfd / Lfd) + (Kid3 * R1d / L1d))))
    KiEfd = (Kid1 * Kdcom * KdEfd + (Kid2 * Wrated * Rfd / Lad))
    KiPhifd = (Kid1 * Kdcom * KdPhifd - (Kid2 * Wrated * Rfd / Lfd)) + KiPhid * (-Kid2 / Kid1)
    KiPhi1d = (Kid1 * Kdcom * KdPhi1d - (Kid3 * Wrated * R1d / L1d)) + KiPhid * (-Kid3 / Kid1)
    Kiid = (Kid1 * Kdcom * Kdid + (Wrated * Ll * ((Kid2 * Rfd / Lfd) + (Kid3 * R1d / L1d)))) + KiPhid * (1 / Kid1)

    KiPhiq = (Kiq1 * Kqcom * KqPhiq + (Wrated * ((Kiq2 * R1q / L1q) + (Kiq3 * R2q / L2q))))
    KiPhi1q = (Kiq1 * Kqcom * KqPhi1q - (Kiq2 * Wrated * R1q / L1q)) + KiPhiq * (-Kiq2 / Kiq1)
    KiPhi2q = (Kiq1 * Kqcom * KqPhi2q - (Kiq3 * Wrated * R2q / L2q)) + KiPhiq * (-Kiq3 / Kiq1)
    Kiiq = (Kiq1 * Kqcom * Kqiq + (Wrated * Ll * ((Kiq2 * R1q / L1q) + (Kiq3 * R2q / L2q)))) + KiPhiq * (1 / Kiq1)

    #---Reference frame transformation----

    UgQ = (+Ugq * np.cos(delta-np.pi) + Ugd * np.sin(delta-np.pi))
    UgD = (-Ugq * np.sin(delta-np.pi) + Ugd * np.cos(delta-np.pi))


    ##element order wint,P,Q,Phid,Phiq,Gamd,Gamq,Icd,Icq,Ufd,Ufq
    wrgen = x[0]
    iqgen = -x[1]
    idgen = -x[2]
    Phi_fd = x[3]
    Phi_1d = x[4]
    Phi_1q = x[5]
    Phi_2q = x[6]

    Efd = x[7]
    Xe1 = x[8]
    Xe2 = x[9]

    Xt1 = x[10]
    Xt2 = x[11]
    Zg = x[12]
    Xt4 = x[13]

    y = x[14]
    Efdtemp = 0
    if Efd > 5.64:
        Efdtemp = Efd
        Efd = 5.64
    elif Efd < -4.53:
        Efdtemp = Efd

        Efd = -4.53


    # Xgov = x(31);
    # -------------------Equations-------------------

    F = np.empty((15))

    # --------------------------------Synch Gen------------------------------

    Te = ((1 / Kiq1) * (iqgen - (Kiq2 * Phi_1q + Kiq3 * Phi_2q))) * idgen - ((1 / Kid1) * (idgen - (Kid2 * Phi_fd + Kid3 * Phi_1d))) * iqgen

    # ----NL turbine---------
    Tm = (((At * (y ** 3)) / (Zg ** 2)) - ((qNL * At * (y ** 2)) / (Zg ** 2)) - (Zg * D * (wrgen - wrefgen)))/wrgen

    #Tm = Zg

    U21_in_q = UgQ
    U21_in_d = UgD


    # Pref = 1
    # Qref = -0.5



    Pmeas = U21_in_q * iqgen + U21_in_d * idgen
    Qmeas = U21_in_d * iqgen - U21_in_q * idgen

    Phi_d = (1 / Kid1) * (idgen - (Kid2 * Phi_fd + Kid3 * Phi_1d))
    Phi_q = (1 / Kiq1) * (iqgen - (Kiq2 * Phi_1q + Kiq3 * Phi_2q))

    ifd = (1 / Lfd) * (Phi_fd - (((1 / Kid1) * (idgen - (Kid2 * Phi_fd + Kid3 * Phi_1d))) + Ll * idgen))
    i1d = (1 / L1d) * (Phi_1d - (((1 / Kid1) * (idgen - (Kid2 * Phi_fd + Kid3 * Phi_1d))) + Ll * idgen))
    i1q = (1 / L1q) * (Phi_1q - (((1 / Kiq1) * (iqgen - (Kiq2 * Phi_1q + Kiq3 * Phi_2q))) + Ll * iqgen))
    i2q = (1 / L2q) * (Phi_2q - (((1 / Kiq1) * (iqgen - (Kiq2 * Phi_1q + Kiq3 * Phi_2q))) + Ll * iqgen))

    ed_gen = (Ltfgen / Wbase) * ((KiEfd * Efd) + (KiPhifd * Phi_fd) + (KiPhi1d * Phi_1d) + (Kiid * idgen) + Kid1 * Kdcom * (wrgen * Ltfgen * iqgen + U21_in_d) - (Kid1 * Kdcom / Kiq1) * iqgen * wrgen + (Kid1 * Kdcom * Kiq2 / Kiq1) * Phi_1q * wrgen + (Kid1 * Kdcom * Kiq3 / Kiq1) * Phi_2q * wrgen) + (
            Rtfgen * idgen) + (wrgen * Ltfgen * iqgen) + U21_in_d
    eq_gen = (Ltfgen / Wbase) * ((KiPhi1q * Phi_1q) + (KiPhi2q * Phi_2q) + (Kiiq * iqgen) + Kiq1 * Kqcom * (-wrgen * Ltfgen * idgen + U21_in_q) + (Kiq1 * Kqcom / Kid1) * idgen * wrgen - (Kiq1 * Kqcom * Kid2 / Kid1) * Phi_fd * wrgen - (Kiq1 * Kqcom * Kid3 / Kid1) * Phi_1d * wrgen) + (
            Rtfgen * iqgen) - (wrgen * Ltfgen * idgen) + U21_in_q


    # ----------------functions------------
    # Loadref =  0.1 * (Pref - Pmeas) + Perr * 0.1
    # Vrefgen = 0.1 * (Qref - Qmeas) + Qerr * 0.1
    # -----Generator----

    F[0] = (1 / (2 * Hgen)) * (Tm - Te - KD * wrgen)

    # --------id and iq derivatives with L load-------

    F[1] = -((KiPhi1q * Phi_1q) + (KiPhi2q * Phi_2q) + (Kiiq * iqgen) + Kiq1 * Kqcom * (-wrgen * Ltfgen * idgen + U21_in_q) + (Kiq1 * Kqcom / Kid1) * idgen * wrgen - (Kiq1 * Kqcom * Kid2 / Kid1) * Phi_fd * wrgen - (Kiq1 * Kqcom * Kid3 / Kid1) * Phi_1d * wrgen)
    F[2] = -((KiEfd * Efd) + (KiPhifd * Phi_fd) + (KiPhi1d * Phi_1d) + (Kiid * idgen) + Kid1 * Kdcom * (wrgen * Ltfgen * iqgen + U21_in_d) - (Kid1 * Kdcom / Kiq1) * iqgen * wrgen + (Kid1 * Kdcom * Kiq2 / Kiq1) * Phi_1q * wrgen + (Kid1 * Kdcom * Kiq3 / Kiq1) * Phi_2q * wrgen)

    F[3] = KPhifd_Efd * Efd + KPhifd_Phifd * Phi_fd + KPhifd_id * idgen + KPhifd_Phi1d * Phi_1d
    F[4] = KPhi1d_Phi1d * Phi_1d + KPhi1d_id * idgen + KPhi1d_Phifd * Phi_fd

    F[5] = (KPhi1q_Phi1q * Phi_1q) + (KPhi1q_iq * iqgen) + (KPhi1q_Phi2q * Phi_2q)
    F[6] = (KPhi2q_Phi2q * Phi_2q) + (KPhi2q_iq * iqgen) + (KPhi2q_Phi1q * Phi_1q)

    # --------Exciter----

    # else:

    if Efdtemp > 5.64:
        Efd = Efdtemp
    elif Efdtemp < -4.53:
        Efd = Efdtemp
    F[7] = (1 / Ta) * (-Efd + (Ka * Xe1 * (1 - (Tc / Tb))) - ((Ka * Tc * (Xe2)) / (Tb)) + ((Ka * Tc * Vrefgen) / (Tb)))
    F[8] = -(1 / Tb) * (Xe1 + Xe2 - Vrefgen)
    F[9] = (1 / Trexc) * (-Xe2 + np.sqrt(ed_gen ** 2 + eq_gen ** 2))
    # ---Non_Elastic water column without surge tank with Mechanical Governor-------

    F[10] = (1 / Tp) * (-Xt1 - ((Rt + Rp) * (Xt2 - Loadref)) + Xt4 - wrgen + wrefgen)

    if Xt2 > 1:
        Xt2 = 1

    F[11] = Qgov * Xt1
    F[12] = (1 / Tg) * (Xt2 - Zg)
    F[13] = (1 / Tr) * ((Rt * (Xt2 - Loadref)) - Xt4)
    F[14] = (1 / Tw) * (h0 - ((y / Zg) ** 2) - (y ** 2) * fp)
    # print(ed_gen,eq_gen)
    # print(Vrefgen,Loadref)

    e_ang = np.arctan2(ed_gen, eq_gen)
    e_mag = np.sqrt(ed_gen ** 2 + eq_gen ** 2)
    v_mag = np.sqrt(UgD ** 2 + UgQ ** 2)
    VAR = np.empty((4))
    VAR[0] = Pmeas * genscaler
    VAR[1] = Qmeas * genscaler
    VAR[2] = wrgen
    VAR[3] = v_mag

    # if unitno == 2:
    #     F = F*0



    return F, VAR
