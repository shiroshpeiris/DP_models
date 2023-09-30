from numba import cuda
from numpy import genfromtxt
import numba as nb
from datetime import datetime
import PUdefs as pu
import numpy as np
from Paramload import readparams
import pyqtgraph as pg
import math
from PyQt5 import QtWidgets

#--------read parameters and operating setpoints for network and convereters
params,inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr,line_arriv,line_leave = readparams()

#--model data definitions-----
no_cases = 8
no_states = 2803*no_cases
delta_t = np.float32(0.00003)
nPlots = 54
X = []
Y = []
out = []
tx = []
plotvarnum = 108 * 4 * 8
outarr = np.zeros([0,plotvarnum])
windowlen = 500

#-----obtain initial conditions for the simulation--------------
x = genfromtxt('init_snapshot.csv', delimiter=',')
state_arr = x
i=0
while i < (no_cases-1):
    state_arr = np.append(state_arr,x)
    i = i+1
x = state_arr

#---cuda kernels for GPU simulation---------------------
@cuda.jit
def nlmodel(x, sol, params, inputs,bus_arr,genbus_ind,loadbus_ind,ldbus_no_arr,genbus_no_arr,line_arriv,line_leave,VARS):

    tid = cuda.grid(1)

    caseid=int(math.floor(tid / 2803)*2803)

    LbaseHV = pu.LbaseHV
    ZbaseHV = pu.ZbaseHV
    Wbase = pu.Wbase
    # ===Split the array of states based on individual function lengths=========================

    gfl = x[0+ caseid:972+ caseid]
    gfr = x[(len(gfl))+ caseid:((len(gfl)) + 918)+ caseid]

    angles = x[(len(gfl) + len(gfr)) + caseid:(len(gfl) + len(gfr)) + 107 + caseid]
    buses = x[(len(angles)+len(gfl) + len(gfr))+ caseid:((len(angles)+len(gfl) + len(gfr)) + 236+ caseid)]
    lines = x[(len(buses)+len(angles)+len(gfl) + len(gfr))+ caseid:((len(buses)+len(angles)+len(gfl) + len(gfr)) + 372+ caseid)]
    loads = x[(len(lines)+len(buses)+len(angles)+len(gfl) + len(gfr))+ caseid:((len(lines)+len(buses)+len(angles)+len(gfl) + len(gfr)) + 198+ caseid)]


    # VAR_GFL=nb.cuda.local.array(54*4, dtype=nb.float32)
    # VAR_GFR=nb.cuda.local.array(54*4, dtype=nb.float32)
    #

    # ====Split parameter arrays for the individual functions========
    p_buses = params[0:118]
    p_loads = params[len(p_buses):len(p_buses)+198]
    p_lines = params[len(p_buses)+len(p_loads):len(p_buses)+len(p_loads)+372]
    p_gfl = params[len(p_buses)+len(p_loads)+len(p_lines):len(p_buses)+len(p_loads)+len(p_lines)+1134]
    p_gfr = params[len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl):len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+1080]

    p_gflscalers = params[len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr):len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr)+54]
    p_gfrscalers = params[len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr)+len(p_gflscalers):len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr)+len(p_gflscalers)+54]
    null = params[3010:3012]

    inputs_gfl = inputs[0:162]
    inputs_gfr = inputs[len(inputs_gfl):len(inputs_gfl) + 216]

    wgflref = inputs_gfl[0 * 3 + 2] + gfl[0 * 18 + 7] * p_gfl[0 * 21 + 9] + gfl[0 * 18 + 2] * p_gfl[0 * 21 + 10]                # angle of the first gfl converter for the reference frame

    # ======calculate frequency  parameters needed for decoupled components in modular sections==


    gfl_angles= angles[0:53]
    gfr_angles = angles[53:107]


    # ===========calling idividual functions to construct the total system====================

    if tid >= 0 + caseid and tid < len(gfl)/18 + caseid:

        i = int(tid - caseid)

        if i == 0:
            # The function decorated with numba.jit may be directly reused
            gflst = gfl[i * 18:i * 18 + 18]
            gflparams = p_gfl[i*21:i*21+21]
            gflinps = inputs_gfl[i*3:i*3+3]
            Ugq_gfl = buses[int(genbus_no_arr[i]*2-2)]
            Ugd_gfl = buses[int(genbus_no_arr[i]*2-1)]
            delta_gfl = 0
        else:
            gflst = gfl[i * 18:i * 18 + 18]
            gflparams = p_gfl[i*21:i*21+21]
            gflinps = inputs_gfl[i*3:i*3+3]
            Ugq_gfl = buses[int(genbus_no_arr[i]*2-2)]
            Ugd_gfl = buses[int(genbus_no_arr[i]*2-1)]
            delta_gfl = gfl_angles[i-1]

        Wrated = gflparams[0]
        wc = gflparams[1]
        Kpi = gflparams[2]
        Kii = gflparams[3]
        Cfilt = gflparams[4]
        Lfilt = gflparams[5]
        Lgrid = gflparams[6]
        Rgrid = gflparams[7]
        Ltf = gflparams[8]
        Kp_pll = gflparams[9]
        Ki_pll = gflparams[10]
        Kp_p = gflparams[11]
        Ki_p = gflparams[12]
        Kp_q = gflparams[13]
        Ki_q = gflparams[14]
        KQ = gflparams[15]
        Tfpll = gflparams[16]
        Cdamp = gflparams[17]
        Ldamp = gflparams[18]
        Rdamp = gflparams[19]
        VratedLLrmsLV = gflparams[20]

        # ----input paramters
        Prefext = gflinps[0]
        Qref = gflinps[1]
        wref = gflinps[2]

        VbaseLV = (pu.VratedLLrmsLV * math.sqrt(2)) / (math.sqrt(3))
        ZbaseLV = (pu.VratedLLrmsLV ** 2) / pu.MVAbaseLV
        LbaseLV = ZbaseLV / pu.Wrated
        CbaseLV = 1 / (pu.Wrated * ZbaseLV)
        Wbase = pu.Wrated

        # -------per unitized values--------------

        Cf = Cfilt / CbaseLV
        Lf = Lfilt / LbaseLV
        Lg = (Lgrid / LbaseLV) + Ltf  # Line and Transformer impedance
        Rg = Rgrid / ZbaseLV
        w0 = Wrated / pu.Wbase

        # -------Filter per unitized values--------------

        Cd = Cdamp / CbaseLV
        Ld = Ldamp / LbaseLV
        Rd = Rdamp / ZbaseLV
        # -------Nonlinear equations-------------------

        alpha = gflst[0]
        beta = gflst[1]
        Psi = gflst[2]
        Icd = gflst[3]
        Icq = gflst[4]
        Ufd = gflst[5]
        Ufq = gflst[6]
        Ufd_pll = gflst[7]
        Isd = gflst[8]
        Isq = gflst[9]  # -----------------change the dq order of these
        I1d = gflst[10]
        I1q = gflst[11]
        Uc2d = gflst[12]
        Uc2q = gflst[13]
        P = gflst[14]
        Q = gflst[15]
        Gamd = gflst[16]
        Gamq = gflst[17]


        # -!!! All decoupled equations should be multiplied by Wbase when per
        # unitization

        # -------convert from common reference frame

        UgQ = (+Ugq_gfl * math.cos(delta_gfl) + Ugd_gfl * math.sin(delta_gfl))  # --------30 degree leading shift from the LV transformer is added to the delta---
        UgD = (-Ugq_gfl * math.sin(delta_gfl) + Ugd_gfl * math.cos(delta_gfl))

        wsys = wref + Ufd_pll * Kp_pll + Ki_pll * Psi
        Pref = Prefext  # + (wref - wsys) * Kp_fq + Ferrder * Ki_fq
        Qref = (Qref)  # + 1 * (1 - np.sqrt(Ufq**2 + Ufd**2)))



        # -------------------Equations-------------------

        Icdref = (Qref - Q) * Kp_q + (alpha * Ki_q)
        Icqref = (Pref - P) * Kp_p + (beta * Ki_p)


        sol[0 + i*18 + caseid] = (Qref - Q)
        sol[1 + i*18 + caseid] = (Pref - P)
        sol[16 + i*18 + caseid] = -Icdref - Icd
        sol[17 + i*18 + caseid] = Icqref - Icq

        sol[3 + i*18 + caseid] = Wbase * ((1 / Lf) * ((-Icdref - Icd) * Kpi + Gamd * Kii + Ufd + Icq * w0 * Lf) - (Ufd / Lf) - wsys * Icq)
        sol[4 + i*18 + caseid] = Wbase * ((1 / Lf) * ((Icqref - Icq) * Kpi + Gamq * Kii + Ufq - Icd * w0 * Lf) - (Ufq / Lf) + wsys * Icd)

        sol[2 + i*18 + caseid] = Ufd_pll

        sol[5 + i*18 + caseid] = Wbase * (((Icd - Isd - I1d) / Cf) - wsys * Ufq)
        sol[6 + i*18 + caseid] = Wbase * (((Icq - Isq - I1q) / Cf) + wsys * Ufd)

        sol[7 + i*18 + caseid] = (1 / Tfpll) * (Ufd - Ufd_pll)


        sol[8 + i*18 + caseid] = Wbase * (Ufd / Lg - (1 / Lg) * (UgD) - (Rg / Lg) * Isd - wsys * Isq)
        sol[9 + i*18 + caseid] = Wbase * (Ufq / Lg - (1 / Lg) * (UgQ) - (Rg / Lg) * Isq + wsys * Isd)

        sol[10 + i*18 + caseid] = Wbase * ((1 / (Rd * Cf)) * (Icd - Isd - I1d) - (I1d / (Rd * Cd)) + ((Ufd - Uc2d) / Ld) - wsys * I1q)
        sol[11 + i*18 + caseid] = Wbase * ((1 / (Rd * Cf)) * (Icq - Isq - I1q) - (I1q / (Rd * Cd)) + ((Ufq - Uc2q) / Ld) + wsys * I1d)

        sol[12 + i*18 + caseid] = Wbase * ((I1d / Cd) - wsys * Uc2q)
        sol[13 + i*18 + caseid] = Wbase * ((I1q / Cd) + wsys * Uc2d)

        sol[14 + i*18 + caseid] = -wc * P + wc * (Ufd * Isd + Ufq * Isq)
        sol[15 + i*18 + caseid] = -wc * Q + wc * (Ufd * Isq - Ufq * Isd)

        VARS[i*4+0+int(caseid/2803)*4*108] = P
        VARS[i*4+1+int(caseid/2803)*4*108] = Q
        VARS[i*4+2+int(caseid/2803)*4*108] = wsys
        VARS[i*4+3+int(caseid/2803)*4*108] = math.sqrt(UgQ**2+UgD**2)



    if tid >= 54 + caseid and tid < 54 + len(gfr)/17 + caseid:

        i = int(tid - 54 - caseid)

        gfrst = gfr[i * 17:i * 17 + 17]
        gfrparams = p_gfr[i*20:i*20+20]
        gfrinps = inputs_gfr[i*4:i*4+4]
        Ugq_gfr = buses[int(genbus_no_arr[i]*2-2)]
        Ugd_gfr = buses[int(genbus_no_arr[i]*2-1)]
        delta_gfr = gfr_angles[i]

        MVAbase = pu.MVAbaseLV

        # ------------control params--------------------

        Wrated = gfrparams[0]
        H = gfrparams[1]
        wc = gfrparams[2]
        Kp = gfrparams[3]
        Kq = gfrparams[4]
        Kd = gfrparams[5]
        Kpi = gfrparams[6]
        Kii = gfrparams[7]
        Rvir = gfrparams[8]
        Lvir = gfrparams[9]
        Cfilt = gfrparams[10]
        Lfilt = gfrparams[11]
        Lgrid = gfrparams[12]
        Rgrid = gfrparams[13]
        Ltf = gfrparams[14]
        Cdamp = gfrparams[15]
        Ldamp = gfrparams[16]
        Rdamp = gfrparams[17]
        SCR = gfrparams[18]
        XR = gfrparams[19]

        # -----input parameters

        Pref = gfrinps[0]
        Qref = gfrinps[1]
        Upcc = gfrinps[2]
        wref = gfrinps[3]

        # base parameters

        VbaseLV = (pu.VratedLLrmsLV * math.sqrt(2)) / (math.sqrt(3))
        ZbaseLV = (pu.VratedLLrmsLV ** 2) / pu.MVAbaseLV
        LbaseLV = ZbaseLV / pu.Wrated
        CbaseLV = 1 / (pu.Wrated * ZbaseLV)
        Wbase = pu.Wrated

        # -------Filter per unitized values--------------

        Cf = Cfilt / CbaseLV
        Lf = Lfilt / LbaseLV

        Cd = Cdamp / CbaseLV
        Ld = Ldamp / LbaseLV
        Rd = Rdamp / ZbaseLV


        Lg = (Lgrid / LbaseLV) + Ltf  # + L_sys
        # Line and Transformer impedance
        Rg = Rgrid / ZbaseLV  # + R_sys
        w0 = Wrated / Wbase

        Lvir = Lvir / LbaseLV
        Rvir = Rvir / ZbaseLV
        # -------Nonlinear equations-------------------

        wint = gfrst[0]
        P = gfrst[1]
        Q = gfrst[2]
        Icqref = gfrst[3]
        Icdref = gfrst[4]
        Gamq = gfrst[5]
        Gamd = gfrst[6]
        Icq = gfrst[7]
        Icd = gfrst[8]
        Ufq = gfrst[9]
        Ufd = gfrst[10]
        Isq = gfrst[11]
        Isd = gfrst[12]
        I1q = gfrst[13]
        I1d = gfrst[14]
        Uc2q = gfrst[15]
        Uc2d = gfrst[16]

        wsys = Kp * wint + wref

        # -!!! All decoupled equations should be multiplied by Wbase when per
        # unitization

        Ucqref = Upcc + Kq * Qref - Kq * Q
        Ucdref = 0

        # -----------Frequency control---------

        sol[0 + 972 + i*17 + caseid] = -(Kd / (2 * H)) * wint + (Pref / (2 * H)) - (P / (2 * H))

        # --------Power Measurements---------

        sol[1 + 972 + i*17 + caseid] = -wc * P + wc * (Ufq * Icq + Ufd * Icd)
        sol[2 + 972 + i*17 + caseid] = -wc * Q + wc * (Ufd * Icq - Ufq * Icd)

        # ------Control Equations--------

        sol[3 + 972 + i*17 + caseid] = Wbase * ((Ucqref / Lvir) - (Ufq / Lvir) - (Rvir / Lvir) * Icqref + w0 * Icdref)
        sol[4 + 972 + i*17 + caseid] = Wbase * ((Ucdref / Lvir) - (Ufd / Lvir) - (Rvir / Lvir) * Icdref - w0 * Icqref)

        sol[5 + 972 + i*17 + caseid] = (Icqref - Icq)
        sol[6 + 972 + i*17 + caseid] = (Icdref - Icd)

        # ------Filter Current--------
        sol[7 + 972 + i*17 + caseid] = Wbase * ((1 / Lf) * ((Icqref - Icq) * Kpi + Gamq * Kii + Ufq - Icd * w0 * Lf) - (Ufq / Lf) + wsys * Icd)
        sol[8 + 972 + i*17 + caseid] = Wbase * ((1 / Lf) * ((Icdref - Icd) * Kpi + Gamd * Kii + Ufd + Icq * w0 * Lf) - (Ufd / Lf) - wsys * Icq)

        # ------------Grid current and Filter voltages------

        sol[9 + 972 + i*17 + caseid] = Wbase * ((Icq - Isq - I1q) / Cf + wsys * Ufd)
        sol[10 + 972 + i*17 + caseid] = Wbase * ((Icd - Isd - I1d) / Cf - wsys * Ufq)

        # -------convert from common reference frame

        UgQ = (+Ugq_gfr * math.cos(delta_gfr) + Ugd_gfr * math.sin(delta_gfr))
        UgD = (-Ugq_gfr * math.sin(delta_gfr) + Ugd_gfr * math.cos(delta_gfr))

        sol[11 + 972 + i*17 + caseid] = Wbase * ((Ufq / Lg) - (1 / Lg) * (UgQ) - (Rg / Lg) * Isq + wsys * Isd)
        sol[12 + 972 + i*17 + caseid] = Wbase * ((Ufd / Lg) - (1 / Lg) * (UgD) - (Rg / Lg) * Isd - wsys * Isq)

        sol[13 + 972 + i*17 + caseid] = Wbase * ((1 / (Rd * Cf)) * (Icq - Isq - I1q) - (I1q / (Rd * Cd)) + ((Ufq - Uc2q) / Ld) + wsys * I1d)
        sol[14 + 972 + i*17 + caseid] = Wbase * ((1 / (Rd * Cf)) * (Icd - Isd - I1d) - (I1d / (Rd * Cd)) + ((Ufd - Uc2d) / Ld) - wsys * I1q)

        sol[15 + 972 + i*17 + caseid] = Wbase * ((I1q / Cd) + wsys * Uc2d)
        sol[16 + 972 + i*17 + caseid] = Wbase * ((I1d / Cd) - wsys * Uc2q)



        VARS[i*4+54*4+0+int(caseid/2803)*4*108] = P
        VARS[i*4+54*4+1+int(caseid/2803)*4*108] = Q
        VARS[i*4+54*4+2+int(caseid/2803)*4*108] = wsys
        VARS[i*4+54*4+3+int(caseid/2803)*4*108] = math.sqrt(UgQ**2+UgD**2)


    if tid >= 108 + caseid and tid < 108 + len(gfl_angles)  + caseid:
        i = int(tid - 108 - caseid)
        wgfl = inputs_gfl[(i+1) * 3 + 2] + gfl[(i+1) * 18 + 7] * p_gfl[(i+1) * 21 + 9] + gfl[(i+1) * 18 + 2] * p_gfl[(i+1) * 21 + 10]       #i + 1 to skip the frequency of the first gfl converter
        sol[i + 1890 + caseid] = pu.Wrated * (wgfl - wgflref)

    if tid >= 161 + caseid and tid < 161 + len(gfr_angles)  + caseid:
        i = int(tid - 161 - caseid)
        wgfr = inputs_gfr[i*4+3] + p_gfr[i*20+3] * gfr[i*17]
        sol[i + 1943 + caseid] = pu.Wrated * (wgfr - wgflref)

    if tid >= 215 + caseid and tid < 215 + len(buses)/2 + caseid:

        i = int((tid) - 215 - caseid)

        GflIQ = nb.cuda.local.array(54, dtype=nb.float32)
        GflID = nb.cuda.local.array(54, dtype=nb.float32)

        GfrIQ = nb.cuda.local.array(54, dtype=nb.float32)
        GfrID = nb.cuda.local.array(54, dtype=nb.float32)

        k = 0
        while k < len(gfl) / 18:
            if k == 0:
                GflIQ[k] = p_gflscalers[k] * gfl[k + 9]
                GflID[k] = p_gflscalers[k] * gfl[k + 8]
            else:
                GflIQ[k] = p_gflscalers[k] * (+gfl[k * 18 + 9] * math.cos(gfl_angles[k - 1]) - gfl[k * 18 + 8] * math.sin(gfl_angles[k - 1]))
                GflID[k] = p_gflscalers[k] * (+gfl[k * 18 + 9] * math.sin(gfl_angles[k - 1]) + gfl[k * 18 + 8] * math.cos(gfl_angles[k - 1]))
            k = k + 1

        k = 0
        while k < len(gfr) / 17:
            GfrIQ[k] = p_gfrscalers[k] * (+gfr[k * 17 + 11] * math.cos(gfr_angles[k]) - gfr[k * 17 + 12] * math.sin(gfr_angles[k]))
            GfrID[k] = p_gfrscalers[k] * (+gfr[k * 17 + 11] * math.sin(gfr_angles[k]) + gfr[k * 17 + 12] * math.cos(gfr_angles[k]))
            k = k + 1

        if genbus_ind[i] == 0:
            GenINJQ = 0
            GenINJD = 0
        else:
            GenINJQ = GflIQ[(genbus_ind[i]-1)] + GfrIQ[(genbus_ind[i] - 1)]
            GenINJD = GflID[(genbus_ind[i]-1)] + GfrID[(genbus_ind[i] - 1)]

        if loadbus_ind[i] == 0:
            LdIq = 0
            LdId = 0
            Pload = 0

        else:
            LdIq = loads[((loadbus_ind[i]-1)*2)]
            LdId = loads[((loadbus_ind[i]-1)*2)+1]
            Pload = p_loads[((loadbus_ind[i]-1)*2)]


        Iin =  nb.cuda.local.array((2), dtype=nb.float32)
        Iout = nb.cuda.local.array((2), dtype=nb.float32)

        j = 0
        while line_arriv[i ,j] > -1:

            Iin[0] = Iin[0] + lines[int(line_arriv[i, j] * 2)]
            Iin[1] = Iin[1] + lines[int(line_arriv[i, j] * 2 + 1)]

            j = j + 1
            if line_arriv[i, j] == -1:
                break

        j = 0
        while line_leave[i, j] > -1:

            Iout[0] = Iout[0] + lines[int(line_leave[i, j] * 2)]
            Iout[1] = Iout[1] + lines[int(line_leave[i, j] * 2 + 1)]

            j = j + 1

            if line_leave[i, j] == -1:
                break


        busst = buses[(i*2):(i*2+2)]
        busparams = p_buses[i]
        GenIq = GenINJQ
        GenId = GenINJD
        Iinq = Iin[0]
        Iind = Iin[1]
        Ioutq = Iout[0]
        Ioutd = Iout[1]
        wconv1 = wgflref

        wC_bus = busparams/ZbaseHV


        # base parameters

        CbaseHV = pu.CbaseHV
        ZbaseHV = pu.ZbaseHV

        Wbase = pu.Wbase



        # -------per unitized values--------------


        C_bus = (wC_bus / Wbase) / CbaseHV


        # ---states--

        Ubus_q = busst[0]
        Ubus_d = busst[1]

        # load calculations
        if Pload > 0:
            Rload = (pu.VratedPhrmsHV ** 2) / Pload
            Rload12 = Rload / ZbaseHV
            LdIq = LdIq + (Ubus_q / Rload12)
            LdId = LdId + (Ubus_d / Rload12)

        else:
            LdIq = LdIq
            LdId = LdId

        # ---------------bus component-----------

        sol[1997+i*2 + caseid] = Wbase * (((2 / (C_bus)) * ((GenIq + Iinq) - (LdIq + Ioutq))) + (wconv1) * Ubus_d)
        sol[1998+i*2 + caseid] = Wbase * (((2 / (C_bus)) * ((GenId + Iind) - (LdId + Ioutd))) - (wconv1) * Ubus_q)

    if tid >= 333 + caseid and tid < 333 +  len(lines)/2    + caseid:
        i = (tid) - 333 - caseid

        linest = lines[(i*2):(i*2+2)]
        wconv1 = wgflref
        lineparams = p_lines[(i*2):(i*2+2)]
        U_in = buses[(bus_arr[i*2]*2-2):(bus_arr[i*2]*2)]
        U_out = buses[(bus_arr[i*2+1]*2-2):(bus_arr[i*2+1]*2)]

        R_12act = lineparams[0] * (pu.ZbaseHV)
        wL_12act = lineparams[1] * (pu.ZbaseHV)

        # base parameters
        # -------per unitized values--------------

        w0 = pu.Wrated / pu.Wbase

        R_12 = R_12act / pu.ZbaseHV
        L_12 = (wL_12act / pu.Wbase) / pu.LbaseHV

        ##element order delta,wint,P,Q,Phid,Phiq,Gamd,Gamq,Icd,Icq,Ufd,Ufq

        U_in_q = U_in[0]
        U_in_d = U_in[1]

        U_out_q = U_out[0]
        U_out_d = U_out[1]

        # ---line 12 states--

        I_12q = linest[0]
        I_12d = linest[1]

        # -------------------Equations-------------------

        sol[2233+i*2 + caseid] = Wbase * (((1 / L_12) * ((U_in_q - U_out_q) - (R_12) * I_12q)) + (wconv1) * I_12d)
        sol[2234+i*2 + caseid] = Wbase * (((1 / L_12) * ((U_in_d - U_out_d) - (R_12) * I_12d)) - (wconv1) * I_12q)

    if tid >= 519 + caseid and tid < 519 +  len(loads)/2    + caseid:
        i = (tid) - 519 - caseid

        loadst = loads[i*2:i*2+2]
        wconv1 = wgflref
        lloadparams = p_loads[(i * 2 + 1)]
        busU = buses[int(ldbus_no_arr[i]*2-2):int(ldbus_no_arr[i]*2)]


        Qload = lloadparams

        Lld = (pu.VratedPhrmsHV ** 2) / (Qload * pu.Wrated)

        # ---------load and HV line 1-2------------

        Lld12 = Lld * pu.Wrated / pu.ZbaseHV


        # ---line 12 states--

        Ubus_q = busU[0]
        Ubus_d = busU[1]
        # ---line 13 states--

        I_L_q = loadst[0]
        I_L_d = loadst[1]

        # -------------------Equations-------------------


        sol[2605+i*2 + caseid] = Wbase * ((Ubus_q/Lld12) + (wconv1) * I_L_d)
        sol[2606+i*2 + caseid] = Wbase * ((Ubus_d/Lld12) - (wconv1) * I_L_q)
@cuda.jit
def RK4_steps(step_reslt, init_vect, calc_result, mux,delta_t):

    i = cuda.grid(1)
    if i < step_reslt.size:
        step_reslt[i] = init_vect[i] + mux * delta_t * calc_result[i]                # First step of RK4
@cuda.jit
def RK4_step_final(X_final, init_vect, step1_res, step2_res, step3_res, step4_res, mux):

    i = cuda.grid(1)
    if i < step_reslt.size:
        X_final[i] = init_vect[i] + (delta_t / 6) * (step1_res[i] + mux * step2_res[i] + mux * step3_res[i] + step4_res[i])


#------initilization of arrays for GPU simulations-------
VARS = cuda.to_device(np.ones((108*4*no_cases),dtype=np.float32))
params = cuda.to_device(np.array(params,dtype=np.float32))
inputs = cuda.to_device(np.array(inputs,dtype=np.float32))
bus_arr = cuda.to_device(np.array(bus_arr,dtype=np.int32))
genbus_ind = cuda.to_device(np.array(genbus_ind,dtype=np.int32))
loadbus_ind = cuda.to_device(np.array(loadbus_ind,dtype=np.int32))
loadbus_no_arr = cuda.to_device(np.array(loadbus_no_arr,dtype=np.int32))
genbus_no_arr = cuda.to_device(np.array(genbus_no_arr,dtype=np.int32))
loadbus_no_arr = cuda.to_device(np.array(loadbus_no_arr,dtype=np.int32))
genbus_no_arr = cuda.to_device(np.array(genbus_no_arr,dtype=np.int32))
line_arriv = cuda.to_device(np.array(line_arriv,dtype=np.int32))
line_leave = cuda.to_device(np.array(line_leave,dtype=np.int32))

F = np.zeros((len(state_arr)),dtype=np.float32)
X_global_mem = cuda.to_device(x)
F_global_mem = cuda.to_device(F)
F_global_mem1 = cuda.to_device(F)
F_global_mem2 = cuda.to_device(F)
F_global_mem3 = cuda.to_device(F)
F_global_mem4 = cuda.to_device(F)

time = np.float32(0)
mux_05 = np.float32(0.5)
mux_2 = np.float32(2)
mux_1 = np.int32(1)
plotstep = 100

step_reslt = np.empty((no_states),dtype=np.float32)
step1_reslt = np.empty((no_states),dtype=np.float32)
step2_reslt = np.empty((no_states),dtype=np.float32)
step3_reslt = np.empty((no_states),dtype=np.float32)
step4_reslt = np.empty((no_states),dtype=np.float32)

init_vect = np.empty((no_states),dtype=np.float32)
calc_result = np.empty((no_states),dtype=np.float32)

sysarr = np.empty((40000),dtype=np.float32)



# Configure the blocks and threads per blocks
threadsperblock = 32
blockspergrid = (x.size + (threadsperblock - 1)) // threadsperblock
print(blockspergrid)

#---configure the PyQTgraph plots for P, Q,V and frequency of individual converters----

win3 = pg.plot()
win3.resize(800, 400)
win3.setBackground('w')

dx = np.arange(118)
dvar = np.empty(118)
plotstatevarderiv = pg.BarGraphItem(x=dx, height=dvar, width=0.6, brush="b")
win3.addItem(plotstatevarderiv)

time_splt = 0
plotwtidow = 10

win2 = pg.GraphicsLayoutWidget()
win2.resize(800, 400)
win2.setBackground('w')
win2.show()

p2 = win2.addPlot(title="GFL Active power (pu)")
p2.showGrid(x=True, y=True, alpha=0.3)
p2.addLegend()

p3 = win2.addPlot(title="GFL Reactive power (pu)")
p3.showGrid(x=True, y=True, alpha=0.3)
p3.addLegend()

p4 = win2.addPlot(title="GFL Frequency (pu)")
p4.showGrid(x=True, y=True, alpha=0.3)
p4.addLegend()

p5 = win2.addPlot(title="GFL Voltage (pu)")
p5.showGrid(x=True, y=True, alpha=0.3)
p5.addLegend()
win2.nextRow()

p6 = win2.addPlot(title="GFR Active power (pu)")
p6.showGrid(x=True, y=True, alpha=0.3)
p6.addLegend()

p7 = win2.addPlot(title="GFR Reactive power (pu)")
p7.showGrid(x=True, y=True, alpha=0.3)
p7.addLegend()

p8 = win2.addPlot(title="GFR Frequency (pu)")
p8.showGrid(x=True, y=True, alpha=0.3)
p8.addLegend()

p9 = win2.addPlot(title="GFR Voltage (pu)")
p9.showGrid(x=True, y=True, alpha=0.3)
p9.addLegend()



curvesPgfl = []
curvesQgfl = []
curvesWgfl = []
curvesVgfl = []

curvesPgfr = []
curvesQgfr = []
curvesWgfr = []
curvesVgfr = []

for idx in range(nPlots):
    curve_P = p2.plot(tx, out, pen=(idx, nPlots * 1.3))
    curve_Q = p3.plot(tx, out, pen=(idx, nPlots * 1.3))
    curve_W = p4.plot(tx, out, pen=(idx, nPlots * 1.3))
    curve_V = p5.plot(tx, out, pen=(idx, nPlots * 1.3))

    p2.legend.addItem(curve_P, 'P' + str(idx + 1) + '(pu)')
    p3.legend.addItem(curve_Q, 'Q' + str(idx + 1) + '(pu)')
    p4.legend.addItem(curve_W, 'W' + str(idx + 1) + '(pu)')
    p5.legend.addItem(curve_V, 'V' + str(idx + 1) + '(pu)')

    curvesPgfl.append(curve_P)
    curvesQgfl.append(curve_Q)
    curvesWgfl.append(curve_W)
    curvesVgfl.append(curve_V)

for idx in range(nPlots):
    curve_P = p6.plot(tx, out, pen=(idx, nPlots * 1.3))
    curve_Q = p7.plot(tx, out, pen=(idx, nPlots * 1.3))
    curve_W = p8.plot(tx, out, pen=(idx, nPlots * 1.3))
    curve_V = p9.plot(tx, out, pen=(idx, nPlots * 1.3))

    p6.legend.addItem(curve_P, 'P' + str(idx + 1) + '(pu)')
    p7.legend.addItem(curve_Q, 'Q' + str(idx + 1) + '(pu)')
    p8.legend.addItem(curve_W, 'W' + str(idx + 1) + '(pu)')
    p9.legend.addItem(curve_V, 'V' + str(idx + 1) + '(pu)')

    curvesPgfr.append(curve_P)
    curvesQgfr.append(curve_Q)
    curvesWgfr.append(curve_W)
    curvesVgfr.append(curve_V)


#----execute the main simulation loop------

while time < 40:

    nlmodel[blockspergrid, threadsperblock](X_global_mem, F_global_mem1, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr,line_arriv,line_leave,VARS)      # initial update of state vector
    RK4_steps[blockspergrid, threadsperblock](F_global_mem, X_global_mem, F_global_mem1, mux_05, delta_t)        # Fisrt step of RK4

    nlmodel[blockspergrid, threadsperblock](F_global_mem, F_global_mem2, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr,line_arriv,line_leave,VARS)      # second update of state vector
    RK4_steps[blockspergrid, threadsperblock](F_global_mem, X_global_mem, F_global_mem2, mux_05, delta_t)        # Second step of RK4

    nlmodel[blockspergrid, threadsperblock](F_global_mem, F_global_mem3, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr,line_arriv,line_leave,VARS)      # second update of state vector
    RK4_steps[blockspergrid, threadsperblock](F_global_mem, X_global_mem, F_global_mem3, mux_1, delta_t)         # Third step of RK4

    nlmodel[blockspergrid, threadsperblock](F_global_mem, F_global_mem4, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr,line_arriv,line_leave,VARS)      # Third update of state vector
    RK4_step_final[blockspergrid, threadsperblock](X_global_mem, X_global_mem, F_global_mem1, F_global_mem2, F_global_mem3, F_global_mem4, mux_2)

    time = time + delta_t

    # measure execution time for 0.1s
    if time > 0 and time < 0.00003:
        start_time = datetime.now()
    if time > 0.1 - 0.00003 and time < 0.1:
        end_time = datetime.now()
        print(end_time-start_time)
    # disable this section to measure real-time simulation performance (plotting in realtime will reduce simulation speed due to CPU-GPU data transfers),
    if time > time_splt:
        if outarr.shape[0] > windowlen:
            outarr = np.vstack((outarr[1:windowlen, :], VARS.copy_to_host()))
        else:
            outarr = np.vstack((outarr, VARS.copy_to_host()))

        if len(tx) > windowlen:
            tx = np.append(tx[1:windowlen], time)
        else:
            tx = np.append(tx, time)

        for i in range(nPlots):
            curvesPgfl[i].setData(tx, outarr[:, i * 4])
            curvesQgfl[i].setData(tx, outarr[:, (i * 4 + 1)])
            curvesWgfl[i].setData(tx, outarr[:, (i * 4 + 2)])
            curvesVgfl[i].setData(tx, outarr[:, (i * 4 + 3)])

        for i in range(nPlots):
            curvesPgfr[i].setData(tx, outarr[:, i * 4 + 216])
            curvesQgfr[i].setData(tx, outarr[:, (i * 4 + 1 + 216)])
            curvesWgfr[i].setData(tx, outarr[:, (i * 4 + 2 + 216)])
            curvesVgfr[i].setData(tx, outarr[:, (i * 4 + 3 + 216)])

        time_splt = time_splt + delta_t * plotstep
        QtWidgets.QApplication.processEvents()



