from numba import njit
from PUdefs import *
import SourcemodelSmach as unit2
import SourcemodelGFoll as unit3
import SourceDroopgfor as unit4

import numpy as np

import Bus_mod as Bus
import Line_mod as Line
import Lloadmod as Indload
from PUdefs import Wrated

@njit#(cache = True)

def nlmodel(x, params, inputs,bus_arr,genbus_ind,loadbus_ind,ldbus_no_arr,genbus_no_arr):

    # ===Split the array of states based on individual function lengths=========================
    gfl = x[0:972]
    gfr = x[(len(gfl)):((len(gfl)) + 918)]

    angles = x[len(gfr)+len(gfl):(len(gfr)+len(gfl)+107)]
    buses = x[(len(gfr)+len(gfl)+len(angles)):(len(gfr)+len(gfl)+(len(angles)) + 236)]
    lines = x[(len(gfr)+len(gfl)+len(buses)+len(angles)):(len(gfr)+len(gfl)+(len(buses)+len(angles)) + 372)]
    loads = x[(len(gfr)+len(gfl)+len(lines)+len(buses)+len(angles)):(len(gfr)+len(gfl)+(len(lines)+len(buses)+len(angles)) + 198)]

    sol= np.zeros((len(x)))
    gflsols = sol[0:972]
    gfrsols = sol[(len(gfl)):((len(gfl)) + 918)]

    delta = sol[len(gfr) + len(gfl):(len(gfr) + len(gfl) + 107)]
    solbus = sol[(len(gfr) + len(gfl) + len(angles)):(len(gfr) + len(gfl) + (len(angles)) + 236)]
    solline = sol[(len(gfr) + len(gfl) + len(buses) + len(angles)):(len(gfr) + len(gfl) + (len(buses) + len(angles)) + 372)]
    solload = sol[(len(gfr) + len(gfl) + len(lines) + len(buses) + len(angles)):(len(gfr) + len(gfl) + (len(lines) + len(buses) + len(angles)) + 198)]



    GflIQ = np.zeros(int((len(gflsols)/18)))
    GflID = np.zeros(int((len(gflsols)/18)))

    GfrIQ = np.zeros(int((len(gfrsols)/17)))
    GfrID = np.zeros(int((len(gfrsols)/17)))

    wgfl = np.zeros(int((len(gflsols)/18)))
    wgfr = np.zeros(int((len(gfrsols)/17)))

    VAR_GFL=np.zeros(int((len(gflsols))*4/18))
    VAR_GFR=np.zeros(int((len(gfrsols))*4/17))


    linelinks = np.zeros((2,len(genbus_ind)+1, len(genbus_ind)+1))              #  genbus_ind is simpl used as its length is equal to the number of buses in the system,ldbis_ind also could be used

    i = 0
    while i < len(bus_arr) / 2:
        if linelinks[0,bus_arr[i*2], bus_arr[i*2+1]] and linelinks[1,bus_arr[i*2], bus_arr[i*2+1]] != 0:
            linelinks[0,bus_arr[i*2], bus_arr[i*2+1]] = linelinks[0,bus_arr[i*2], bus_arr[i*2+1]] + lines[i*2]
            linelinks[1,bus_arr[i*2], bus_arr[i*2+1]] = linelinks[1,bus_arr[i*2], bus_arr[i*2+1]] + lines[i*2+1]
        else:
            linelinks[0,bus_arr[i*2], bus_arr[i*2+1]] = lines[i*2]
            linelinks[1,bus_arr[i*2], bus_arr[i*2+1]] = lines[i*2+1]
        i = i + 1

    # ====Specify the parameters for the individual functions========
    p_buses = params[0:118]
    p_loads = params[len(p_buses):len(p_buses)+198]
    p_lines = params[len(p_buses)+len(p_loads):len(p_buses)+len(p_loads)+372]
    p_gfl = params[len(p_buses)+len(p_loads)+len(p_lines):len(p_buses)+len(p_loads)+len(p_lines)+1134]
    p_gfr = params[len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl):len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+1080]

    p_gflscalers = params[len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr):len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr)+54]
    p_gfrscalers = params[len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr)+len(p_gflscalers):len(p_buses)+len(p_loads)+len(p_lines)+len(p_gfl)+len(p_gfr)+len(p_gflscalers)+54]
    null = params[3010:3012]

    inputs_gfl = inputs[0:162]
    inputs_gfr = inputs[len(inputs_gfl):len(inputs_gfl)+216]



    # ======calculate frequency related parameters needed for decoupled components in modular sections

    i = 0
    while i < len(gfl)/18:
        wgfl[i] = inputs_gfl[i*3+2] + gfl[i*18+7] * p_gfl[i*21+9] + gfl[i*18+2] * p_gfl[i*21+10]
        i = i + 1

    i = 0
    while i < len(gfr)/17:
        wgfr[i] = inputs_gfr[i*4+3] + p_gfr[i*20+3] * gfr[i*17]
        i = i + 1


    # ===========Reference frame Transformed currents from generating sources====================
    gfl_angles= angles[0:53]
    gfr_angles = angles[53:107]

    gfl_delta= delta[0:53]
    gfr_delta = delta[53:107]


    i =0
    while i < len(gfl) / 18:
        if i == 0:
            GflIQ[i] = p_gflscalers[i] * gfl[i + 9]
            GflID[i] = p_gflscalers[i] * gfl[i + 8]
        else:
            GflIQ[i] = p_gflscalers[i] * (+gfl[i * 18 + 9] * np.cos(gfl_angles[i-1]) - gfl[i * 18 + 8] * np.sin(gfl_angles[i-1]))
            GflID[i] = p_gflscalers[i] * (+gfl[i * 18 + 9] * np.sin(gfl_angles[i-1]) + gfl[i * 18 + 8] * np.cos(gfl_angles[i-1]))
        i = i + 1

    i =0
    while i < len(gfr) / 17:
        GfrIQ[i] = p_gfrscalers[i] * (+gfr[i * 17 + 11] * np.cos(gfr_angles[i]) - gfr[i * 17 + 12] * np.sin(gfr_angles[i]))
        GfrID[i] = p_gfrscalers[i] * (+gfr[i * 17 + 11] * np.sin(gfr_angles[i]) + gfr[i * 17 + 12] * np.cos(gfr_angles[i]))
        i = i + 1



    #===========================line functions=======================================
    i = 0
    while i < len(solline)/2:
        solline[i*2:i*2+2] = Line.linemodel(lines[(i*2):(i*2+2)], wgfl[0],p_lines[(i*2):(i*2+2)], buses[(bus_arr[i*2]*2-2):(bus_arr[i*2]*2)], buses[(bus_arr[i*2+1]*2-2):(bus_arr[i*2+1]*2)])
        i = i + 1
    #===========================bus functions=======================================
    i = 0
    g = 0
    l = 0
    bus_arriving = np.zeros((2))
    bus_leaving =  np.zeros((2))
    while i < len(solbus)/2:
        if genbus_ind[i] == 0:
            GenINJQ = 0
            GenINJD = 0
        else:
            GenINJQ = GflIQ[g]+GfrIQ[g]
            GenINJD = GflID[g]+GfrID[g]
            g = g + 1

        if loadbus_ind[i] == 0:
            load_parm = null
            load_st = null
        else:
            load_parm = p_loads[(l*2):(l*2+2)]
            load_st = loads[(l*2):(l*2+2)]
            l = l + 1

        bus_arriving[0] = np.sum(linelinks[0][:,i+1])
        bus_arriving[1] = np.sum(linelinks[1][:,i+1])

        bus_leaving[0] = np.sum(linelinks[0][i+1])
        bus_leaving[1] = np.sum(linelinks[1][i+1])


        solbus[i*2:i*2+2] = Bus.busmodel(buses[(i*2):(i*2+2)], wgfl[0], p_buses[i], load_parm, GenINJQ, GenINJD, load_st, bus_arriving, bus_leaving)
        i = i + 1
    #===========================load functions=======================================
    i = 0
    while i < len(solload)/2:
        solload[i*2:i*2+2] = Indload.Lloadmodel(loads[i*2:i*2+2], wgfl[0], p_loads[(i*2+1)], buses[int(ldbus_no_arr[i]*2-2):int(ldbus_no_arr[i]*2)])
        i = i+1
    #===========================generator functions=======================================

    i = 0
    while i < len(gflsols)/18:
        if i == 0:
            gflsols[i*18:i*18+18], VAR_GFL[i*4:i*4+4] = unit3.unitmodel(gfl[i*18:i*18+18], p_gfl[i*21:i*21+21], inputs_gfl[i*3:i*3+3], buses[int(genbus_no_arr[i]*2-2)], buses[int(genbus_no_arr[i]*2-1)], 0,p_gflscalers[i],i)
        else:
            gflsols[i*18:i*18+18], VAR_GFL[i*4:i*4+4] = unit3.unitmodel(gfl[i*18:i*18+18], p_gfl[i*21:i*21+21], inputs_gfl[i*3:i*3+3], buses[int(genbus_no_arr[i]*2-2)], buses[int(genbus_no_arr[i]*2-1)], gfl_angles[i-1],p_gflscalers[i],i)

        i = i + 1

    i = 0
    while i < len(gfrsols)/17:
        gfrsols[i*17:i*17+17], VAR_GFR[i*4:i*4+4] = unit4.unitmodel(gfr[i*17:i*17+17], p_gfr[i*20:i*20+20], inputs_gfr[i*4:i*4+4], buses[int(genbus_no_arr[i]*2-2)], buses[int(genbus_no_arr[i]*2-1)], gfr_angles[i],p_gfrscalers[i],i)
        i = i + 1

    #===========================reference angle functions=======================================
    i = 0
    while i < len(gfl_delta):
        gfl_delta[i] = Wrated * (wgfl[i+1] - wgfl[0])
        i = i+1


    i = 0
    while i < len(gfr_delta):
        gfr_delta[i] = Wrated * (wgfr[i] - wgfl[0])
        i = i+1

    # gfl_delta = gfl_delta * 0
    # gfr_delta = gfr_delta * 0
    delta = np.concatenate((gfl_delta,gfr_delta))
    sols = np.concatenate((gflsols,gfrsols,delta,solbus,solline,solload))
    VARS = np.concatenate((VAR_GFL,VAR_GFR))

    # freqs = np.array((wgfl[0],wgfl[1],wgfl[2]))
    # print("===========")
    # print(VAR)
    # print("===========")
    # print(sols)
    return sols, VARS
