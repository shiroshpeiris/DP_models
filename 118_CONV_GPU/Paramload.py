import pandas as pd
import numpy as np
import csv
# sys.path.insert(0, 'Parameters')

def readparams():
    import pandas as pd
    import numpy as np

    cols_buscap = [0, 2, 8, 9, 10]
    cols_loaddata = [0, 6, 7]
    cols_busdata = [0]
    cols_gendata = [0]

    cols_genpardata = np.arange(40)
    cols_geninpdata = [41,42,43,44,45]

    cols_gflpardata = np.arange(21)
    cols_gflinpdata = [22,23,24,25]

    cols_gfrpardata = np.arange(20)
    cols_gfrinpdata = [21,22,23,24,25]


    # only read specific columns from an excel file
    df = pd.read_excel('Parameters/Linedata.xls', usecols=cols_buscap)
    df2 = pd.read_excel('Parameters/Loaddata.xls', usecols=cols_loaddata)
    df3 = pd.read_excel('Parameters/GenData.xls', usecols=cols_gendata)
    df4 = pd.read_excel('Parameters/Busdata.xls', usecols=cols_busdata)


    genpars = []
    geninps = []
    genscalers = []

    gflpars = []
    gflinps = []
    gflscalers = []

    gfrpars = []
    gfrinps = []
    gfrscalers = []

    busparams = []
    lineparams = []
    ldparams = []

    # -------Read and store Generator Parameters and inputs---------------------
    genpar_params = pd.read_excel('Parameters/GenParams.xls',usecols=cols_genpardata,sheet_name='SM')
    i = 0
    while i <= len(genpar_params.index) - 1:
        pars = genpar_params.values[i].tolist()
        genpars = np.hstack((genpars, pars))
        i = i + 1

    genpar_inputs = pd.read_excel('Parameters/GenParams.xls', usecols=cols_geninpdata,sheet_name='SM')
    i = 0
    while i <= len(genpar_inputs.index) - 1:
        inps = genpar_inputs.values[i].tolist()
        geninps.append(inps[0])
        geninps.append(inps[1])
        geninps.append(inps[2])
        geninps.append(inps[3])
        genscalers.append(inps[4])
        i = i + 1

    gflpar_params = pd.read_excel('Parameters/GenParams.xls',usecols=cols_gflpardata,sheet_name='GFL')
    i = 0
    while i <= len(gflpar_params.index) - 1:
        pars = gflpar_params.values[i].tolist()
        gflpars = np.hstack((gflpars, pars))
        i = i + 1

    gflpar_inputs = pd.read_excel('Parameters/GenParams.xls', usecols=cols_gflinpdata,sheet_name='GFL')
    i = 0
    while i <= len(gflpar_inputs.index) - 1:
        inps = gflpar_inputs.values[i].tolist()
        gflinps.append(inps[0])
        gflinps.append(inps[1])
        gflinps.append(inps[2])
        gflscalers.append(inps[3])
        i = i + 1

    gfrpar_params = pd.read_excel('Parameters/GenParams.xls',usecols=cols_gfrpardata,sheet_name='GFR')
    i = 0
    while i <= len(gfrpar_params.index) - 1:
        pars = gfrpar_params.values[i].tolist()
        gfrpars = np.hstack((gfrpars, pars))
        i = i + 1

    gfrpar_inputs = pd.read_excel('Parameters/GenParams.xls', usecols=cols_gfrinpdata,sheet_name='GFR')
    i = 0
    while i <= len(gfrpar_inputs.index) - 1:
        inps = gfrpar_inputs.values[i].tolist()
        gfrinps.append(inps[0])
        gfrinps.append(inps[1])
        gfrinps.append(inps[2])
        gfrinps.append(inps[3])
        gfrscalers.append(inps[4])
        i = i + 1


    # print(genpars)
    # print(geninps)

    # ------Read the bus suceptaces and add to the array-----
    i = 0
    while i <= len(df4.index) - 1:
        busno = df4.values[i].tolist()
        # print((df['B'].where(df['From Number'] == i ).dropna()).tolist())
        bus_suscept1 = sum((df['B'].where(df['From Number'] == busno[0]).dropna()).tolist())
        bus_suscept2 = sum((df['B'].where(df['To Number'] == busno[0]).dropna()).tolist())
        bus_suscept = bus_suscept1 + bus_suscept2
        # print(bus_suscept)
        busparams.append(bus_suscept)
        i = i + 1

    # ------Read the R and X values and add to the array--------
    i = 0
    while i < len(df.index):
        line_rx = df.values[i].tolist()
        lineparams.append(line_rx[2])
        lineparams.append(line_rx[3])
        i = i + 1

    # -------Read the load values and add to the array----------
    i = 0
    while i < len(df2.index):
        bus_ld = df2.values[i].tolist()
        ldparams.append(bus_ld[1])
        ldparams.append(bus_ld[2])
        i = i + 1

    busparams = np.array((busparams), dtype=float)
    lineparams = np.array((lineparams), dtype=float)
    ldparams = np.array((ldparams), dtype=float)

    # print(busparams)
    # print(lineparams)
    # print(ldparams)

    genpars = np.array(genpars)
    geninps = np.array(geninps)
    linedata = np.array(lineparams)
    loaddata = np.array(ldparams)
    busdata = np.array(busparams)


    ####################################################################################
    #index Generator
    ####################################################################################


    i = 0
    bus_arr = []
    while i < len(df.index):
        bus_inout = df.values[i].tolist()
        bus_arr = (np.append(bus_arr, bus_inout[0:2]))
        i = i + 1
    bus_arr = bus_arr.astype(int)

    # print(bus_arr)

    i = 0
    loadbus_no_arr = []
    while i < len(df2.index):
        bus_ldno = df2.values[i].tolist()
        loadbus_no_arr = (np.append(loadbus_no_arr, bus_ldno[0]))
        i = i + 1
    loadbus_no_arr = loadbus_no_arr.astype(int)

    # print(loadbus_no_arr)

    i = 0
    genbus_no_arr = []
    while i < len(df3.index):
        gen_no = df3.values[i].tolist()
        genbus_no_arr = (np.append(genbus_no_arr, gen_no[0]))
        i = i + 1
    genbus_no_arr = genbus_no_arr.astype(int)

    # print(genbus_no_arr)

    i = 1
    j = 1
    k = 1
    genbus_ind = []
    loadbus_ind = []
    while i <= len(busparams):
        bus_load = df2['Number of Bus'].where(df2['Number of Bus'] == i).dropna().tolist()
        bus_gen = df3['Gen Number of Bus'].where(df3['Gen Number of Bus'] == i).dropna().tolist()

        if bus_load:
            loadbus_ind.append(j)
            j = j + 1
        else:
            loadbus_ind.append(0)

        if bus_gen:
            genbus_ind.append(k)
            k = k + 1
        else:
            genbus_ind.append(0)
        i = i + 1


    genbus_ind = np.array(genbus_ind)
    loadbus_ind = np.array(loadbus_ind)
    loadbus_no_arr = np.array(loadbus_no_arr)
    genbus_no_arr = np.array(genbus_no_arr)

    null = np.zeros((2))
    pars = np.concatenate((busdata,loaddata,linedata,gflpars,gfrpars,gflscalers,gfrscalers,null))
    srcinps = np.concatenate((gflinps,gfrinps))

    ##----Arrivng----
    line_arriv = np.ones((118, 6)) * -2             # provide additional headroom in the array when initializing, or the extraction loop may miss values
    k = 0
    while k < 119:
        i = 1
        j = 0

        while i < len(bus_arr):
            if bus_arr[i] == k:
                line_arriv[k - 1, j] = i - 1
                j = j + 1
            i = i + 2
        k = k + 1
    line_arriv = line_arriv / 2
    # print(line_arriv)


    ##----LEaving----
    line_leave = np.ones((118, 8)) * -2             # provide additional headroom in the array when initializing, or the extraction loop may miss values
    k = 0
    while k < 119:
        i = 0
        j = 0

        while i < len(bus_arr):
            if bus_arr[i] == k:
                line_leave[k - 1, j] = i
                j = j + 1
            i = i + 2
        k = k + 1
    line_leave = line_leave/2
    # print(line_leave)


    return pars,srcinps,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr,line_arriv,line_leave

pars,srcinps,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr,line_arriv,line_leave = readparams()

#genbus ind,loadbus ind are different from the CPU models
#line_arrive and line_leave are different from the models that utilize the tensor for line assignment to bus functions


print(loadbus_ind)