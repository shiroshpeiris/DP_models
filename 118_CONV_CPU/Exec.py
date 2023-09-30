#------------------import libraries-------------------------
import sys
# from numba import jit, config
# config.DISABLE_JIT = True

from numpy import linalg as LA
from threading import Thread
import math
from numpy import genfromtxt
from Paramload import readparams


from funcs import *
from Init_Load import *
from PUdefs import *
from scipy.optimize import *
import pyqtgraph as pg
from PyQt5 import QtWidgets


import PySimpleGUI as sg
import pandas as pd

# NEED TO reduce THE gfl transformer impedances bu 1/10th to get the GFL/Sync Cond mode working

pertb_inp=4
sim_length = 200000


#-------------------------- control params------------------------------------------


#-----------------------initialize global variables--------------------------

#-----sim parameter variables----------
varobs_y = 1
varobs_x = 0
windowlen = 500
delta_t = 3e-5

plotstep = 100

#--------conditional variables--------------
showeigs = 0
paramset = 0
novals = 20
showpfvals = 0
changevarobs = 0
showpf = 0
recsnapshot = 0
plotderivs = 0
plottimeasx = 1
initsim = 0
plotvarnum = 432
nPlots = 54

def systemfunc():
    pi = math.pi

    # ---------------------------------get initial conditions from File-----------------------------
    x = initload()
    # x = np.ones((967))*1e-20
    ind = 0



    global params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr

    params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr = readparams()
    x= genfromtxt('init_snapshot.csv', delimiter=',')

    #------------solve for initialization------------------
    # x0 = np.ones((113)) * 1e-20
    # x = fsolve(func = initsolver, x0 = x, args=(params,inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr))#np.ones((74)) * 1e-20 #
    # x = leastsq(func = initsolver, x0 = x, args=(params,inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr),ftol=1.49012e-10, xtol=1.49012e-10)               # leastsq works fine for the case where it doesnt solve with fsolve due to the 60Hz oscillatory mode in the detailed non aggregate system
    x = root(fun = initsolver,method='hybr', x0 = x, args=(params,inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr))#np.ones((74)) * 1e-20 #
    x=x.x


    #-----------------initialize graph objects----------------------

    dx = np.arange(len(x))
    dvar = np.empty([len(x)])

    #-----Eigenvalue Window--------------------
    win1 = pg.GraphicsLayoutWidget()
    win1.resize(800, 400)
    win1.setBackground('w')
    win1.show()
    #-----Transient Response Window--------------------
    win2 = pg.GraphicsLayoutWidget()
    win2.resize(800, 400)
    win2.setBackground('w')
    win2.show()
    #-----State Var Derivative Window--------------------
    win3 = pg.plot()
    win3.resize(800, 400)
    win3.setBackground('w')
    plotstatevarderiv = pg.BarGraphItem(x = dx, height = dvar,width = 0.6, brush="b")
    win3.addItem(plotstatevarderiv)



    p1 = win1.addPlot()
    p1.showGrid(x=True, y=True, alpha=0.3)

    X=[]
    Y=[]
    out = [0]
    tx = [0]

    ploteignum = p1.plot(X, Y, symbolBrush=(255, 0, 0), symbol='o',pen=pg.mkPen(None))

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









    # data_PSCAD = np.genfromtxt("PSCAD_data.txt", delimiter=",", names=["x", "y"])
    # plotpscad = p4.plot(data_PSCAD['x'], data_PSCAD['y'], pen="b", name='PSCAD Plot')  # ----------------plot PSCAD response for validation SAved--------------------

    global text
    text = {}
    r = np.ones(76)
    for number in range(0, len(r)):
        text["text%i" % number] = pg.TextItem('[%0.1f, %0.1f]' % (0, 0))
        text["text%i" % number].setColor('k')
        p1.addItem(text["text"+str(number)])

    w = pg.TableWidget()
    w.show()
    w.resize(600, 900)
    w.setWindowTitle('pyqtgraph example: TableWidget')




    def solver(x_hist):
        """

        :param Pref1:
        :param x_hist:
        :param delta_t:
        :return:
        """
        # assign history term value
        time = 0
        out = []
        tx = []

        global windowlen,plotvarnum

        outarr = np.zeros([0,plotvarnum])
        print(outarr)
        x_temp = np.zeros((len(x_hist)))
        time_splt = 0
        out_eigtest = np.zeros((len(x_hist)))

        global delta_t
        while time < sim_length:

            #--------------Re initialize the simulation----------
            global initsim,params,inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr

            if initsim == 1:
                time = 0
                out = []
                tx = []
                x_temp = np.zeros((len(x_hist)))
                time_splt = 0
                outarr = np.zeros([0, plotvarnum ])

                params, inputs, bus_arr, genbus_ind, loadbus_ind, loadbus_no_arr, genbus_no_arr = readparams()
                x = genfromtxt('init_snapshot.csv', delimiter=',')
                x = fsolve(func=initsolver, x0=x, args=(params, inputs, bus_arr, genbus_ind, loadbus_ind, loadbus_no_arr, genbus_no_arr))  # np.ones((74)) * 1e-20 #

                x_hist = x
                initsim = 0

            # ---------------------load parameters from excel
            global paramset
            if paramset == 1:
                params, inputs, bus_arr, genbus_ind, loadbus_ind, loadbus_no_arr, genbus_no_arr = readparams()
                paramset = 0
            #------------------create simulation snapshot-------------------------
            global recsnapshot
            if recsnapshot == 1:
                 np.savetxt("init_snapshot.csv", x_hist)
                 print("snapshot recorded")
                 recsnapshot = 0
            # call 4th order RK method
            x_hist,VARS  = integrator(params, inputs, x_hist, delta_t,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)
            time = time + delta_t

            #----------plot within plot step------------------
            if time > time_splt:
                global changevarobs
                if changevarobs == 1:
                    out = []
                    tx = []
                    changevarobs = 0

                if outarr.shape[0] > windowlen:
                    outarr = np.vstack((outarr[1:windowlen,:],VARS))
                else:
                    outarr = np.vstack((outarr,VARS))

                global plottimeasx
                if plottimeasx == 1:
                    if len(tx) > windowlen:
                        tx = np.append(tx[1:windowlen], time)
                    else:
                        tx = np.append(tx, time)

                if plottimeasx == 0:
                    if len(tx) > windowlen:
                        tx = np.append(tx[1:windowlen], VARS[varobs_x])
                    else:
                        tx = np.append(tx, VARS[varobs_x])
                #-------------show all datapoints on plot-------------------------
                #--display solver solution-----
                #


                num_entries = 27  # Number of entries
                data = []
                for i in range(num_entries):
                    base_index = i * 8
                    entry = (
                        'P' + str(i * 2 + 1), VARS[base_index], 'Q' + str(i * 2 + 1), VARS[base_index + 1], 'V' + str(i * 2 + 1), VARS[base_index + 3],
                        'P' + str(i * 2 + 2), VARS[base_index + 4], 'Q' + str(i * 2 + 2), VARS[base_index + 5],'V' + str(i * 2 + 2), VARS[base_index + 7]
                    )
                    data.append(entry)
                #
                # for i in range(num_entries):
                #     idx = i * 4  # Calculate the starting index for VARS based on entry number
                #     entry = ('P' + str(i + 1), VARS[idx], 'Q' + str(i + 1), VARS[idx + 1],
                #              'V' + str(i + 1), VARS[idx + 3 + 216])
                #     data.append(entry)



                w.setData(data)

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


                win2.setWindowTitle("Timestep: " + str(delta_t))







                global plotderivs
                if plotderivs == 1:
                    plotstatevarderiv.setOpts(height = np.abs((x_temp - x_hist) / delta_t))
                    # x_arr = x_hist[863:1099]
                    # data = np.zeros((len(x_arr)))
                    # j = 0
                    # i = 0
                    # while i < len(x_arr) - 1:
                    #     data[j] = np.sqrt(x_arr[i] ** 2 + x_arr[i + 1] ** 2)
                    #     i = i + 2
                    #     j = j + 1
                    #
                    # plotstatevarderiv.setOpts(height = data)

                #-------------calculate eigenvalues dynamically-------------------------
                global showeigs
                if showeigs == 1:
                    Xnum, Ynum, jac = num_linmodd(params, inputs, x_hist,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)
                    ploteignum.setData(Xnum, Ynum)
                    win1.setWindowTitle("Eigenvalue Plot")
                    if showpfvals == 1:
                        pfabs = pfgen(jac)
                        eigno = 0
                        for eigtextref in text:
                            pfordered = pfsort(pfabs, eigno, novals)

                            states = pfordered[0, :]
                            states = states.astype(int)
                            pfvals = pfordered[1, :]

                            res = "\n".join("State:{}   PF:{}".format(x, y) for x, y in zip(states, pfvals))

                            text[eigtextref].setPos(Xnum[eigno], Ynum[eigno])
                            text[eigtextref].setText('Values: [%0.3f, %0.3f]\nDamping ratio:[%0.3f]\n%s' % (
                            Xnum[eigno], Ynum[eigno], (-Xnum[eigno] / (np.sqrt(Xnum[eigno] ** 2 + Ynum[eigno] ** 2))),
                            res))
                            eigno = eigno + 1
                    elif showpfvals == 0:
                        eigno = 0
                        for eigtextref in text:
                            text[eigtextref].setPos(Xnum[eigno], Ynum[eigno])
                            text[eigtextref].setText('Values: [%0.3f, %0.3f]\nDamping ratio:[%0.3f]' % (
                            Xnum[eigno], Ynum[eigno], (-Xnum[eigno] / (np.sqrt(Xnum[eigno] ** 2 + Ynum[eigno] ** 2)))))
                            eigno = eigno + 1
                    showeigs =0
                    #---------------plot eiganvalues and participation factors-------------------
                    global showpf
                    if showpf == 1:
                        if np.var((x_temp - x_hist) / delta_t) < 1e-10:
                            print("Variance of state Derivatives = ", np.var((x_temp - x_hist) / delta_t))
                            pfcalc(jac)
                            eigplot(Xnum, Ynum)
                        else:
                            print("Variance of state Derivatives = ", np.var((x_temp - x_hist) / delta_t))
                            print("not reached steady state")

                        showpf = 0

                p1.enableAutoRange("xy", False)
                QtWidgets.QApplication.processEvents()
                time_splt = time_splt + delta_t * plotstep
                x_temp = x_hist

        return tx, out


    solver(x)
    win2.close()


def controlgui():

    def setparams():
        global paramset
        paramset = 1

    def recsnapshot():
        global recsnapshot
        recsnapshot = 1

    def changevarobs(varsel,varobsx,varobsy):
        global changevarobs
        global varobs_y
        global varobs_x
        global plottimeasx
        changevarobs = 1
        if varsel == True:
            plottimeasx = 1
            varobs_y = int(varobsy)
        elif varsel == False:
            plottimeasx = 0
            varobs_y = int(varobsy)
            varobs_x = int(varobsx)

    # def disable_input_varobsobsx():
    #     if plotvarx_sel.get() == 1:
    #         E9.config(state = 'disabled')
    #     elif plotvarx_sel.get() == 2:
    #         E9.config(state = 'normal')


    def showeigs(var):
        global showeigs
        if var == True:
            showeigs = 1
        elif var == False:
            showeigs = 0

    def showpfvals(var):
        global showpfvals
        if var == True:
            showpfvals = 1
        elif var == False:
            showpfvals = 0

    def plotderivs(var):
        global plotderivs
        if var == True:
            plotderivs = 1
        elif var == False:
            plotderivs = 0

    def genpf():
        global showpf
        showpf = 1
    def reinit():
        global initsim
        initsim = 1


    def setwindowlen(var):
        global windowlen
        windowlen = int(var)

    def setdt(var):
        global delta_t
        delta_t = float(var)

    def setpltwndw(var):
        global plotstep
        plotstep = int(var)


    def main_gui():


        layout = [  [sg.Text("Plotting Variables", text_color= "Blue")],     # Part 2 - The Layout
                    [sg.Text("Y axis Plotting Variable"), sg.Input(varobs_y, size = (10,1))],
                    [sg.Text("X axis Plotting Variable")],
                    [sg.Radio("Time", "RADIO1", default=True, change_submits = True,  key = "seltime")],
                    [sg.Radio("State Variable", "RADIO1",default=False, change_submits = True, key = "selstatevar"), sg.Input(0, size = (10,1))],
                    [sg.Button('Clear and Plot')],
                    [sg.Text("Simulation parameters", text_color= "Blue")],
                    [sg.Text("Plot window length"), sg.Input(windowlen, size = (10,1)), sg.Button('Update Window size')],
                    [sg.Text("Simulation timestep"), sg.Input(delta_t, size = (10,1)), sg.Button('Update Timestep')],
                    [sg.Text("Plotstep"), sg.Input(plotstep, size = (10,1)), sg.Button('Update Plotstep')],
                    [sg.Text("View Analysis", text_color= "Blue")],
                    [sg.Checkbox("Generate Eigenvalues on Runtime", change_submits = True, key='reltimeeig'),sg.Checkbox("Show Participation and States", change_submits=True, key='shwpfvals')],
                    [sg.Checkbox("View Statevar Derivatives", change_submits = True, key='viewderiv'),sg.Button('Generate PF and Eigenvalue plots')],
                    [sg.Button('Update Parameters')],
                    [sg.Button("Re Initialize")],
                    [sg.Button('Generate initialization snapshot'),],
                    ]

        window = sg.Window('Control GUI', layout)

        # Display and interact with the Window using an Event Loop
        while True:
            event, values = window.read()
            s = pd.Series(values)
            values = s.values
            print(values)
            if event == 'Update Parameters':
                setparams()
            if event == 'Update Window size':
                setwindowlen(values[4])
            if event == 'Update Timestep':
                setdt(values[5])
            if event == 'Update Plotstep':
                setpltwndw(values[6])
            if event == 'reltimeeig':
                showeigs(values[7])
            if event == 'shwpfvals':
                showpfvals(values[8])
            if event == 'Generate PF and Eigenvalue plots':
                genpf()
            if event == 'viewderiv':
                plotderivs(values[9])
            if event == 'Generate initialization snapshot':
                recsnapshot()
            if event == 'seltime':
                changevarobs(values[1],values[3],values[0])
            if event == 'selstatevar':
                changevarobs(values[1],values[3],values[0])
            if event == 'Clear and Plot':
                changevarobs(values[1],values[3],values[0])
            if event == 'Re Initialize':
                reinit()
            # See if user wants to quit or window was closed
            if event == sg.WINDOW_CLOSED or event == 'Quit':
                break
        # Finish up by removing from the screen
        window.close()
    main_gui()

if __name__ == "__main__":
    thread_gui = Thread(target=controlgui)
    thread_sys = Thread(target=systemfunc)
    thread_gui.start()
    thread_sys.start()

