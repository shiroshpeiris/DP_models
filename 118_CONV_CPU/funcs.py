from nl_model import *
import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import plotly.express as px


@njit
def eigcalc(MAT):
    eigs = la.eig(MAT)
    Y = (eigs[0].imag)
    X = (eigs[0].real)
    return X, Y


def eigplot(X,Y):

    fig = px.scatter(x=X, y=Y)
    config = dict({'scrollZoom': True})
    fig.update_layout(dragmode='pan')
    fig.show(config=config)


# @njit
def integrator(params, inputs, x_hist, delta_t,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr):
    """

    :param Pref1:
    :param x_hist:
    :param delta_t:
    :return:
    """
    fn,VARS = nlmodel(x_hist, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)

    fn1_int = x_hist + 0.5 * delta_t * fn
    fn1,_ = nlmodel(fn1_int, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)

    fn2_int = x_hist + 0.5 * delta_t * fn1
    fn2,_ = nlmodel(fn2_int, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)

    fn3_int = x_hist + delta_t * fn2
    fn3,_ = nlmodel(fn3_int, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)

    x_curr = x_hist + (delta_t / 6) * (fn + 2 * fn1 + 2 * fn2 + fn3)
    return x_curr,VARS

def num_linmodd(params,inputs,x_hist,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr):

    i = 0
    inc = 1e-10
    #jac_x = np.zeros((len(x_hist),len(x_hist)))
    while i < len(x_hist):
        x_pert = x_hist.copy()
        x_pert[i] = x_pert[i] + inc
        fn,_ = nlmodel(x_hist, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)
        fn_inc,_ = nlmodel(x_pert, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)
        jac_x = (fn_inc - fn)/(inc)
        if i == 0:
            jac = jac_x.copy()
        if i > 0:
            jac = np.vstack((jac, jac_x))
        i = i + 1
    jac = np.asarray(jac)
    jac = np.transpose(jac)
    eigs = la.eig(jac)
    Y = (eigs[0].imag)
    X = (eigs[0].real)
    # print(eigs[0])
    return X, Y, jac



def pfcalc(A):
    eigs = la.eig(A)
    x_labels = np.round(eigs[0], 3)
    x_labels = x_labels.astype('U25')
    y_labels = np.arange(0, len(eigs[0]), 1)
    P = (eigs[1])
    QT = inv(P)

    Q = QT.transpose()

    pf = np.zeros(np.shape(A), dtype = 'complex_')
    j = 0

    # j is index on columns (modes)
    # i is index on rows (states)

    while j < len(eigs[0]):
        i = 0
        while i < len(eigs[0]):
            pf[i, j] = Q[i, j]*P[i, j]
            i = i + 1

        j = j + 1
    pfabs = np.abs(pf)
    pfabs = np.round(pfabs,1)
    print(pfabs)
    fig = px.imshow(pfabs, color_continuous_scale = 'Jet',
                    labels=dict(x="Eigenvalues", y="State Number", color="Participation"),
                    x=x_labels,
                    y=y_labels
                    )
    # config = dict({'scrollZoom': True})
    # fig.update_layout(dragmode='pan')
    # fig.show(config=config)
    fig.show()

def pfgen(A):
    eigs = la.eig(A)
    P = (eigs[1])
    QT = inv(P)

    Q = QT.transpose()

    pf = np.zeros(np.shape(A), dtype='complex_')
    j = 0

    # j is index on columns (modes)
    # i is index on rows (states)

    while j < len(eigs[0]):
        i = 0
        while i < len(eigs[0]):
            pf[i, j] = Q[i, j] * P[i, j]
            i = i + 1

        j = j + 1
    pfabs = np.abs(pf)
    # pfabs = np.round(pfabs,1)

    return pfabs


def pfsort(pfabs, eigno, novals):
    pfarr = pfabs[:, eigno]
    ind = np.flip(np.argsort(pfabs[:, eigno]))
    # print(pfarr, ind)
    i = 0
    if novals > len(pfarr):
        novals = len(pfarr)
    pfordered = np.zeros((2, novals))

    while i < novals:
        pfordered[0, i] = ind[i]
        pfordered[1, i] =  np.round(pfarr[ind[i]],3)
        i+=1
    return pfordered

def initsolver(x0,params,inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr):
    x, _ = nlmodel(x0, params, inputs,bus_arr,genbus_ind,loadbus_ind,loadbus_no_arr,genbus_no_arr)
    return x

