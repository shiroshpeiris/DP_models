import numpy as np
from numpy import genfromtxt

def initload():

    X_GFL = np.ones((972))*1e-20    #genfromtxt('init_snapshot_GFL.csv', delimiter=',')
    X_GFR = np.ones((918))*1e-20    #genfromtxt('init_snapshot_GFR.csv', delimiter=',')
    X_ANG_GFL = np.ones((53))*1e-20#genfromtxt('init_snapshot_ANG_GFL.csv', delimiter=',')
    X_ANG_GFR = np.ones((54))*1e-20    #genfromtxt('init_snapshot_ANG_GFR.csv', delimiter=',')

    X_NW = genfromtxt('init_snapshot_NW.csv', delimiter=',')

    X_init = np.concatenate((X_GFL,X_GFR,X_ANG_GFL,X_ANG_GFR,X_NW))
    print(len(X_init))
    return X_init
