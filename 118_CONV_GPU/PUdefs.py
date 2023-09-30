import numpy as np

#PU values

VratedLLrmsHV = 230
VratedLLrmsLV = 0.69

Wrated = 376.991

VratedPhrmsHV = VratedLLrmsHV / np.sqrt(3)


#------Base value calculations----------

MVAbase = 100
MVAbaseLV = 2

ZbaseHV = (VratedLLrmsHV**2)/MVAbase
LbaseHV = ZbaseHV/Wrated
CbaseHV = 1/(Wrated*ZbaseHV)
Wbase = Wrated
