# DP_models
Dynamic Phasor Models for CPU  and GPU Simulations

# Simulation Case Files #

The Folders contain the python simulation cases.The ___Exec.py___ will execute the case and generate the responses along with the interface to generate eigenvalues of the system. 
1. For the Model that is simulated in CPU, the files are stored in  __(118_CONV_CPU)__
2. For the Model that is simulated in CPU, the files are stored in  __(118_CONV_GPU)__

The parameters can be changed or viewed by changing the Excel files that are included in the __(Parameters)__ folders
The real-time plotting tools use pyQTgraph plotting libraries. For optimal performance gains the plot step should be increased, or the plots may be disabled.

The relevant dependencies to be installed are in the heading of each file and are listed below. They can be installed through __pip__ or __anaconda__ packages. 

* numpy  {1.20.3 (defaults/win-64) 
* numba  {0.54.1 (defaults/win-64)
* scipy  {1.7.1  (defaults/win-64)} 
* math
* Threading
* time
* pyqtgraph  {0.11.0 (anaconda/noarch) 
* pysimplegui {4.56.0 (conda-forge/noarch)}
* pandas  {1.3.4 (defaults/win-64)}
* sympy  {1.9 (defaults/win-64)}
* plotly {5.5.0 (plotly/noarch)}


