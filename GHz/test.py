import numpy as np
from numpy import (power, outer, sqrt, exp, sin, cos, conj, dot, pi,
                   einsum, arctan, array, arccos, conjugate, flip, angle, tan, arctan2)
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from py_pol import jones_matrix
from py_pol import jones_vector
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
import string
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
from results import *
from functions import setup
from consts import *

def func(phi, a, b, delta):
    phi = phi
    return np.abs(np.cos(phi)*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2))

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light()

J_qwp = jones_matrix.create_Jones_matrices('QWP')
J_qwp.quarter_waveplate(azimuth=45*pi/180)

res = result_GHz
j, f, wls = setup(res, return_vals=True)
pnt_cnt = len(f)//3
j,f,wls = j[::len(f)//pnt_cnt], f[::len(f)//pnt_cnt], wls[::len(f)//pnt_cnt]

a_lst,b_lst,delta_lst = [],[],[]
for idx in range(len(f)):
    print(idx)
    J = jones_matrix.create_Jones_matrices(res['name'])
    J.from_matrix(j[idx])

    J_A = jones_matrix.create_Jones_matrices('A')
    J_A.diattenuator_perfect(azimuth=0*pi/180)

    intensities = []
    angles = np.linspace(0,2*pi,36)
    for phi in angles:
        J_P = jones_matrix.create_Jones_matrices('P')
        J_P.diattenuator_perfect(azimuth=phi)

        J_out = J_A*J_P*J*Jin_l
        intensity = J_out.parameters.intensity()
        intensities.append(np.sqrt(intensity))

    popt, pcov = curve_fit(func, angles, intensities)
    a = popt[0]
    b = popt[1]
    delta = popt[2]
    a_lst.append(a)
    b_lst.append(b)
    delta_lst.append(delta)
    print(a, b, delta)
    plt.polar(angles, intensities, '.')
    plt.plot(angles, func(angles, a, b, delta))
    plt.show()
#plt.plot(f, a_lst)
#plt.plot(f, b_lst)
plt.plot(f, delta_lst)
plt.show()