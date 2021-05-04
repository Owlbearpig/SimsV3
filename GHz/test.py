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
import skrf as rf
from results import *
from functions import setup
from consts import *

def func(phi, a, b, delta):
    phi = phi
    return np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=0*pi/180)

J_qwp = jones_matrix.create_Jones_matrices('QWP')
J_qwp.quarter_waveplate(azimuth=45*pi/180)

res = result_GHz
j, f, wls = setup(res, return_vals=True)
#pnt_cnt = len(f)//10
#j,f,wls = j[::len(f)//pnt_cnt], f[::len(f)//pnt_cnt], wls[::len(f)//pnt_cnt]
#j = np.random.random((1401, 2, 2)) + 1j*np.random.random((1401, 2, 2))

ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(100))
p2_100deg = np.abs(ntwk.s[:,0,1])
ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(190))
p1_190deg = np.abs(ntwk.s[:,0,1])

angles = np.arange(0,370,10)
for phi in angles:
    phi = phi #+ 0.5 + np.random.random()
    J_P = jones_matrix.create_Jones_matrices('P')
    J_P.diattenuator_linear(p1=1, p2=0, azimuth=phi*pi/180)

    amplitudes = []
    f_cut = np.array([])
    a_lst, b_lst, delta_lst = np.array([]), np.array([]), np.array([])
    for idx in range(len(f)):
        if idx%25 != 0:
            continue
        f_cut = np.append(f_cut, f[idx])
        print(idx)
        J = jones_matrix.create_Jones_matrices(res['name'])
        J.from_matrix(j[idx])
        J.rotate(angle=0*pi/180)
        p2 = 0# p2_100deg[idx]
        p1 = 1-p2#p1_190deg[idx]  # 0.5+0.5*(idx/len(f))
        J_A = jones_matrix.create_Jones_matrices('A')
        J_A.diattenuator_linear(p1=p1, p2=p2, azimuth=0*pi/180)

        #amplitudes = []
        J_out = J_A*J_P*J*Jin_l
        intensity = J_out.parameters.intensity()
        amplitudes.append(np.sqrt(intensity))

    plt.plot(f_cut / 10 ** 9, amplitudes, label=round(phi, 2))

plt.legend()
plt.show()

plt.figure()
f_cut = np.array([])
delta = np.array([])
a, b = np.array([]), np.array([])
for idx in range(len(f)):
    if idx%50 != 0:
        continue
    print(idx)
    #phi_offset = 4.725 + (14.84-4.725)*idx/1400

    f_cut = np.append(f_cut, f[idx])

    J = jones_matrix.create_Jones_matrices(res['name'])
    J.from_matrix(j[idx])
    J.rotate(angle=0 * pi / 180)
    p2 = 0  # p2_100deg[idx]
    p1 = 1 - p2  # p1_190deg[idx]  # 0.5+0.5*(idx/len(f))
    J_A = jones_matrix.create_Jones_matrices('A')
    J_A.diattenuator_linear(p1=p1, p2=p2, azimuth=0 * pi / 180)

    phi = np.array([])
    s21 = np.array([])
    for angle in angles:
        J_P = jones_matrix.create_Jones_matrices('P')
        J_P.diattenuator_linear(p1=1, p2=0, azimuth=angle * pi / 180)

        phi = np.append(phi, angle)
        J_out = J_A * J_P * J * Jin_l
        intensity = J_out.parameters.intensity()
        s21 = np.append(s21, np.sqrt(intensity))

    phi = np.deg2rad(phi)
    popt, pcov = curve_fit(func, phi, s21)

    print(*popt)
    #p1, p2, a, b, delta
    a = np.append(a, popt[0])
    b = np.append(b, popt[1])
    delta = np.append(delta, np.abs(popt[2]))
    continue
    plt.polar(phi, s21, label='messung')
    plt.plot(phi, func(phi, *popt), label='fit')
    plt.legend()
    plt.show()

print(result_GHz)
plt.plot(f_cut/10**9, delta/np.pi, '.-',label = 'Messung')
plt.plot(f_cut/10**9, f_cut*0+0.5*1.03, 'k--',label='+3%')
plt.plot(f_cut/10**9, f_cut*0+0.5*0.97, 'k--',label='-3%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.ylim([0, 1])
plt.xlim([75,110])
plt.show()