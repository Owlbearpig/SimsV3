# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:12:20 2021

@author: ulm02
"""
from generate_plotdata import export_csv
import skrf as rf
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(phi, a, b, delta):
    phi = phi
    return np.abs(np.cos(phi) * np.sqrt((a * np.cos(phi) + b * np.sin(phi) * np.cos(delta)) ** 2
                                        + (b * np.sin(phi) * np.sin(delta)) ** 2))


plt.style.use('fast')

ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p' % (10))

angles = np.arange(0, 370, 10)
offsets = np.linspace(0, 15, 30)
sigmas = np.zeros((len(ntwk.f), len(offsets), 3))
"""
for idx in range(len(ntwk.f)):
    print(idx)
    if idx % 50:
        continue
    for i, offset in enumerate(offsets):
        phi = np.array([])
        s21 = np.array([])
        s12 = np.array([])
        for angle in angles:
            ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p' % (angle))
            f = ntwk.f[idx]
            phi = np.append(phi, angle - offset)
            s21 = np.append(s21, np.abs(ntwk.s[idx, 1, 0]))
            s12 = np.append(s12, np.abs(ntwk.s[idx, 0, 1]))


        phi = np.deg2rad(phi)
        
        popt, pcov = curve_fit(func, phi, s21)

        sigmas[idx, i] = np.sqrt(np.diag(pcov))
"""

sigmas = np.load('sigmas.npy')

plt.figure()
#plt.title(str(idx))
#plt.plot(offsets, sigmas[:,0], label='sigma a')
#plt.plot(offsets, sigmas[:,1], label='sigma b')
for i in range(28):
    plt.plot(offsets, sigmas[i*50, :, 2], label=f'sigma delta {i}')
plt.legend()
plt.show()

# 6 deg seems to be best fit...
