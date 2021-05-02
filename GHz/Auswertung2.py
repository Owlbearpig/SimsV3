# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:12:20 2021

@author: ulm02
"""

import skrf as rf
import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

f = np.load('f.npy')
a = np.load('a.npy')
b = np.load('b.npy')
phi = np.load('phi.npy')
delta = np.load('delta.npy')

for idx in range(0, len(f)):
    if round(f[idx]/10**9, 2) % 5 != 0:
        continue

    Ex, Ey = np.array([]), np.array([])
    for phase in np.linspace(0, 1, 360)*2*pi:
        x, y = a[idx]*cos(phase), b[idx]*cos(phase+delta[idx])
        Ex, Ey = np.append(Ex, x), np.append(Ey, y)

    plt.plot(Ex, Ey, label=f[idx]/10**9)

plt.ylim((-1.1, 1.1))
plt.xlim((-1.1, 1.1))
plt.legend()
plt.show()

plt.plot(f / 10 ** 9, b / a, label='Messung')
# plt.plot(f/10**9, delta/np.pi, '.-',label = 'Messung')
# plt.plot(f/10**9, f*0+0.5*1.03, 'k--',label='+3%')
# plt.plot(f/10**9, f*0+0.5*0.97, 'k--',label='-3%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
# plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.xlim([75, 110])
plt.ylim([0, 1])
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.savefig('Retardation.pdf', bbox_inches='tight')
plt.show()
