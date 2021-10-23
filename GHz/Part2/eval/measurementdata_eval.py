# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:12:20 2021

@author: ulm02
"""
from pathlib import Path
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from consts import ROOT_DIR

def func(phi, a, b, delta):
    phi = phi
    return np.abs(np.cos(phi) * np.sqrt((a * np.cos(phi) + b * np.sin(phi) * np.cos(delta)) ** 2
                                        + (b * np.sin(phi) * np.sin(delta)) ** 2))

data_folder = ROOT_DIR / Path('GHz') / Path('Part2') / Path('measurement_data_david')

plt.style.use('fast')

angles = np.arange(0, 370, 10)
polarizer_offset = -1

phi = np.array([])
s21 = np.array([])
s12 = np.array([])
idx = 0
for angle in angles:
    ntwk = rf.Network(data_folder / Path('Polarisator_%ddeg_time_gated_bp_c50ps_s500ps_d100ps.s2p' % (angle)))
    f = ntwk.f[idx]
    phi = np.append(phi, angle - 90 - polarizer_offset)
    s21 = np.append(s21, np.abs(ntwk.s[idx, 1, 0]))
    s12 = np.append(s12, np.abs(ntwk.s[idx, 0, 1]))

plt.figure()
phi = np.deg2rad(phi)
popt, pcov = curve_fit(func, phi, s21)
print(pcov[0,0], pcov[1,1], pcov[2,2])
a = popt[0]
b = popt[1]
delta = popt[2]
plt.polar(phi, s21, '.')
phi = np.linspace(0, 2 * np.pi, 1000)
plt.plot(phi, func(phi, a, b, delta))
plt.xlabel('$\phi$ in deg.')
plt.show()
plt.close()

f = np.array([])
delta = np.array([])
rel = np.array([])
eta = np.array([])
for idx in range(ntwk.f.size):
    if idx % 50 != 0:
        continue
    print(idx)

    phi = np.array([])
    s21 = np.array([])
    s12 = np.array([])
    ntwk = rf.Network(data_folder / Path('Polarisator_%ddeg_time_gated_bp_c50ps_s500ps_d100ps.s2p' % (angle)))
    f = np.append(f, ntwk.f[idx])
    for angle in angles:
        ntwk = rf.Network(data_folder / Path('Polarisator_%ddeg_time_gated_bp_c50ps_s500ps_d100ps.s2p' % (angle)))
        phi = np.append(phi, angle - 90 - polarizer_offset)
        s21 = np.append(s21, np.abs(ntwk.s[idx, 1, 0]))
        s12 = np.append(s12, np.abs(ntwk.s[idx, 0, 1]))

    phi = np.deg2rad(phi)
    popt, pcov = curve_fit(func, phi, s21)
    rel = np.append(rel, popt[0] / popt[1])
    eta = np.append(eta, (popt[0] ** 2 + popt[1] ** 2))
    delta = np.append(delta, np.abs(popt[2]))
plt.figure()
plt.plot(f / 10 ** 9, delta / np.pi, '.-', label='measurement')
plt.plot(f / 10 ** 9, f * 0 + 0.5 * 1.1, 'k--', label='0,5+10%')
plt.plot(f / 10 ** 9, f * 0 + 0.5 * 0.9, 'k--', label='0,5-10%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.xlim([75, 110])
plt.ylim([0.3, 0.6])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure()
plt.plot(f / 10 ** 9, rel, '.-', label='measurement')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{a}{b}$")
plt.xlim([75, 110])
plt.ylim([0, 2])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure()
plt.plot(f / 10 ** 9, eta, '.-', label='measurement')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\eta$")
plt.xlim([75, 110])
plt.ylim([0, 1])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()