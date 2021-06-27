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
from scipy.constants import c as c0
from scipy.optimize import curve_fit


def func(phi, a, b, delta):
    phi = phi
    return np.abs(np.cos(phi) * np.sqrt((a * np.cos(phi) + b * np.sin(phi) * np.cos(delta)) ** 2
                                        + (b * np.sin(phi) * np.sin(delta)) ** 2))

plt.style.use('fast')

angles = np.arange(0, 370, 10)
offset_angle = 6.00

phi = np.array([])
s21 = np.array([])
s12 = np.array([])
idx = 1400
for angle in angles:
    ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p' % (angle))
    f = ntwk.f[idx]
    phi = np.append(phi, angle - offset_angle)
    s21 = np.append(s21, np.abs(ntwk.s[idx, 1, 0]))
    s12 = np.append(s12, np.abs(ntwk.s[idx, 0, 1]))

plt.figure()
phi = np.deg2rad(phi)
popt, pcov = curve_fit(func, phi, s21)
a = popt[0]
b = popt[1]
delta = popt[2]
plt.polar(phi, s21, '.')
phi = np.linspace(0, 2 * np.pi, 1000)
plt.plot(phi, func(phi, a, b, delta))
plt.xlabel('$\phi$ in deg.')
#plt.savefig(f'Polarplot_{f / 10 ** 9}GHz.pdf')
plt.close()
plt.figure()

f = np.array([])
delta, a, b = np.array([]), np.array([]), np.array([])
for idx in range(ntwk.f.size):
    if idx % 50 != 0:
        continue
    print(idx)

    phi = np.array([])
    s21 = np.array([])
    s12 = np.array([])
    ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p' % (angle))
    f = np.append(f, ntwk.f[idx])
    for angle in angles:
        ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p' % (angle))
        phi = np.append(phi, angle - offset_angle)
        s21 = np.append(s21, np.abs(ntwk.s[idx, 1, 0]))
        s12 = np.append(s12, np.abs(ntwk.s[idx, 0, 1]))

    phi = np.deg2rad(phi)
    popt, pcov = curve_fit(func, phi, s21)
    delta = np.append(delta, np.abs(popt[2]))
    a = np.append(a, popt[0])
    b = np.append(b, popt[1])


dn, L = 0.08, 1.0125*10**-2
single_wp_delta = 2*pi*dn*L*f/c0


#export_csv({'freqs': f/10**9, 'a': a, 'b': b, 'delta': delta, 'delta1wp': single_wp_delta}, 'measurement_result.csv')

plt.plot()
plt.plot(f / 10 ** 9, delta / pi, '.-', label='Messung')
plt.plot(f / 10 ** 9, single_wp_delta / pi, '.-', label='Single wp')
plt.plot(f / 10 ** 9, f * 0 + 0.5 * 1.03, 'k--', label='+3%')
plt.plot(f / 10 ** 9, f * 0 + 0.5 * 0.97, 'k--', label='-3%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.xlim([75, 110])
#plt.ylim([0.3, 0.6])
plt.legend()
plt.show()

plt.plot(f / 10 ** 9, a, '.-', label='a')
plt.plot(f / 10 ** 9, b, '.-', label='b')
plt.grid(True)
plt.xlabel('$f$ in GHz')
# plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.xlim([75, 110])
# plt.ylim([0.3,0.6])
plt.legend()
plt.show()

# plt.savefig('Retardation.pdf', bbox_inches='tight')