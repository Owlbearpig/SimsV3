# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:12:20 2021

@author: ulm02
"""
from generate_plotdata import export_csv
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

um = 10**-6
THz = 10**12
GHz = 10**9

plt.style.use('fast')

plt.figure()

r = np.linspace(0.1, 6, 1000)

L = 1*10**-2
dn = 0.3
single_wp_delta = 2*pi*dn*r #% pi

s = ''
for i in range(1, 10):
    if round(i/(4*dn)) > 6:
        break
    s += f'({round(i/(4*dn), 3)}, 0.5) '

print(s)

s = ''
for i in range(1, 10):
    if round(i/(2*dn)) > 6:
        break
    s += f'({round(i/(2*dn), 3)}, 1) '

print(s)

new_r_array = np.array([])
new_delta_array = np.array([])
prev_val = single_wp_delta[0]
for i, (r_val, val) in enumerate(zip(r, single_wp_delta)):
    if np.abs(val-prev_val) > 1000:
        new_delta_array = np.append(new_delta_array, 'nan')
        new_r_array = np.append(new_r_array, (r[i]+r[i+1])/2)
    new_delta_array = np.append(new_delta_array, val)
    new_r_array = np.append(new_r_array, r_val)
    prev_val = val



export_csv({'x': new_r_array, 'delta': new_delta_array}, 'monochromaticwaveplates_wJumps.csv')
export_csv({'x': r, 'delta': single_wp_delta}, 'monochromaticwaveplates.csv')

plt.plot(r, single_wp_delta, label='Single wp')
plt.plot([(i+1)*pi/2 for i in range(10)], '.', label='i*pi/2')
#plt.gca().invert_xaxis()
plt.grid(True)
plt.xlabel(f'thickness/wavelength')
plt.ylabel(r'delta')
#plt.xlim([75, 110])
#plt.ylim([0.3, 0.6])
plt.legend()
plt.show()
