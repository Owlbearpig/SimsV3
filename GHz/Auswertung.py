# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:12:20 2021

@author: ulm02
"""

import skrf as rf
import numpy as np
from numpy import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func1(phi, a, b, delta):
    phi = phi
    A = np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)
    return A

def func2(phi, p1, p2, a, b, delta):
    phi = phi
    I0 = (a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2+(b*np.sin(phi)*np.sin(delta))**2
    return sqrt(p1**2*cos(phi)**2+p2**2*sin(phi)**2)*sqrt(I0)

def func21(phi, p1):
    phi = phi
    a,b,delta = 1,1, pi/2
    p1 = p1
    p2 = 1-p1
    I0 = (a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2+(b*np.sin(phi)*np.sin(delta))**2
    return sqrt(p1**2*cos(phi)**2+p2**2*sin(phi)**2)*sqrt(I0)

def func22(phi, a, b, delta):
    print(p1,p2)
    phi = phi
    I0 = (a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2+(b*np.sin(phi)*np.sin(delta))**2
    return sqrt(p1**2*cos(phi)**2+p2**2*sin(phi)**2)*sqrt(I0)

def func3(freq, ampl, omega, shift, shift2):
    freq = freq
    shift = 13.558989995949837
    omega = 0.08900501265729782
    return ampl*sin(omega*freq+shift)+shift2


plt.style.use('fast')

angles = np.arange(0,370,10)

ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(190))
normalization = np.abs(ntwk.s[:,1,0])

for angle in angles:
    if angle > 120:
        continue
    if angle < 70:
        continue

    ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(angle))
    plt.plot(ntwk.f/10**9, np.abs(ntwk.s[:,1,0]), label=str(angle))

plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"amplitude")
plt.xlim([75,110])
#plt.ylim([0.0,1.1])
plt.legend()
plt.show()

ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(10))
normalization = np.abs(ntwk.s[:,1,0])
print(normalization)
phi = np.array([])
s21 = np.array([])
s12 = np.array([])
phi_offset = 9.64#9.64#14.84#10#9.64 # 4.5
idx = 1400
for angle in angles:

    ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(angle))
    f = ntwk.f[idx]
    phi = np.append(phi, angle-phi_offset)
    s21 = np.append(s21, (np.abs(ntwk.s[idx,1,0])))
    s12 = np.append(s12, np.abs(ntwk.s[idx,0,1]))

#ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(190))
#plt.plot(ntwk.f/10**9, np.abs(ntwk.s[:,1,0]))
#plt.show()


ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(100))
f = ntwk.f
popt, pcov = curve_fit(func3, f/10**9, np.abs(ntwk.s[:,0,1]), p0=[0.01951316817797292, 0.08900501265729782,
                                                                  13.558989995949837, 0.0585470010609849]) # 0.05*sin(0.2*f/10**9 + pi)+pi/2

print(*popt)

plt.plot(f/10**9, func3(f/10**9, *popt), label='fit')
plt.plot(f/10**9, np.abs(ntwk.s[:,0,1]), '.-',label = '100 deg')
plt.legend()
plt.show()


plt.figure()
phi = np.deg2rad(phi)
popt, pcov = curve_fit(func1, phi, s21)
a = popt[0]
b = popt[1]
delta = popt[2]
print('p1, p2, a, b, delta')
print(popt)
plt.polar(phi, s21,'.')
phi = np.linspace(0,2*np.pi,1000)
plt.plot(phi, func1(phi, *popt))
plt.xlabel('$\phi$ in deg.')
#plt.savefig(f'Polarplot_{f/10**9}GHz.pdf')
plt.show()
plt.close()

#exit()

plt.figure()
f = np.array([])
delta = np.array([])
a, b = np.array([]), np.array([])
p1_arr, p2_arr = np.array([]), np.array([])
for idx in range(ntwk.f.size):
    if idx%1 != 0:
        continue
    print(idx)
    #phi_offset = 4.725 + (14.84-4.725)*idx/1400
    phi = np.array([])
    s21 = np.array([])
    s12 = np.array([])
    ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(angle))
    f = np.append(f, ntwk.f[idx])

    for angle in angles:
        ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(angle))
        phi = np.append(phi, angle-phi_offset)
        s21 = np.append(s21, (np.abs(ntwk.s[idx,1,0])))
        s12 = np.append(s12, np.abs(ntwk.s[idx,0,1]))

    phi = np.deg2rad(phi)
    popt, pcov = curve_fit(func21, phi, s21, p0=[1])

    #p1, p2 = popt[0], popt[1]
    p1 = 0#popt[0]
    p2 = 1-p1
    p1_arr = np.append(p1_arr, popt[0])
    p2_arr = np.append(p2_arr, p2)

    popt, _ = curve_fit(func1, phi, s21)

    print(*popt)
    #p1, p2, a, b, delta
    a = np.append(a, popt[0])
    b = np.append(b, popt[1])
    delta = np.append(delta, np.abs(popt[2]))

#np.save('delta_phi0.npy', delta)

plt.plot(f, p1_arr, label='p1')
plt.plot(f, p2_arr, label='p2')
plt.legend()
plt.show()

plt.plot(f/10**9, a/b, label='a/b')
plt.legend()
plt.show()

from generate_plotdata import export_csv
export_csv({'freq': f, 'delta': delta}, 'delta_measured')

plt.plot(f/10**9, delta/np.pi, '.-',label = 'Messung')
plt.plot(f/10**9, f*0+0.5*1.03, 'k--',label='+3%')
plt.plot(f/10**9, f*0+0.5*0.97, 'k--',label='-3%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.xlim([75,110])
plt.ylim([0.0,1.0])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.savefig('Retardation.pdf', bbox_inches='tight')
plt.show()

exit()
np.save('f', f)
np.save('a', a)
np.save('b', b)
np.save('phi', phi)
np.save('delta', delta)


#delta = np.load('delta.npy')


popt, pcov = curve_fit(func3, f/10**9, delta, p0=[0.05, 0.2, pi, pi/2]) # 0.05*sin(0.2*f/10**9 + pi)+pi/2
print(pcov)
print(*popt)

#plt.plot(f/10**9, func3(f/10**9, 0.05, 0.2, pi, pi/2), label='me')
plt.plot(f/10**9, func3(f/10**9, *popt), label='fit')
plt.plot(f/10**9, delta, '.-',label = 'Messung')
plt.legend()
plt.show()

plt.plot(f/10**9, (func3(f/10**9, *popt)-delta+pi/2)/pi)
plt.plot(f/10**9, f*0+0.5*1.03, 'k--',label='+3%')
plt.plot(f/10**9, f*0+0.5*0.97, 'k--',label='-3%')
plt.xlim([75,110])
plt.ylim([0.0,1.0])
plt.show()

#plt.plot(f/10**9, a/b, label = 'a/b')
#plt.grid(True)
#plt.show()
plt.plot(f/10**9, delta/np.pi, '.-',label = 'Messung')
plt.plot(f/10**9, f*0+0.5*1.03, 'k--',label='+3%')
plt.plot(f/10**9, f*0+0.5*0.97, 'k--',label='-3%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.xlim([75,110])
plt.ylim([0.0,1.0])
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.savefig('Retardation.pdf', bbox_inches='tight')
plt.show()
