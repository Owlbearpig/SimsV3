import numpy as np
from numpy import exp, pi, sqrt, arccos, sin, cos
from scipy.constants import c as c0
from functions import material_values
from results import result_GHz
import matplotlib.pyplot as plt

um = 10**-6
GHz = 10**9

f = np.linspace(100, 1000, 1000)*GHz

wls = c0/f
n1, n2 = 1.54, 1
a, b = 200*um, 300*um
omega = 2*pi*f

def k(omega, beta):
    return sqrt((n1*omega/c0)**2-beta**2), sqrt((n2*omega/c0)**2-beta**2)

def PsiN_(N, A, D):
    KL = arccos(0.5*(A+D))
    return sin(N*KL)/sin(KL)

def A_(k1, k2):
    return exp(-1j*k1*a)*(cos(k2*b)-0.5*1j*(k2/k1+k1/k2)*sin(k2*b))

def D_(k1, k2):
    return exp(1j*k1*a)*(cos(k2*b)+0.5*1j*(k2/k1+k1/k2)*sin(k2*b))

beta_arr = 2*pi*np.linspace(0.0, 1.9, len(omega))/(a+b)

#beta_arr = 2*pi/wls

idx = 111
lhs = np.array([])
for beta in beta_arr:
    k1, k2 = k(omega[idx], beta)
    A, D = A_(k1, k2), D_(k1, k2)
    lhs = np.append(lhs, np.abs(0.5*(A+D)))

plt.plot(beta_arr, lhs)
plt.show()




"""
plt.plot(omega, beta_arr*c0/omega, label = r'$n_{eff}$')
plt.grid(True)
plt.xlabel('$\omega$ in Hz')
plt.ylabel(r'$n_{eff}$')
#plt.xlim([75,110])
#plt.ylim([0.0,3.5])
plt.legend()
plt.show()
"""

"""
N = 1
#idx = 500
roots = np.array([])
omega_cut = np.array([])
for idx in range(len(f)):
    if idx % 10 != 0:
        continue
    print(f'prog: {idx}/{len(f)}')
    omega_cut = np.append(omega_cut, omega[idx])

    lhs = np.array([])
    for beta in beta_arr:
        k1, k2 = k(omega[idx], beta)
        q, p = -1j*k1, k2
        A, D = A_(q, p), D_(q, p)
        lhs = np.append(lhs, A*PsiN_(N, A, D)-PsiN_(N-1, A, D))

    plt.plot(beta_arr, lhs)
    plt.show()
    roots = np.append(roots, np.argmin(np.abs(lhs)))




plt.plot(omega_cut, roots)
plt.grid(True)
plt.xlabel(r'$\omega$ in Hz')
plt.ylabel(r'roots')
#plt.xlim([75,110])
#plt.ylim([0.0,3.5])
plt.legend()
plt.show()
"""
"""
plt.plot(beta_arr, lhs, label = r'$LHS(\beta)$')
plt.grid(True)
plt.xlabel(r'$\beta$ in 1/m')
plt.ylabel(r'$LHS(\beta)$')
#plt.xlim([75,110])
#plt.ylim([0.0,3.5])
plt.legend()
plt.show()
"""