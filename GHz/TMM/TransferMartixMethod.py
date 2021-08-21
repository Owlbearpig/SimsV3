import numpy as np
from numpy import exp, pi, sqrt, arccos, sin, cos
from scipy.constants import c as c0
from functions import material_values
from results import result_GHz
import matplotlib.pyplot as plt

um = 10**-6
GHz = 10**9

"""
d1, d2 = result_GHz['x'][-2], result_GHz['x'][-1]
eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
eps_mat1, eps_mat2, f, wls = eps_mat1.flatten(), eps_mat2.flatten(), f.flatten(), wls.flatten()
n1_measured, n2_measured = sqrt(abs(eps_mat1) + eps_mat1.real) / sqrt(2), sqrt(abs(eps_mat2) + eps_mat2.real) / sqrt(2)
"""

f = np.linspace(100, 1000, 1000)*GHz
n1, n2 = 1.54, 1
d1,d2 = 200*um, 300*um
omega = 2*pi*f

def k(omega, beta):
    return sqrt((n1*omega/c0)**2-beta**2), sqrt((n2*omega/c0)**2-beta**2)


def D(k):
    return np.array([[1+1j*k, 1-1j*k],
                     [1-1j*k, 1+1j*k]])

def Dinv(k):
    return np.array([[k-1j, 1-1j*k],
                     [1-1j*k, 1+1j*k]])

def P(k, d):
    return np.array([[exp(-1j*k*d), 0],
                     [0, exp(1j*k*d)]])

def M0(omega, beta):
    k1, k2 = k(omega, beta)

    T1 = np.dot(D(k2), np.dot(P(k2, d2), Dinv(k2)))
    T2 = np.dot(D(k2), np.dot(P(k2, d2), Dinv(k2)))
    M0 = np.dot(T1, T2)
    return M0

def K(M):
    M11, M22 = M[0,0], M[1,1]
    return 1/(d1+d2) * arccos(0.5*(M11+M22))

def PsiN(N, M):
    KL = (d1+d2)*K(M)
    return sin(N*KL)/sin(KL)


beta_arr = np.linspace(0, 1, len(omega))*(d1+d2)

plt.plot(omega, beta_arr*omega/c0, label = r'$n_{eff}$')
plt.grid(True)
plt.xlabel('$\omega$ in GHz')
plt.ylabel(r'$n_{eff}$')
#plt.xlim([75,110])
#plt.ylim([0.0,3.5])
plt.legend()
plt.show()

N = 1
idx = 700

rhs = np.array([])
for beta in beta_arr:
    M = M0(omega[idx], beta)
    rhs = np.append(rhs, M[0, 0]*PsiN(N, M)-PsiN(N-1, M))

plt.plot(beta_arr, rhs, label = r'$RHS(\beta)$')
plt.grid(True)
plt.xlabel(r'$\beta$ in m')
plt.ylabel(r'$RHS(\beta)$')
#plt.xlim([75,110])
#plt.ylim([0.0,3.5])
plt.legend()
plt.show()
