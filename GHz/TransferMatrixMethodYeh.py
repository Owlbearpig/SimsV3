import numpy as np
from numpy import exp, pi, arccos, sin, cos, sqrt
from scipy.constants import c as c0
from functions import material_values
from results import result_GHz
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt as csqrt

um = 10**-6
GHz = 10**9

eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
#f = np.linspace(100, 1000, 1000)*GHz
f, wls, eps_mat1, eps_mat2 = f.flatten(), wls.flatten()*10**-6, eps_mat1.flatten(), eps_mat2.flatten()

#wls = c0/(np.linspace(100, 1000, 1000)*GHz)

n1, n2 = 1.54, 1 # overwritten in loop
#a, b = 200*um, 300*um
a, b = 628*um, 517*um
omega = 2*pi*f


def k(omega, beta):
    return csqrt((n1*omega/c0)**2-beta**2), csqrt((n2*omega/c0)**2-beta**2)

def PsiN_(N, A, D):
    KL = arccos(0.5*(A+D))
    return sin(N*KL)/sin(KL)

def A_TE(k1, k2):
    return exp(-1j*k1*a)*(cos(k2*b)-0.5*1j*(k2/k1+k1/k2)*sin(k2*b))

def D_TM(k1, k2):
    return exp(1j*k1*a)*(cos(k2*b)+0.5*1j*(k2/k1+k1/k2)*sin(k2*b))

def A_TM(k1, k2):
    return exp()

beta_arr = 2*pi/wls
beta_arr = np.linspace(0, 4, 1000)*1000

ref_min, ref_max = [],[]
for idx in range(len(omega)):
    #idx = 335
    n1, n2 = sqrt(abs(eps_mat1[idx]) + eps_mat1[idx].real) / sqrt(2), \
             sqrt(abs(eps_mat2[idx]) + eps_mat2[idx].real) / sqrt(2)
    print(f[idx])
    lhs = np.array([])
    for beta in beta_arr:
        k1, k2 = k(omega[idx], beta)
        if isinstance(k1, complex) or isinstance(k2, complex):
            pass
            #print(beta)
        Ate, Dte = A_TE(k1, k2), D_TM(k1, k2)
        lhs = np.append(lhs, np.abs(0.5 * (Ate + Dte).real) - 1)

    #plt.plot(beta_arr, lhs)
    #plt.show()

    zeros = np.array([], dtype=int)
    prev_pnt = lhs[0]
    for i, pnt in enumerate(lhs):
        if prev_pnt*pnt < 0:
            zeros = np.append(zeros, int(i))
        prev_pnt = pnt

    print(zeros)
    res = beta_arr[zeros]*c0/omega[idx]
    print(res)
    try:
        ref_min.append(res[-2])
    except IndexError:
        ref_min.append(1)
    ref_max.append(res[-1])

plt.plot(f, ref_min)
plt.plot(f, ref_max)
plt.ylim((1, 1.5))
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

plt.plot(beta_arr, lhs, label = r'$LHS(\beta)$')
plt.grid(True)
plt.xlabel(r'$\beta$ in 1/m')
plt.ylabel(r'$LHS(\beta)$')
#plt.xlim([75,110])
#plt.ylim([0.0,3.5])
plt.legend()
plt.show()
"""