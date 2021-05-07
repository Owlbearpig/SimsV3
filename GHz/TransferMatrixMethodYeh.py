import numpy as np
from numpy import exp, pi, arccos, sin, cos, sqrt
from scipy.constants import c as c0
from functions import material_values
from results import result_GHz, x_ghz
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt as csqrt

result_HIPS_HHI = {
        'name': 'c_random',
        'comments': '',
        'x': '',
        'bf': 'form',
        'mat_name': ('HIPS_HHI', '')
}

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

def D_TE(k1, k2):
    return exp(1j*k1*a)*(cos(k2*b)+0.5*1j*(k2/k1+k1/k2)*sin(k2*b))

def A_TM(k1, k2):
    return exp(-1j*k1*a)*(cos(k2*b)-0.5j*((n2**2*k1)/(n1**2*k2)+(n1**2*k2)/(n2**2*k1))*sin(k2*b))

def D_TM(k1, k2):
    return exp(1j*k1*a)*(cos(k2*b)+0.5j*((n2**2*k1)/(n1**2*k2)+(n1**2*k2)/(n2**2*k1))*sin(k2*b))

beta_arr = 2*pi/wls
beta_arr = np.linspace(0, 4, len(omega))*1000

f_cut = np.array([])
res_min_te, res_max_te = [], []
res_min_tm, res_max_tm = [], []
for idx in range(len(omega)):
    if idx % 10 != 0:
        pass
    print(idx, f[idx]/10**9)
    f_cut = np.append(f_cut, f[idx])

    n1, n2 = sqrt(abs(eps_mat1[idx]) + eps_mat1[idx].real) / sqrt(2), \
             sqrt(abs(eps_mat2[idx]) + eps_mat2[idx].real) / sqrt(2)

    lhs_te, lhs_tm = np.array([]), np.array([])
    for beta in beta_arr:
        k1, k2 = k(omega[idx], beta)
        if isinstance(k1, complex) or isinstance(k2, complex):
            pass
            #print(beta)
        Ate, Dte = A_TE(k1, k2), D_TE(k1, k2)
        lhs_te = np.append(lhs_te, np.abs(0.5 * (Ate + Dte)) - 1)

        Atm, Dtm = A_TM(k1, k2), D_TM(k1, k2)
        lhs_tm = np.append(lhs_tm, np.abs(0.5 * (Atm + Dtm)) - 1)

    zeros_te, zeros_tm = np.array([], dtype=int), np.array([], dtype=int)
    prev_pnt_te, prev_pnt_tm = lhs_te[0], lhs_tm[0]
    for i, (pnt_te, pnt_tm) in enumerate(zip(lhs_te, lhs_tm)):
        if prev_pnt_te*pnt_te < 0:
            zeros_te = np.append(zeros_te, i)
        prev_pnt_te = pnt_te

        if prev_pnt_tm*pnt_tm < 0:
            zeros_tm = np.append(zeros_tm, i)
        prev_pnt_tm = pnt_tm

    print('zeros_te', zeros_te)
    res_te = beta_arr[zeros_te]*c0/omega[idx]
    try:
        res_min_te.append(res_te[-2])
    except IndexError:
        res_min_te.append(1)
    res_max_te.append(res_te[-1])

    print('zeros_tm', zeros_tm)
    res_tm = beta_arr[zeros_tm] * c0 / omega[idx]
    try:
        res_min_tm.append(res_tm[-2])
    except IndexError:
        res_min_tm.append(1)
    res_max_tm.append(res_tm[-1])

np.save('HIPS_HHI_yeh_te.npy', res_max_te)
np.save('HIPS_HHI_yeh_tm.npy', res_max_tm)

#plt.plot(f_cut, ref_min)
plt.plot(f_cut, res_max_te, label='te')
plt.plot(f_cut, res_max_tm, label='tm')
#plt.ylim((0, 1.5))
plt.legend()
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