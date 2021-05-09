import skrf as rf
import numpy as np
from scipy.optimize import curve_fit

def func(phi, a, b, delta):
    phi = phi
    return np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)


def delta_measured_eval(phi_offset):
    angles = np.arange(0,370,10)
    ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(100))

    f = np.array([])
    delta = np.array([])
    a, b = np.array([]), np.array([])
    for idx in range(ntwk.f.size):
        if idx%50 != 0:
            pass
        print(idx)
        #phi_offset = 4.725 + (14.84-4.725)*idx/1400
        phi = np.array([])
        s21 = np.array([])
        s12 = np.array([])
        ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(10))
        f = np.append(f, ntwk.f[idx])

        for angle in angles:
            ntwk = rf.Network('%d deg_time_gated_bp_c0ps_s100ps_d20ps.s2p'%(angle))
            phi = np.append(phi, angle-phi_offset)
            s21 = np.append(s21, np.abs(ntwk.s[idx,1,0]))
            s12 = np.append(s12, np.abs(ntwk.s[idx,0,1]))

        phi = np.deg2rad(phi)
        popt, pcov = curve_fit(func, phi, s21)

        print(*popt)
        #p1, p2, a, b, delta
        a = np.append(a, popt[0])
        b = np.append(b, popt[1])
        delta = np.append(delta, np.abs(popt[2]))

    return delta
