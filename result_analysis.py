import numpy as np
from numpy import (power, outer, sqrt, exp, sin, cos, conj, dot, pi,
                   einsum, arctan, array, arccos, conjugate, flip, angle, tan, arctan2)
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from py_pol import jones_matrix
from py_pol import jones_vector
from py_pol import stokes
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
import string
from plotting import poincare_path, draw_ellipse
import matplotlib.pyplot as plt
import sys
from consts import *
from functions import setup, thickness_for_1thz
from results import *
# plotting single values and stuff

if __name__ == '__main__':

    res_test = {
        'name': 'test',
        'comments': '',
        'x': np.concatenate(([45*pi/180], [1.525*10**3])),
        'bf': 'intrinsic',
        'mat_name': ('ceramic_fast', 'ceramic_slow')
    }

    res = result_masson_full
    #res = result_masson
    res = result1
    res = result_GHz

    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light()

    #j_individual, _, _ = setup(res, return_vals=True, return_individual=True)
    #poincare_path(Jin_l, j_individual)

    j, f, wls = setup(res, return_vals=True)

    #j,f,wls = j[::len(f)//20], f[::len(f)//20], wls[::len(f)//20]
    T = jones_matrix.create_Jones_matrices(res['name'])
    T.from_matrix(j)

    print(np.round(f[700]*10**-9, 1), T[700])

    S = stokes.create_Stokes()
    S.from_Jones(T[700]*Jin_l)

    print(S)

    Jin_c = jones_vector.create_Jones_vectors('RCP')
    Jin_c.circular_light(kind='r')

    diattenuation = T.parameters.diattenuation()
    retardance = T.parameters.retardance()
    inhomogeneity = T.parameters.inhomogeneity()

    plt.plot(f, diattenuation, label='diattenuation')
    plt.legend()
    plt.show()
    plt.plot(f, retardance, label='retardance')
    plt.legend()
    plt.show()
    plt.plot(f, inhomogeneity, label='inhomogeneity')
    plt.legend()
    plt.show()

    #J.rotate(angle=0.1)
    #slice = np.where(f < 1.6*THz)[0]
    #from dataexport import save, pe_export
    #f = (np.arange(0.2, 2.0, 0.05)*THz)[:]
    #print(np.argmin(np.abs(f-0.861*THz)))
    m = len(wls)
    #thickness_for_1thz(res)
    #print(len(f))
    #exit()

    #int_x = j[:, 0, 0]*np.conjugate(j[:, 0, 0])
    #int_y = j[:, 1, 0]*np.conjugate(j[:, 1, 0])
    q = j[:, 1, 0] / j[:, 0, 0]
    L_state = q.real ** 2 + (q.imag - 1) ** 2
    print('argmin:', np.argmin(L_state), 'f(argmin):', f[np.argmin(L_state)]*(1/THz))
    print('min:', min(L_state))
    #L = L / max(L)
    #print(np.mean(L), np.std(L))
    A, B = j[:, 0, 0], j[:, 0, 1]
    #print(j)
    delta_equiv = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))
    L_ret = (delta_equiv-pi/2)**2
    #print(L_ret)
    #print(L_state)
    from generate_plotdata import export_csv
    plt.semilogy(f, L_state, label='state obj. func')
    #export_csv({'freq': f.flatten(), 'L': L}, 'plot_data/masson/MassLoss.csv')
    #plt.semilogy(f, L_ret, label='ret')
    #plt.ylim((-2.5*10**-4, 2.5*10**-3))
    plt.legend()
    plt.show()

    #int_x, int_y = 10*np.log10(int_x.real), 10*np.log10(int_y.real)

    #J.remove_global_phase()
    #J.set_global_phase(0)
    #J.analysis.retarder(verbose=True)
    #print(j[:,0,0])
    #print(j[:, 0, 1])
    #print(j[:, 1, 0])
    #print(j[:, 1, 1])
    J_lowres = jones_matrix.create_Jones_matrices('J_lr')
    print(f[::30])
    J_lowres.from_matrix(j[::30])

    J_o_lowres = J_lowres*Jin_l
    J_o_lowres.draw_ellipse()
    plt.show()

    v1, v2, E1, E2 = T.parameters.eig(as_objects=True)

    plt.plot(E1.parameters.azimuth(), label='E1 azimuth')
    plt.plot(E2.parameters.azimuth(), label='E2 azimuth')
    plt.legend()
    plt.show()

    plt.plot(90 - (E1.parameters.azimuth() - E2.parameters.azimuth())*rad, label='90 - E1 azimuth - E2 azimuth')
    plt.legend()
    plt.show()

    plt.plot(E1.parameters.eccentricity(), label='E1 eccentricity')
    plt.plot(E2.parameters.eccentricity(), label='E2 eccentricity')
    plt.legend()
    plt.show()

    v1_lr, v2_lr, E1_lr, E2_lr = J_lowres.parameters.eig(as_objects=True)

    E1_lr.draw_ellipse()
    E2_lr.draw_ellipse()
    plt.show()
    #J.parameters.global_phase(verbose=True)
    #Jqwp = jones_matrix.create_Jones_matrices()
    #Jqwp.retarder_linear()

    T = jones_matrix.create_Jones_matrices()
    T.from_matrix(j)

    #print(J.analysis.retarder(verbose=True))
    #E1[11].draw_ellipse()
    #E2[11].draw_ellipse()
    #plt.show()
    #E1.draw_ellipse()
    #E2.draw_ellipse()

    #print(360-np.abs(np.angle(v1[0])-np.angle(v2[0]))*180/pi)
    #E1.parameters.global_phase(verbose=True)
    #(J*E1).parameters.global_phase(verbose=True)
    #E2.remove_global_phase()
    #(J * E2).parameters.global_phase(verbose=True)
    #J.parameters.retardance(verbose=True)
    #print(np.angle(j)[:,0,0]-np.angle(j)[:,0,1])
    #v1, v2, E1, E2 = Jqwp.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()

    #Jhi.remove_global_phase()
    #v1, v2, E1, E2 = Jhi.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()
    #plt.show()
    #print(Jhi.parameters)
    #J.remove_global_phase()
    #print(J[11].parameters)
    #J_qwp = jones_matrix.create_Jones_matrices('J_qwp')
    #J_qwp.quarter_waveplate(azimuth=pi/4)

    #Jin_c.draw_ellipse()
    #plt.show()


    #Jin_l.draw_ellipse()

    #J_ideal_out = J_qwp*J_qwp*Jin_c

    #J_ideal_out.draw_ellipse()
    #plt.show()

    #Jin_c.draw_ellipse()
    #plt.show()
    #Jin.draw_ellipse()
    #Jout_l = J * Jin_l
    #Jout_l.draw_ellipse()
    #plt.show()
    Jout = T * Jin_l
    #Jout_l.normalize()

    Ex, Ey = draw_ellipse(Jout, return_values=True)
    print(len(Ex))

    circ_pol_deg = Jout.parameters.degree_circular_polarization()
    lin_pol_deg = Jout.parameters.degree_linear_polarization()

    plt.plot(f, circ_pol_deg, label='circ. pol. degree')
    plt.plot(f, lin_pol_deg, label='lin. pol. degree')
    plt.legend()
    plt.show()

    a, b = Jout.parameters.ellipse_axes()
    plt.plot(f, a, label='a')
    plt.plot(f, b, label='b')
    plt.legend()
    plt.show()

    plt.plot(f, b/a, label='b/a')
    plt.legend()
    plt.show()

    alpha = Jout.parameters.alpha()
    delay = Jout.parameters.delay()
    #plt.plot(f, delay-pi, label='delay')
    """
    plt.plot(f, delta_equiv/pi, label='delta equiv')
    plt.xlim((0, 1.75*THz))
    plt.ylim((0.2, 0.6))
    plt.legend()
    plt.show()
    """
    azimuth = Jout.parameters.azimuth()
    ellipticity_angle = Jout.parameters.ellipticity_angle()

    eccentricity = Jout.parameters.eccentricity()

    intensity = Jout.parameters.intensity()

    plt.plot(f, -10*np.log10(intensity), label='intensity')
    #plt.plot(f, intensity, label='intensity')
    plt.legend()
    plt.show()

    plt.plot(f, alpha * rad, label='alpha')
    plt.legend()
    plt.show()

    plt.plot(f, alpha*rad, label='alpha')
    plt.plot(f, delay*rad, label='delay')
    plt.plot(f, azimuth*rad, label='azimuth')
    plt.plot(f, ellipticity_angle*rad, label='ellipticity_angle')
    plt.legend()
    plt.show()

    plt.plot(f, eccentricity, label='eccentricity')
    plt.legend()
    plt.show()

    #circ_pol_deg = np.array(Jout_l.parameters.degree_circular_polarization())

    #Jout_l.draw_ellipse()
    #plt.show()
    for i in range(0, len(Ex)):
        if i % 25 != 0:
            continue
        if i < 300:
            pass
        print(i)
        #print(str(np.round((1/THz)*f[i], 2)))
        freq = str(np.round(f[i]*(1/THz), 3))
        fact = 1#np.max([Ex[i,:], Ey[i,:]])
        #print(Ex[i,:]/fact, Ey[i,:]/fact)
        plt.plot(Ex[i,:]/fact, Ey[i,:]/fact, label=freq)
    plt.ylim((-1.1, 1.1))
    plt.xlim((-1.1, 1.1))
    #plt.ylim((-5*10**-7, 5*10**-7))
    #plt.xlim((-5*10**-7, 5*10**-7))
    plt.legend()
    plt.show()

    #plt.plot(int_x)
    #plt.plot(int_y)
    #plt.show()
    #Jout = Jhi * Jin
    #Jout[::3].draw_ellipse()
    #Jout.normalize()
    #print(Jout.parameters.delay())
    #print(Jout.parameters)
    #plt.show()
    #Jout_l.draw_ellipse()
    #plt.show()
    #Jout_c.draw_ellipse()
    #plt.show()
    #Jout.draw_ellipse()
    #plt.show()

    #print(Jout.parameters)

    #A, B = j[:, 0, 0], j[:, 0, 1]
    #res_mass = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))

    #Jhwpi = jones_matrix.create_Jones_matrices()
    #Jhwpi.half_waveplate(azimuth=pi/4)

    #Jlin = jones_vector.create_Jones_vectors()
    #Jlin.linear_light()

    #Jhi = jones_matrix.create_Jones_matrices()
    #Jhi.retarder_linear(R=res_mass)

    #Jo = Jhwpi*Jlin

    #plt.plot(2*pi-Jo.parameters.delay())
    #plt.plot(res_mass)
    #plt.show()

    #Jo.draw_ellipse()
    #plt.show()
    #plt.plot(res_mass)
    #plt.show()
    #plt.plot(2*(Jout.parameters.delay()-pi)/pi)
    #plt.plot(2*delt_min / pi, label='delt min')
    #plt.plot(2*delt/pi, label='wu chipman')
    #plt.plot(2*res_mass/pi, label='masson')
    #plt.legend()
    #plt.show()

