import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from scipy.constants import c as c0
from numpy import pi, cos, sin, dot
from py_pol import jones_vector

ntwk = rf.Network('100 deg_time_gated_bp_c0ps_s100ps_d20ps.s2p')

a_arr, b_arr = np.array([]), np.array([])
for idx in range(len(ntwk.f)):
    f = ntwk.f[idx]
    dn, L = 0.08, 1.0125*10**-2
    single_wp_delta = 2*pi*dn*L*f/c0

    Jin = np.array([1, 0])

    T = np.array([[cos(single_wp_delta/2)+1j*cos(2*pi/4)*sin(single_wp_delta/2), 1j*sin(2*pi/4)*sin(single_wp_delta/2)],
                  [1j*sin(2*pi/4)*sin(single_wp_delta/2), cos(single_wp_delta/2)-1j*cos(2*pi/4)*sin(single_wp_delta/2)]])

    Jout = dot(T, Jin)

    J = jones_vector.create_Jones_vectors()
    J.from_matrix(Jout)

    a, b = J.parameters.ellipse_axes()

    a_arr = np.append(a_arr, a)
    b_arr = np.append(b_arr, b)

plt.plot(ntwk.f, a_arr, label='a')
plt.plot(ntwk.f, b_arr, label='b')
plt.legend()
plt.show()

from generate_plotdata import export_csv
export_csv({'freq': ntwk.f, 'b/a': b_arr/a_arr}, 'standard_waveplate_bDiva.csv')


plt.plot(ntwk.f, b_arr/a_arr, label='b/a')
plt.legend()
plt.show()