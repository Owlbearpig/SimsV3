import numpy as np
from py_pol import jones_matrix
from py_pol import jones_vector
from consts import *
from functions import setup
import matplotlib.pyplot as plt
from results import *

if __name__ == '__main__':
    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light()

    f_meas, delta_meas = np.load('eval/f_meas.npy'), np.load('eval/delta_meas.npy')
    rel_meas = np.load('eval/rel_meas_0degPolOffset.npy')

    res = {
        'name': 'result_p2',
        'x': p2_thick_single_plate_x,
        'bf': 'intrinsic',
        'mat_name': ('7g_s', '7g_f')
    }

    j, f, _ = setup(res, return_vals=True)
    f = f[::25].flatten()

    T = jones_matrix.create_Jones_matrices(res['name'])
    T.from_matrix(j[::25])

    J_out = T * Jin_l

    a, b = J_out.parameters.ellipse_axes()

    plt.plot(f_meas / 10 ** 9, rel_meas, label=f'b/a meas')
    plt.plot(f / 10 ** 9, b/a, label=f'b/a thick plate sim')
    plt.title('ellipse axis ratio, measurement and thick plate (45deg, sum(d_design))')
    plt.legend()
    plt.show()

    plt.plot(f_meas / 10 ** 9, delta_meas / pi, '--', label=f'delta_measured')
    plt.plot(f/10**9, J_out.parameters.delay() / pi, label=f'p2_thick_single_plate_x')
    plt.title(f'Measurement compared to sim of single thick plate at 45 deg. \n {p2_thick_single_plate_d} $\mu m$')
    plt.xlim([75, 110])
    plt.ylim([0.4, 0.6])
    plt.ylabel('delta/pi')
    plt.xlabel('freq (GHz)')
    plt.legend()
    plt.show()
