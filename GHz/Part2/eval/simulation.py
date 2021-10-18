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

    f_meas, delta_meas = np.load('f_meas.npy'), np.load('delta_meas.npy')
    plt.plot(f_meas / 10 ** 9, delta_meas / pi, '--', label=f'delta_measured')

    p2_x = np.concatenate((p2_angles, p2_d))

    res = {
        'name': 'result_p2',
        'x': p2_x,
        'bf': 'intrinsic',
        'mat_name': ('7g_s', '7g_f')
    }

    angle_errs = np.linspace(-5, 5, 11)

    for angle_err in angle_errs:
        p2_angles_new = p2_angles.copy()
        p2_angles_new = np.rad2deg(p2_angles_new) + angle_err
        p2_angles_new = np.deg2rad(p2_angles_new)
        res['x'] = np.concatenate((p2_angles_new, p2_d))

        j, f, _ = setup(res, return_vals=True)
        f = f[::50].flatten()

        T = jones_matrix.create_Jones_matrices(res['name'])
        T.from_matrix(j[::50])

        J_out = T*Jin_l

        plt.plot(f/10**9, J_out.parameters.delay() / pi, label=f'angles + {angle_err} deg')

    plt.xlim([75, 110])
    plt.ylabel('delta/pi')
    plt.xlabel('freq (GHz)')
    plt.title('Misalignment error. Same angle added to all plates.')
    plt.legend()
    plt.show()
