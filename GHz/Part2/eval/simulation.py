from py_pol import jones_matrix
from py_pol import jones_vector
from consts import *
from functions import setup
import matplotlib.pyplot as plt
from results import *

if __name__ == '__main__':
    p2_x = np.concatenate((p2_angles, p2_d))

    res = {
        'name': 'result_p2',
        'x': p2_x,
        'bf': 'intrinsic',
        'mat_name': ('7g_s', '7g_f')
    }

    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light()

    j, f, wls = setup(res, return_vals=True)
    f, wls = f[::50].flatten(), wls[::50].flatten()

    T = jones_matrix.create_Jones_matrices(res['name'])
    T.from_matrix(j[::50])

    J_out = T*Jin_l

    plt.plot(f/10**9, J_out.parameters.delay()/pi, label=f'{round(1, 3)}')

    plt.ylabel('delta/pi')
    plt.xlabel('freq (GHz)')
    plt.legend()
    plt.show()
