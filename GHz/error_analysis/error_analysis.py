from py_pol import jones_matrix
from py_pol import jones_vector
from consts import *
from functions import setup
import matplotlib.pyplot as plt
from results import *

if __name__ == '__main__':
    from generate_plotdata import export_csv
    from results import d_ghz, angles_ghz, stripes_ghz

    res = result_GHz

    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light()

    data = {}
    for i in range(50):
        err = 10-20*np.random.random()*np.ones(4)
        print(err[0])

        x_ghz_err = np.concatenate((angles_ghz + np.deg2rad(err), d_ghz, stripes_ghz))

        res['x'] = x_ghz_err

        j, f, wls = setup(res, return_vals=True)
        f, wls = f[::50].flatten(), wls[::50].flatten()
        if i == 0:
            data['freq'] = f

        T = jones_matrix.create_Jones_matrices(res['name'])
        T.from_matrix(j[::50])

        J_out = T*Jin_l

        data[f'delta_{i}_{round(err[0], 2)}'] = J_out.parameters.delay()
        plt.plot(J_out.parameters.delay(), label=f'{i}')
    plt.legend()
    plt.show()
    print(data)
    export_csv(data, 'pm10deg_misalign.csv')