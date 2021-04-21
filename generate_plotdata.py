import numpy as np
from consts import *
from functions import setup
from py_pol import jones_matrix, jones_vector
import pathlib
import pandas as pd
from plotting import draw_ellipse
from functions import loss

"""
data format: {'name_array1': array1, 'name_array2': array2, ...}
"""
def export_csv(data, path):
    df = pd.DataFrame(data=data)
    df.to_csv(path)


def pe_export(f, jones_vec, path, normalize):
    Ex, Ey = draw_ellipse(jones_vec, return_values=True)
    data = {}
    for ind, freq in enumerate(f.flatten()):
        col_name = str(round((1/THz)*freq, 2))
        normalization_factor = 1
        if normalize:
            normalization_factor /= np.max([Ex[ind, :], Ey[ind, :]])
        data[col_name + '_X'] = Ex[ind, :]*normalization_factor
        data[col_name + '_Y'] = Ey[ind, :]*normalization_factor

    export_csv(data, path=path)


if __name__ == '__main__':
    from results import *
    result = result1

    res_name = result['name']
    dir = plot_data_dir / Path(res_name)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    j, f, wls = setup(result, return_vals=True)
    f_flat = f.flatten()

    J = jones_matrix.create_Jones_matrices(result['name'])
    J.from_matrix(j)

    Jin_c = jones_vector.create_Jones_vectors('LCP')
    Jin_c.circular_light(kind='l')

    Jin_l = jones_vector.create_Jones_vectors('LP_0')
    Jin_l.linear_light()

    Jout_l = J * Jin_l
    Jout_c = J * Jin_c

    lp_circ_pol_deg = Jout_l.parameters.degree_circular_polarization()
    lp_lin_pol_deg = Jout_l.parameters.degree_linear_polarization()

    c_circ_pol_deg = Jout_c.parameters.degree_circular_polarization()
    c_lin_pol_deg = Jout_c.parameters.degree_linear_polarization()

    alpha = Jout_l.parameters.alpha()
    delay = Jout_l.parameters.delay()

    azimuth = Jout_l.parameters.azimuth()
    ellipticity_angle = Jout_l.parameters.ellipticity_angle()

    export_csv({'freq': f_flat,
          'alpha': alpha * rad, 'delay': delay * rad,
          'azimuth': azimuth * rad, 'ellipticity_angle': ellipticity_angle * rad},
               path=dir / Path('params.csv'))

    L = loss(j)
    export_csv({'freq': f_flat, 'L': L}, path=dir / Path('L.csv'))

    # not really interested in pol deg for circ input
    export_csv({'freq': f_flat, 'lp_cp_deg': lp_circ_pol_deg}, path=dir / Path('lp_cp_deg.csv'))
    export_csv({'freq': f_flat, 'lp_lp_deg': lp_lin_pol_deg}, path=dir / Path('lp_lp_deg.csv'))

    pe_export(f, Jout_l, path=dir / Path('lp_pe.csv'), normalize=True)
    pe_export(f, Jout_c, path=dir / Path('cp_pe.csv'), normalize=True)


