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


def export_csv(data, path, index=True):
    columns_same_length = True

    data_dict_keys = list(data.keys())
    for key in data_dict_keys:
        if len(data[key]) != len(data[data_dict_keys[0]]):
            columns_same_length = False

    if columns_same_length:
        df = pd.DataFrame(data=data)
        df.to_csv(path, index=index)
    else: # fill with NaNs
        pd_series_lst = []
        for key, column in data.items():
            pd_series_lst.append(pd.Series(column, name=key))

        df = pd.concat(pd_series_lst, axis=1)
        df.to_csv(path, index=index)


def pe_export(f, jones_vec, path, normalize):
    Ex, Ey = draw_ellipse(jones_vec, return_values=True)
    data = {}
    for ind, freq in enumerate(f.flatten()):
        col_name = str(round((1 / THz) * freq, 4))
        normalization_factor = 1
        if normalize:
            normalization_factor /= np.max([Ex[ind, :], Ey[ind, :]])
        data[col_name + '_X'] = Ex[ind, :] * normalization_factor
        data[col_name + '_Y'] = Ey[ind, :] * normalization_factor

    export_csv(data, path=path)


if __name__ == '__main__':
    from results import *

    delta6 = np.load('E:\CURPROJECT\SimsV3\GHz\delta6.npy')
    f = np.load(r'E:\CURPROJECT\SimsV3\GHz\f.npy')
    export_csv({'freqs': f, 'delta': delta6}, 'delta_measured_6degOffset.csv')
    exit()

    # result = result_masson_full
    # result = result1
    result = result_GHz

    res_name = result['name']
    dir = plot_data_dir / Path(res_name)
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    j, f, wls, n_s, n_p, k_s, k_p = setup(result, return_all=True)
    f_flat = f.flatten()

    J = jones_matrix.create_Jones_matrices(result['name'])
    J.from_matrix(j)

    Jin_c = jones_vector.create_Jones_vectors('RCP')
    Jin_c.circular_light(kind='r')

    # print(Jin_c)
    # exit()

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


    k_s, k_p = 4*pi*k_s/(wls*10**-4), 4*pi*k_p/(wls*10**-4)

    export_csv({'freq': f_flat, 'n_s': n_s.flatten(), 'n_p': n_p.flatten(), 'bf': abs(n_p.flatten()-n_s.flatten()),
                'k_s': k_s.flatten(), 'k_p': k_p.flatten()},
               path=dir / Path('effective_RI.csv'))

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

    intensity = Jout_l.parameters.intensity()
    export_csv({'freq': f_flat, 'intensity': 10 * np.log10(intensity)}, path=dir / Path('intensity.csv'))

    diattenuation = J.parameters.diattenuation()
    export_csv({'freq': f_flat, 'diattenuation': diattenuation}, path=dir / Path('diattenuation.csv'))
    retardance = J.parameters.retardance()
    export_csv({'freq': f_flat, 'retardance': retardance}, path=dir / Path('retardance.csv'))

    inhomogeneity = J.parameters.inhomogeneity()
    export_csv({'freq': f_flat, 'inhomogeneity': inhomogeneity}, path=dir / Path('inhomogeneity.csv'))

    v1, v2, E1, E2 = J.parameters.eig(as_objects=True)

    export_csv({'freq': f_flat, 'E1 ellipticity angle': E1.parameters.ellipticity_angle(),
                'E2 ellipticity angle': E2.parameters.ellipticity_angle()},
               path=dir / Path('eigenstate_ellipticity.csv'))

    export_csv({'freq': f_flat, 'E1 eccentricity': E1.parameters.eccentricity(),
                'E2 eccentricity': E2.parameters.eccentricity()}, path=dir / Path('eigenstate_eccentricity.csv'))

    export_csv({'freq': f_flat, 'E1 azimuth': E1.parameters.azimuth() * rad,
                'E2 azimuth': E2.parameters.azimuth() * rad}, path=dir / Path('eigenstate_azimuths.csv'))

    a, b = Jout_l.parameters.ellipse_axes()
    export_csv({'freq': f_flat, 'a': a, 'b': b}, path=dir / Path('ellipse_axes.csv'))