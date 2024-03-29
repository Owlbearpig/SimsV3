import numpy as np
from numpy import sqrt
from py_pol import jones_matrix
from py_pol import jones_vector
from consts import *
from functions import j_stack, wp_cnt, get_einsum, interpol
import matplotlib.pyplot as plt
from results import *
import pandas
from scipy.signal import savgol_filter

def material_vals():
    resolution = 1

    mat_paths = {
        '7g_f': Path('material_data/7grating_fast_s.csv'),
        '7g_s': Path('material_data/7grating_slow_p.csv'),
    }

    df_s = pandas.read_csv(ROOT_DIR / mat_paths['7g_s'])
    df_f = pandas.read_csv(ROOT_DIR / mat_paths['7g_f'])

    freq_dict_key = [key for key in df_s.keys() if "freq" in key][0]

    eps_mat_r_key_s = [key for key in df_s.keys() if "epsilon_r" in key][0]
    eps_mat_i_key_s = [key for key in df_s.keys() if "epsilon_i" in key][0]

    eps_mat_r_key_f = [key for key in df_s.keys() if "epsilon_r" in key][0]
    eps_mat_i_key_f = [key for key in df_s.keys() if "epsilon_i" in key][0]

    f = np.array(df_s[freq_dict_key])

    data_slice = np.where((f > 0) &
                          (f < 150*GHz))
    data_slice = data_slice[0][::resolution]
    m = len(data_slice)

    f = f[data_slice].reshape(m, 1)

    wls = (c0 / f) * m_um

    eps_mat_r_s, eps_mat_i_s = np.array(df_s[eps_mat_r_key_s])[data_slice], np.array(df_s[eps_mat_i_key_s])[data_slice]
    eps_mat_r_f, eps_mat_i_f = np.array(df_f[eps_mat_r_key_f])[data_slice], np.array(df_f[eps_mat_i_key_f])[data_slice]

    eps_mat_s = (eps_mat_r_s + eps_mat_i_s * 1j).reshape(m, 1)
    eps_mat_f = (eps_mat_r_f + eps_mat_i_f * 1j).reshape(m, 1)

    n_s, n_p = sqrt(np.abs(eps_mat_s) + eps_mat_s.real) / sqrt(2), sqrt(np.abs(eps_mat_f) + eps_mat_f.real) / sqrt(2)
    k_s, k_p = sqrt(np.abs(eps_mat_f) - eps_mat_f.real) / sqrt(2), sqrt(np.abs(eps_mat_f) - eps_mat_f.real) / sqrt(2)

    return n_s, n_p, k_s, k_p, f, wls, m

if __name__ == '__main__':
    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light()

    x = np.concatenate((p2_angles, p2_d))

    res = {
        'name': 'result_p2',
        'x': x,
        'bf': 'intrinsic',
        'mat_name': ('7g_s', '7g_f')
    }

    n = wp_cnt(res)

    n_s, n_p, k_s, k_p, f_design, wls, m_design = material_vals()
    f_design = f_design.flatten()

    f_meas, delay_meas = np.load('f_meas.npy'), np.load('delta_meas.npy')
    f_max_design_idx = np.argmin(np.abs(f_design[-1] - f_meas))
    f_meas, delay_meas = f_meas[:f_max_design_idx], delay_meas[:f_max_design_idx] / pi

    m_meas = len(f_meas)

    einsum_str, einsum_path = get_einsum(m_design, n)

    bf = n_s - n_p
    bf_smooth = savgol_filter(bf.flatten(), 101, 2)

    plt.plot(f_design / GHz, bf, label='original')
    plt.plot(f_design / GHz, bf_smooth, label='smoothed')
    plt.title('Design birefringence (n_s - n_p, 7grating avg.)')
    plt.show()

    design_delays = []

    bf_offset_resolution = 100
    bf_offsets = np.linspace(-0.05, 0.05, bf_offset_resolution)
    for bf_offset in bf_offsets:
        j = j_stack(x, m_design, n, wls, n_s, n_p + bf_offset, k_s, k_p, einsum_str, einsum_path)

        T = jones_matrix.create_Jones_matrices(res['name'])
        T.from_matrix(j)

        J_out = T * Jin_l

        delay_design = J_out.parameters.delay() / pi

        delay_design = np.interp(f_meas, f_design, delay_design)

        design_delays.append(delay_design)

    design_delays = np.array(design_delays)

    plt.plot(f_meas / GHz, delay_meas, '.', label='Delay measured')

    for i, delay in enumerate(design_delays):
        if i % 10 != 0:
            continue
        plt.plot(f_meas / GHz, delay, label=f'BF(7grating avg.)+{round(bf_offsets[i], 3)}')

    plt.xlim([75, 110])
    plt.ylabel('delta/pi')
    plt.xlabel('freq (GHz)')
    plt.title('Design delay compared to measurement for different const. bf offsets.')
    plt.legend()
    plt.show()

