import numpy as np
from numpy import sqrt
from py_pol import jones_matrix
from py_pol import jones_vector
from consts import *
from functions import j_stack, wp_cnt, get_einsum, interpol
import matplotlib.pyplot as plt
import pandas
from pol_offset_cov import meas_with_polOffset
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

    pol_offset = -2.25
    f_meas, delay_meas, rel, eta, var1, var2, var3 = meas_with_polOffset(pol_offset, rez=200)
    delay_meas /= pi

    angle_misalignment = 4
    d_err = array([-400, 0, 0])
    angle_err = array([0, 0, 0])

    p2_angles = np.deg2rad(array([246.54, 171.27, 38.65]))
    p2_d = array([14136.4, 13111.6, 6995.5])

    angles_with_error = np.rad2deg(p2_angles) + angle_err + angle_misalignment
    p2_d_with_error = p2_d + d_err

    x = np.concatenate((np.deg2rad(angles_with_error), p2_d_with_error))

    res = {
        'name': 'result_p2',
        'x': x,
        'bf': 'intrinsic',
    }

    n = wp_cnt(res)

    n_s, n_p, k_s, k_p, f_design, wls, m_design = material_vals()
    f_design = f_design.flatten()

    m_meas = len(f_meas)

    einsum_str, einsum_path = get_einsum(m_design, n)

    bf = n_s - n_p
    bf_smooth = savgol_filter(bf.flatten(), 101, 2)

    plt.plot(f_design / GHz, bf, label='original')
    plt.plot(f_design / GHz, bf_smooth, label='smoothed')
    plt.title('Design birefringence (n_s - n_p, 7grating avg.)')
    plt.legend()
    plt.show()

    n_s_smooth = savgol_filter(n_s.flatten(), 101, 2).reshape(n_s.shape)
    n_p_smooth = savgol_filter(n_p.flatten(), 101, 2).reshape(n_p.shape)

    design_delays = []

    bf_offset_resolution = 200
    bf_offsets = np.append(np.linspace(-0.06, 0.06, bf_offset_resolution), 0)
    for bf_offset in bf_offsets:
        j = j_stack(x, m_design, n, wls, n_s_smooth, n_p_smooth + bf_offset, k_s, k_p, einsum_str, einsum_path)

        T = jones_matrix.create_Jones_matrices(res['name'])
        T.from_matrix(j)

        J_out = T * Jin_l

        delay_design = J_out.parameters.delay() / pi

        design_delays.append(delay_design)

    design_delays = np.array(design_delays)

    plt.plot(f_meas / GHz, delay_meas, '-.', label=f'Delay measured \n \\w {pol_offset} deg. pol. offset')

    for i, delay in enumerate(design_delays):
        if i % (bf_offset_resolution // 5) != 0 and not i == np.argmin(np.abs(bf_offsets)):
            continue
        plt.plot(f_design / GHz, delay, label=f'BF(7grating avg.)+{round(bf_offsets[i], 3)}')

    plt.plot(f_meas / 10 ** 9, f_meas * 0 + 0.5 * 1.1, 'k--', label='0,5+10%')
    plt.plot(f_meas / 10 ** 9, f_meas * 0 + 0.5 * 0.9, 'k--', label='0,5-10%')

    #plt.xlim([75, 110])
    plt.ylabel('delta/pi')
    plt.xlabel('freq (GHz)')
    plt.title(f'Design: {angle_misalignment} deg. angle misalignment, const. bf offsets\n'
              f'AngleErr: {angle_err} (deg), '
              f'dErr: {d_err} (um).\n'
              f'Eval: {pol_offset} deg. pol. offset\n')
    plt.legend()
    plt.show()

    f_intersection_idx_l = np.argmin(np.abs(f_design[0] - f_meas))
    f_intersection_idx_r = np.argmin(np.abs(f_design[-1] - f_meas))

    delay_meas_interp = np.interp(f_design,
                                  f_meas[f_intersection_idx_l:f_intersection_idx_r],
                                  delay_meas[f_intersection_idx_l:f_intersection_idx_r])

    delay_diff = design_delays - delay_meas_interp

    bf_fitted = []
    for i in range(len(delay_meas_interp)):
        bf_fitted.append(bf_offsets[np.argmin(np.abs(delay_diff[:, i]))])

    bf_corrected = bf_smooth + np.array(bf_fitted)

    plt.plot(f_design / GHz, bf_smooth, label='Bf original (7grating avg.)')
    plt.plot(f_design / GHz, bf_corrected, label='Bf original + bf offset')
    #plt.xlim([75, 110])
    plt.ylabel('Birefringence')
    plt.xlabel('freq (GHz)')
    plt.title(f'BF original vs BF fitted...')
    plt.legend()
    plt.show()
