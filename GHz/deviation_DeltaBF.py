import numpy as np
import pandas
from numpy import pi
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from results import stripes_ghz, result_GHz, d_ghz, angles_ghz
from functions import material_values, get_einsum, load_material_data, load_custom_material
from scipy.constants import c as c0
from numpy import cos, sin, exp, power, outer, sqrt
from py_pol import jones_matrix
from py_pol import jones_vector
from scipy.optimize import curve_fit
from AuswertungImport import delta_measured_eval
from eps_from_bf_rytov import eps_from_bf


um = 10**6
GHz = 10**9
THz = 10**12
phi_offset = 6

delta_measured = np.load(f'delta6.npy')
f_measured = np.load('f.npy')
m, n  = len(f_measured), 4
einsum_str, einsum_path = get_einsum(m, n)

wls = ((c0/f_measured)*um).reshape((m, 1))

def form_birefringence(delta_bf):
    """
    :return: array with length of frequency, frequency resolved [ns, np, ks, kp]
    """
    #material = 'MUT1 0deg noFP_D=2000'
    #material = 'S2 0deg noFP_D=2000'
    #eps_mat1 = load_custom_material(material, f_measured)

    eps_mat1, _, _, _ = load_material_data('HIPS_MUT_1_1')  # HIPS_HHI_linePrnt

    eps_mat2 = np.ones_like(eps_mat1)

    l_mat1, l_mat2 = np.array([734.55, 392.95])
    #l_mat1, l_mat2 = stripes_ghz

    a = (1 / 3) * power(outer(1 / wls, (l_mat1 * l_mat2 * pi) / (l_mat1 + l_mat2)), 2)

    # first order s and p
    wp_eps_s_1 = outer((eps_mat2 * eps_mat1), (l_mat2 + l_mat1)) / (
            outer(eps_mat2, l_mat1) + outer(eps_mat1, l_mat2))

    wp_eps_p_1 = outer(eps_mat1, l_mat1 / (l_mat2 + l_mat1)) + outer(eps_mat2, l_mat2 / (l_mat2 + l_mat1))

    # 2nd order
    wp_eps_s_2 = wp_eps_s_1 + (a * power(wp_eps_s_1, 3) * wp_eps_p_1 * power((1 / eps_mat1 - 1 / eps_mat2), 2))
    wp_eps_p_2 = wp_eps_p_1 + (a * power((eps_mat1 - eps_mat2), 2))

    # returns
    n_p, n_s = (
        sqrt(abs(wp_eps_p_2) + wp_eps_p_2.real) / sqrt(2),
        sqrt(abs(wp_eps_s_2) + wp_eps_s_2.real) / sqrt(2)
    )
    k_p, k_s = (
        sqrt(abs(wp_eps_p_2) - wp_eps_p_2.real) / sqrt(2),
        sqrt(abs(wp_eps_s_2) - wp_eps_s_2.real) / sqrt(2)
    )

    return np.array([n_s, n_p+delta_bf, k_s, k_p])

def j_stack(n_s, n_p, k_s, k_p):
    j = np.zeros((m, n, 2, 2), dtype=complex)

    phi_s, phi_p = (2 * n_s * pi / wls) * d_ghz.T, (2 * n_p * pi / wls) * d_ghz.T
    alpha_s, alpha_p = -(2 * pi * k_s / wls) * d_ghz.T, -(2 * pi * k_p / wls) * d_ghz.T

    x, y = 1j * phi_s + alpha_s, 1j * phi_p + alpha_p

    angles = np.tile(angles_ghz, (m, 1))

    j[:, :, 0, 0] = exp(y) * sin(angles) ** 2 + exp(x) * cos(angles) ** 2
    j[:, :, 0, 1] = 0.5 * sin(2 * angles) * (exp(x) - exp(y))
    j[:, :, 1, 0] = j[:, :, 0, 1]
    j[:, :, 1, 1] = exp(x) * sin(angles) ** 2 + exp(y) * cos(angles) ** 2

    np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

    j = j[:, 0]

    return j

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=pi/180)

def calc_delta(delta_bf):
    n_s, n_p, k_s, k_p = form_birefringence(delta_bf)
    j = j_stack(n_s, n_p, k_s, k_p)
    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j)

    J_out = J*Jin_l

    delta = J_out.parameters.delay()

    return delta

if __name__ == '__main__':
    #resolution = 10
    #f = f_measured[::resolution]
    from generate_plotdata import export_csv

    plt.plot(f_measured/GHz, delta_measured/pi, label='delta measured')

    delta_bf_line = np.arange(0, 0.055, 0.0025)
    max_dev = []
    for i, delta_bf in enumerate(delta_bf_line):
        print(delta_bf, i+1, len(delta_bf_line))
        delta_calc = calc_delta(delta_bf)

        if i == 0:
            plt.plot(f_measured / GHz, delta_calc/pi, label=f'expected phase shift DeltaBF={delta_bf}')
            #np.save('delta_expected', delta_calc)
            plt.plot(f_measured / GHz, f_measured * 0 + 0.5 * 1.03, 'k--', label='+3%')
            plt.plot(f_measured / GHz, f_measured * 0 + 0.5 * 0.97, 'k--', label='-3%')
        if i != 0:
            #pass
            plt.plot(f_measured / GHz, delta_calc / pi, label=f'expected phase shift DeltaBF={delta_bf}')

        max_dev.append(np.max(0.5-delta_calc/pi))
        print(max_dev[-1])

    plt.legend()
    plt.show()

    plt.plot(delta_bf_line, max_dev)
    min_pnt, max_pnt = [delta_bf_line[np.argmin(max_dev)], min(max_dev)], \
                       [delta_bf_line[np.argmax(max_dev)], max(max_dev)]
    plt.plot([0, max_pnt[0]], [0, max_pnt[1]])
    plt.xlabel('Form BF + x')
    plt.ylabel(r'max|0.5 - delta($\lambda$)/pi|')
    plt.legend()
    plt.show()
