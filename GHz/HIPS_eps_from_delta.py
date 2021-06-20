import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from results import stripes_ghz, result_GHz, d_ghz, angles_ghz
from functions import material_values, get_einsum, load_material_data
from scipy.constants import c as c0
from numpy import cos, sin, exp, power, outer, sqrt
from py_pol import jones_matrix
from py_pol import jones_vector
from scipy.optimize import curve_fit
from AuswertungImport import delta_measured_eval
from eps_from_bf_rytov import eps_from_bf

um = 10**6
THz = 10**12
phi_offset = 6

delta_measured = np.load(f'delta6.npy')
f_measured = np.load('f.npy')
m, n  = len(f_measured), 4
einsum_str, einsum_path = get_einsum(m, n)

wls = ((c0/f_measured)*um).reshape((m, 1))

plt.plot(f_measured, delta_measured)
plt.show()


def form_birefringence(delta_eps, idx):
    """
    :return: array with length of frequency, frequency resolved [ns, np, ks, kp]
    """
    eps_mat1, _, _, _ = load_material_data('HIPS_MUT_1_1')

    eps_mat1[idx] += delta_eps

    eps_mat2 = np.ones_like(eps_mat1)

    l_mat1, l_mat2 = np.array([734.55, 392.95])

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

    return np.array([n_s, n_p, k_s, k_p])

def j_stack(n_s, n_p):
    j = np.zeros((m, n, 2, 2), dtype=complex)

    phi_s, phi_p = (2 * n_s * pi / wls) * d_ghz.T, (2 * n_p * pi / wls) * d_ghz.T

    x, y = 1j * phi_s, 1j * phi_p
    angles = np.tile(angles_ghz, (m, 1))

    j[:, :, 0, 0] = exp(y) * sin(angles) ** 2 + exp(x) * cos(angles) ** 2
    j[:, :, 0, 1] = 0.5 * sin(2 * angles) * (exp(x) - exp(y))
    j[:, :, 1, 0] = j[:, :, 0, 1]
    j[:, :, 1, 1] = exp(x) * sin(angles) ** 2 + exp(y) * cos(angles) ** 2

    np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

    j = j[:, 0]

    return j

def func(phi, a,b,delta):
    phi = phi
    return np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=pi/180)


def calc_delta_measlike(delta_eps, idx):
    n_s, n_p, _, _ = form_birefringence(delta_eps, idx)
    j = j_stack(n_s, n_p)
    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_A = jones_matrix.create_Jones_matrices('A')
    J_A.diattenuator_linear(p1=1, p2=0, azimuth=0 * pi / 180)

    angles = np.arange(0, 370, 10)

    phi = np.array([])
    s21 = np.array([])
    for angle in angles:
        J_P = jones_matrix.create_Jones_matrices('P')
        J_P.diattenuator_linear(p1=1, p2=0, azimuth=(angle+6) * pi / 180)

        phi = np.append(phi, angle)
        J_out = J_A * J_P * J * Jin_l
        intensity = J_out.parameters.intensity()
        s21 = np.append(s21, np.sqrt(intensity))

    phi = np.deg2rad(phi)
    popt, pcov = curve_fit(func, phi, s21)

    # p1, p2, a, b, delta
    a = popt[0]
    b = popt[1]
    delta = popt[2]

    if idx < 0:
        plt.polar(phi, s21, label='messung')
        plt.plot(phi, func(phi, *popt), label='fit')
        plt.legend()
        plt.show()

    return delta

if __name__ == '__main__':
    best_fits, best_fits_delta = [], []
    for idx in range(m):
        if idx % 50:
            continue
        print(idx, delta_measured[idx])

        best_fit, min_deviation = None, 10
        delta_eps_line = np.linspace(-5, 0, 100)
        for delta_eps in delta_eps_line:
            delta_calc = calc_delta_measlike(delta_eps, idx)
            delta_meas = delta_measured[idx]
            if abs(delta_meas-delta_calc) < min_deviation:
                min_deviation = abs(delta_meas-delta_calc)
                best_fit = delta_eps
        best_fits.append(best_fit)
        best_fits_delta.append(calc_delta_measlike(best_fit, idx))

    print(best_fits)
    print(best_fits_delta)
    plt.plot(best_fits_delta, label='best_fits_delta')
    plt.plot(delta_measured[::50], label='measured')
    plt.legend()
    plt.show()
