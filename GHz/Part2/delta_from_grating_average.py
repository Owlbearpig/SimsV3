import numpy as np
import pandas
from numpy import cos, sin, pi, exp, sqrt, array
import matplotlib.pyplot as plt
from py_pol import jones_matrix
from py_pol import jones_vector
from results import d_ghz, angles_ghz, result_GHz
from functions import get_einsum, setup, material_values, form_birefringence
from scipy.constants import c as c0
from pathlib import Path
from scipy.optimize import curve_fit

um = 10**6
GHz = 10**9
THz = 10**12

stripes_ghz = np.array([750, 450.1])
f_measured = np.load('f.npy')
delta_measured = np.load('delta5.npy')

m, n = len(f_measured), 4
einsum_str, einsum_path = get_einsum(m, n)
f_measured = f_measured.reshape((m, 1))
wls = ((c0/f_measured)*um)

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=0*pi/180)

angles = np.arange(0,370,10)

def func(phi, a,b,delta):
    phi = phi
    return np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)

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

def calc_delta_measlike(n_s, n_p, k_s, k_p, idx):
    j = j_stack(n_s, n_p, k_s, k_p)
    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_A = jones_matrix.create_Jones_matrices('A')
    J_A.diattenuator_linear(p1=1, p2=0, azimuth=0 * pi / 180)

    phi = np.array([])
    s21 = np.array([])
    for angle in angles:
        J_P = jones_matrix.create_Jones_matrices('P')
        J_P.diattenuator_linear(p1=1, p2=0, azimuth=angle * pi / 180)

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

    return delta

def calc_delta(n_s, n_p, k_s, k_p, idx):
    j = j_stack(n_s, n_p, k_s, k_p)

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_out = J * Jin_l

    delta = np.array(J_out.parameters.delay())

    return delta

def load_7_grating_data():
    csv_file_path = Path(r'E:\MEGA\AG\BFWaveplates\Data\GHz\Part II\part2gratingBF.csv')

    df = pandas.read_csv(csv_file_path)

    f = df['frequency']*GHz
    eps_real_p, eps_real_s = array(df['eps_real_mean_p']), array(df['eps_real_mean_s'])
    eps_imag_p, eps_imag_s = array(df['eps_imag_mean_p']), array(df['eps_imag_mean_s'])

    eps_p, eps_s = eps_real_p + 1j*eps_imag_p, eps_real_s + 1j*eps_imag_s

    n_p, n_s = (
        sqrt(abs(eps_p) + eps_p.real) / sqrt(2),
        sqrt(abs(eps_s) + eps_s.real) / sqrt(2)
    )
    k_p, k_s = (
        sqrt(abs(eps_p) - eps_p.real) / sqrt(2),
        sqrt(abs(eps_s) - eps_s.real) / sqrt(2)
    )

    return f, n_s, n_p, k_s, k_p


def main():
    f, n_s_arr, n_p_arr, k_s_arr, k_p_arr = load_7_grating_data()

    delta_new = np.array([])
    f_new = np.array([])
    for idx, (n_s, n_p, k_s, k_p) in enumerate(zip(n_s_arr, n_p_arr, k_s_arr, k_p_arr)):
        if idx % 50 != 0:
            pass
        f_new = np.append(f_new, f[idx])
        delta_new = np.append(delta_new, calc_delta(n_s, n_p, k_s, k_p, idx))

    plt.plot(f_new / GHz, delta_new / pi, '.', label='$\delta \ 7Grating Meas (jones calc)$')

    eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
    stripes = stripes_ghz[-2], stripes_ghz[-1]
    n_s_arr, n_p_arr, k_s_arr, k_p_arr = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

    f_full = f.flatten()

    delta_calc = np.array([])
    delta_calc_measlike = np.array([])
    f = np.array([])
    for idx, (n_s, n_p, k_s, k_p) in enumerate(zip(n_s_arr, n_p_arr, k_s_arr, k_p_arr)):
        if idx % 50 != 0:
            continue
        print(idx)
        f = np.append(f, f_full[idx])
        delta_calc = np.append(delta_calc, calc_delta(n_s, n_p, k_s, k_p, idx))
        delta_calc_measlike = np.append(delta_calc_measlike, calc_delta_measlike(n_s, n_p, k_s, k_p, idx))

    plt.plot(f / GHz, delta_calc / pi, '.-', label='$\delta \ Expected (jones calc)$')
    plt.plot(f / GHz, delta_calc_measlike / pi, '.-', label='$\delta \ Expected (meas like)$')
    plt.plot(f_full / GHz, delta_measured / pi, '.', label=r'$\delta \ Measured$')
    plt.plot(f_full / GHz, f_full * 0 + 0.5 * 1.03, 'k--', label='+3%')
    plt.plot(f_full / GHz, f_full * 0 + 0.5 * 0.97, 'k--', label='-3%')
    plt.grid(True)
    plt.xlabel('$f$ in GHz')
    plt.ylabel(r"$\frac{\delta}{\pi}$")
    plt.xlim([75, 110])
    # plt.ylim([0.3, 0.6])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
