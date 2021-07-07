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
    material = 'MUT1 0deg noFP_D=2000'
    material = 'S2 0deg noFP_D=2000'
    eps_mat1 = load_custom_material(material, f_measured)

    #eps_mat1, _, _, _ = load_material_data('HIPS_MUT_1_1_constEps')  # HIPS_HHI_linePrnt

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

    return np.array([n_s, n_p+delta_bf, k_s, k_p])

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


def calc_delta_measlike(delta_bf, idx):
    n_s, n_p, _, _ = form_birefringence(delta_bf)
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
        J_P.diattenuator_linear(p1=1, p2=0, azimuth=(angle) * pi / 180)

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
    resolution = 10
    f = f_measured[::resolution]
    from generate_plotdata import export_csv
    #plt.plot(f_measured / 10 ** 9, delta_measured, label='Delta measured')

    delta_bf_line = np.arange(0, 0.045, 0.005)
    average_deviation = []
    for i, delta_bf in enumerate(delta_bf_line):
        print(delta_bf, i+1, len(delta_bf_line))
        phase_shift = np.array([])
        for idx in range(m):
            if idx % resolution:
                continue
            delta_calc = calc_delta_measlike(delta_bf, idx)
            phase_shift = np.append(phase_shift, delta_calc)
        average_deviation.append(0.5*pi/np.mean(phase_shift))

    plt.plot(delta_bf_line, average_deviation)
    #plt.plot(f_measured[::resolution] / 10 ** 9, phase_shift, label=f'Form BF + {delta_bf}')

    plt.xlabel('Form BF + x')
    plt.ylabel('0.5*pi/avg(phase shift)')
    plt.legend()
    plt.show()

    """
    n_s, n_p, _, _ = form_birefringence(0)

    df = pandas.read_csv(r'E:\CURPROJECT\SimsV3\GHz\FullPlates_bf.csv')
    #ZigZag,2mm,8mm
    plt.plot(df['freq'], df['ZigZag'], label='ZigZag')
    plt.plot(df['freq'], df['2mm'], label='2mm')
    plt.plot(df['freq'], df['8mm'], label='8mm')

    plt.plot(f_measured[::resolution]/10**9, best_fits, label='bf_difference')
    plt.plot(f_measured/10**9, n_p-n_s, label='form birefringence')
    plt.ylabel('Birefringence')
    plt.xlim((70, 115))
    plt.legend()
    plt.show()

    plt.plot(f_measured[::resolution]/10**9, best_fits_delta, label='best_fits_delta')
    plt.plot(f_measured/10**9, delta_measured, label='measured')
    plt.legend()
    plt.show()
    """