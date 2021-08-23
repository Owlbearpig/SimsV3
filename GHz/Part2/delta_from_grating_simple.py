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

stripes_ghz = np.array([734.55, 392.95]) #np.array([750, 450.1])
f_measured = np.load('f.npy')
phi = 4.7
delta_measured = np.load(f'delta_phi{phi}.npy')

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=0*pi/180)

angles = np.arange(0,370,10)

def func(phi, a,b,delta):
    phi = phi
    return np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)

def j_stack(f, n_s, n_p, k_s, k_p):
    m, n = len(f), 4
    einsum_str, einsum_path = get_einsum(m, n)
    n_s, n_p, k_s, k_p = n_s.reshape((m, 1)), n_p.reshape((m, 1)), k_s.reshape((m, 1)), k_p.reshape((m, 1))
    f = f.reshape((m, 1))

    wls = ((c0 / f) * um)

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

def calc_delta(f, n_s, n_p, k_s, k_p):
    j = j_stack(f, n_s, n_p, k_s, k_p)

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j)

    J_out = J * Jin_l

    delta = np.array(J_out.parameters.delay())

    return delta

def load_7_grating_data(id_ = 1, f_min=75, f_max=110):
    #csv_file_path = Path(r'E:\MEGA\AG\BFWaveplates\Data\GHz\Part II\part2gratingBF.csv')
    csv_file_path = Path(r'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/GHz/Part II/7gratingMeans.csv')

    g_idx = id_ # or 'mean' for avg of all 7 gratings

    df = pandas.read_csv(csv_file_path)

    f = array(df['frequency/GHz']*GHz)

    data_slice = ((f > f_min*GHz) & (f <= f_max*GHz))

    f = f[data_slice]
    eps_real_p, eps_real_s = array(df[f'eps_real_{g_idx}_p'])[data_slice], array(df[f'eps_real_{g_idx}_s'])[data_slice]
    eps_imag_p, eps_imag_s = array(df[f'eps_imag_{g_idx}_p'])[data_slice], array(df[f'eps_imag_{g_idx}_s'])[data_slice]

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
    for id_ in [*range(1, 8), 'mean']:
        if id_ not in ['mean']:
            continue
        f, n_s, n_p, k_s, k_p = load_7_grating_data(id_, 75, 110)

        delta_new = calc_delta(f, n_s, n_p, k_s, k_p)

        plt.plot(f / GHz, delta_new / pi, '.', label=f'$\delta \ 7Grating {id_}$')
        print(id_, np.sum(np.abs(delta_measured[:1225]-delta_new)))

    plt.plot(f_measured / GHz, delta_measured / pi, '.', label=r'$\delta \ Measured$')
    plt.plot(f / GHz, f * 0 + 0.5 * 1.03, 'k--', label='+3%')
    plt.plot(f / GHz, f * 0 + 0.5 * 0.97, 'k--', label='-3%')
    plt.grid(True)
    plt.xlabel('$f$ in GHz')
    plt.ylabel(r"$\frac{\delta}{\pi}$")
    plt.xlim([75, 110])
    plt.ylim([0.2, 0.6])
    plt.legend()
    plt.show()

    from generate_plotdata import export_csv
    export_csv({'freq': f, 'delta_7g': delta_new}, f'delta_g{phi}_bf.csv')


if __name__ == '__main__':
    main()
