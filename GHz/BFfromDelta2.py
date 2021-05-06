import numpy as np
import matplotlib.pyplot as plt
from functions import get_einsum
from scipy.constants import c as c0
from numpy import pi, sin, cos, exp
from results import d_ghz, angles_ghz
from py_pol import jones_matrix
from py_pol import jones_vector


um = 10**6
THz = 10**12


f_measured = np.load('f.npy')
delta_measured = np.load('delta.npy')

m, n = len(f_measured), 4
einsum_str, einsum_path = get_einsum(m, n)
f_measured = f_measured.reshape((m, 1))
wls = ((c0/f_measured)*um)

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

n_s_range = np.linspace(1.0, 1.45, 500)
n_p_range = np.linspace(1.0, 1.45, 500)

idx = 450

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=0*pi/180)

angles = np.arange(0,370,10)

def calc_delta(n_s, n_p):
    j = j_stack(n_s, n_p)

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_out = J * Jin_l

    delta = np.array(J_out.parameters.delay())

    return delta

img = np.load(f'{idx}.npy')

plt.imshow(img)
plt.show()
