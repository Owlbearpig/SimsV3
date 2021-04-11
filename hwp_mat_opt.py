import numpy as np
from numpy import power, outer, sqrt, exp, sin, cos, conj, dot, pi, einsum, arctan, array, arccos, conjugate, flip
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from py_pol_calcs import jones_matrix
from py_pol_calcs import jones_vector
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
import string
import matplotlib.pyplot as plt
import sys

THz = 10 ** 12
m_um = 10 ** 6  # m to um conversion


def opt(n, x=None, ret_j=False):
    # d = np.ones(n)*204#*4080/n
    # d = flip(np.array([3360, 6730, 6460, 3140, 3330, 8430]), 0)

    # setup einsum_str
    s0 = string.ascii_lowercase + string.ascii_uppercase

    einsum_str = ''
    for i in range(n):
        einsum_str += s0[n + 2] + s0[i] + s0[i + 1] + ','

    # remove last comma
    einsum_str = einsum_str[:-1]
    einsum_str += '->' + s0[n + 2] + s0[0] + s0[n]

    # einsum path
    test_array = np.zeros((n, m, 2, 2))
    einsum_path = np.einsum_path(einsum_str, *test_array, optimize='greedy')

    # calc matrix chain from m_n_2_2 tensor
    # (If n > 2x alphabet length, einsum breaks -> split into k parts... n/k)
    """
    1_n_(2x2)*...*1_3_(2x2)*1_2_(2x2)*1_1_(2x2) -> (2x2)_1
    2_n_(2x2)*...*2_3_(2x2)*2_2_(2x2)*2_1_(2x2) -> (2x2)_2
    .
    .
    m_n_(2x2)*...*m_3_(2x2)*m_2_(2x2)*m_1_(2x2) -> (2x2)_m
    """

    def matrix_chain_calc(matrix_array):
        return

    s = 8 * 10 ** 3 / (2 * pi)

    def erf(x):
        # d, angles = x[0:n], x[n:2*n]
        # d, angles = x[0:n], np.deg2rad(x[n:2*n])
        # angles = np.deg2rad(x[0:n])

        j = np.zeros((m, n, 2, 2), dtype=complex)

        angles, d = x[0:n], x[n:2 * n] * s

        delta = pi * outer(bf / wls, d)  # delta/2
        sd = 1j * sin(delta)

        sdca = sd * cos(2 * angles)

        j[:, :, 0, 0] = j[:, :, 1, 1] = cos(delta)
        j[:, :, 0, 1] = j[:, :, 1, 0] = sd * sin(2 * angles)
        j[:, :, 0, 0] += sdca
        j[:, :, 1, 1] -= sdca

        np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

        j = j[:, 0]

        if ret_j:
            return j

        # delta_equiv = 2*arccos(0.5*np.abs(j[:, 0, 0]+conjugate(j[:, 0, 0])))

        # hwp 1 int opt
        # res = (1 / m) * sum((1 - j[:, 1, 0] * conj(j[:, 1, 0])) ** 2 + (j[:, 0, 0] * conj(j[:, 0, 0])) ** 2)

        # hwp 2 int opt
        # res = (1 / m) * sum((1 - j[:, 1, 0].real)**2 + (j[:, 1, 0].imag) ** 2 + (j[:, 0, 0] * conj(j[:, 0, 0])) ** 2)

        # hwp 3 mat opt
        # print(((np.angle(j[:,1,0])-np.angle(j[:,0,1]))**2))
        # print((j[:, 1, 0].imag - j[:, 0, 1].imag) ** 2)
        # print()
        """
        res = (1 / m) * sum(np.absolute(j[:,0,0])**2+np.absolute(j[:,1,1])**2+
                            #(1-np.abs(j[:,1,0].real))**2+(1-np.abs(j[:,0,1].real))**2)
                            (1-j[:,1,0].real)+(1-j[:,0,1].real)+
                            (j[:,1,0].imag)**2+(j[:,0,1].imag)**2)
        """

        # hwp 4 mat opt back to start
        """
        print(np.absolute(j[:, 0, 0]) ** 2)
        print(np.absolute(j[:, 1, 1]) ** 2)
        print((1-j[:, 0, 1].imag) ** 2)
        print((1-j[:, 1, 0].imag) ** 2)
        print()
        """
        res = sum(np.absolute(j[:, 0, 0]) ** 2 + np.absolute(j[:, 1, 1]) ** 2) \
              + sum((1 - j[:, 0, 1].imag) ** 2 + (1 - j[:, 1, 0].imag) ** 2)

        # qwp state opt
        # q = j[:, 0, 0] / j[:, 1, 0]
        # res = (1 / m) * sum(q.real ** 2 + (q.imag - 1) ** 2)

        # Masson ret. opt.
        # A, B = j[:, 0, 0], j[:, 0, 1]
        # delta_equiv = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))
        # res = (1/m)*np.sum((delta_equiv-pi)**2)

        return res

    def print_fun(x, f, accepted):
        print(x, f, accepted)

    bounds = list(zip([0] * n, [4 * pi] * n)) + list(zip([0] * n, [16 * pi] * n))

    minimizer_kwargs = {}

    class MyBounds(object):
        def __init__(self):
            self.xmax = np.ones(2 * n) * (16 * pi)
            self.xmin = np.ones(2 * n) * (0)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))

            return (tmax and tmin)

    mybounds = MyBounds()

    """ Custom step-function """

    class RandomDisplacementBounds(object):
        """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
            Modified! (dropped acceptance-rejection sampling for a more specialized approach)
        """

        def __init__(self, xmin, xmax, stepsize=2.4):
            self.xmin = xmin
            self.xmax = xmax
            self.stepsize = stepsize

        def __call__(self, x):
            """take a random step but ensure the new position is within the bounds """
            min_step = np.maximum(self.xmin - x, -self.stepsize)
            max_step = np.minimum(self.xmax - x, self.stepsize)

            random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
            xnew = x + random_step

            return xnew

    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))

    x0 = np.concatenate((np.random.random(n) * 16 * pi, np.random.random(n) * 16 * pi))

    if ret_j:
        return erf(x)

    #return minimize(erf, x0)
    return basinhopping(erf, x0, niter=100, callback=print_fun, take_step=bounded_step, disp=True)




def R(v):
    return array([[cos(v), sin(v)],
                  [-sin(v), cos(v)]])


if __name__ == '__main__':
    f = (np.arange(0.25, 1.6, 0.05) * THz)[:]

    wls = (c0 / f) * m_um
    m = len(wls)

    no = 3.39 # 2.108  # 3.39
    ne = 3.07 # 2.156  # 3.07
    bf = np.ones_like(f) * (no - ne)

    # np.random.seed(1000)
    n = 12

    ret = opt(n=12)
    print(ret)

    j = opt(n=6, ret_j=True, x=ret.x)

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j)

    v1, v2, E1, E2 = J.parameters.eig(as_objects=True)
    R = J.parameters.retardance()
    alpha = pi / 2 - E1.parameters.alpha()
    delta = E1.parameters.delta() + pi

    # INPUT
    Jin_c = jones_vector.create_Jones_vectors('Jin_c')
    Jin_c.circular_light(kind='l')
    Jin_l = jones_vector.create_Jones_vectors('Jin_l')
    Jin_l.linear_light(azimuth=0 * pi / 2)
    Jin_l.draw_ellipse()
    plt.show()
    Jout_l = J * Jin_l
    Jout_c = J * Jin_c
    print(Jout_l.parameters.azimuth())
    Jout_l.draw_ellipse()
    plt.show()
    # Jout_c.draw_ellipse()
    # plt.show()

    J_w = (1 / sqrt(2)) * array([1j * cos(2 * pi / 4) + 1, -1j * sin(2 * pi / 4)])
    J_ = jones_vector.create_Jones_vectors('J_w')
    J_.from_matrix(J_w)
    J_.draw_ellipse()
    plt.show()
    # print(Jout.parameters)

    A, B = j[:, 0, 0], j[:, 0, 1]
    res_mass = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))

    Jhwpi = jones_matrix.create_Jones_matrices()
    Jhwpi.half_waveplate(azimuth=pi / 4)

    Jlin = jones_vector.create_Jones_vectors()
    Jlin.linear_light()

    J0 = jones_vector.create_Jones_vectors()
    J0.from_matrix([-1, 1])
    # J0.draw_ellipse()
    # plt.show()

    Jhi = jones_matrix.create_Jones_matrices()
    Jhi.retarder_linear(R=res_mass)

    Jo = Jhwpi * Jlin

