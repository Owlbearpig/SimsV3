import numpy as np
from numpy import power, outer, sqrt, exp, sin, cos, conj, dot, pi, einsum, arctan, array, arccos, conjugate
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from py_pol import jones_matrix
from py_pol.mueller import Mueller
from py_pol import jones_vector
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
import string
import matplotlib.pyplot as plt
import sys

THz = 10**12
m_um = 10**6 # m to um conversion


def opt(n, res_only=False, x=None, ret_j=False):
    d = np.ones(n)*204#*4080/n
    d = array([3360, 6730, 6460, 3140, 3330, 8430])
    delta = pi * outer(bf / wls, d)  # delta/2

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

    def erf(angles):
        #d, angles = x[0:n], x[n:2*n]
        #d, angles = x[0:n], np.deg2rad(x[n:2*n])
        #angles = np.deg2rad(x[0:n])

        j = np.zeros((m, n, 2, 2), dtype=complex)

        sd = 1j*sin(delta)

        sdca = sd*cos(2 * angles)

        j[:, :, 0, 0] = j[:, :, 1, 1] = cos(delta)
        j[:, :, 0, 1] = j[:, :, 1, 0] = sd*sin(2 * angles)
        j[:, :, 0, 0] += sdca
        j[:, :, 1, 1] -= sdca

        np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

        j = j[:, 0]
        A, B = j[:, 0, 0], j[:, 0, 1]

        delta_equiv = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))

        if ret_j:
            return j

        if res_only:
            return delta_equiv

        return (1/(n*m))*np.sum((delta_equiv-pi/2)**2) #return np.sum((2*delta_equiv/pi-1)**2)


    def print_fun(x, f, accepted):
        print(x, f, accepted)

    bounds = list(zip([0]*n, [2*pi]*n))

    minimizer_kwargs = {}

    class MyBounds(object):
        def __init__(self):
            self.xmax = np.ones(n)*(2*pi)
            self.xmin = np.ones(n)*(0)
        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    mybounds = MyBounds()


    """ Custom step-function """
    class RandomDisplacementBounds(object):
        """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
            Modified! (dropped acceptance-rejection sampling for a more specialized approach)
        """
        def __init__(self, xmin, xmax, stepsize=0.5):
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

    x0 = np.random.random(n)*2*pi

    if res_only or ret_j:
        return erf(x)

    return basinhopping(erf, x0, niter=200, callback=print_fun, take_step=bounded_step, disp=True)


x20 =\
array([4.83958375, 5.46695672, 2.70029202, 1.59720805, 1.38049486,
       0.67858639, 0.47854223, 1.06206799, 4.40208982, 3.92385128,
       0.89218459, 0.38393526, 3.01364689, 3.41504969, 3.57882995,
       4.2264747 , 3.54250995, 0.30855872, 2.85858943, 0.48572504])
x6 = np.deg2rad(array([31.7, 10.4, 118.7, 24.9, 5.1, 69.0]))

def R(v):
    return array([[cos(v), sin(v)],
                  [-sin(v), cos(v)]])


if __name__ == '__main__':

    f = np.arange(0.2, 2.0, 0.005)*THz

    wls = (c0/f)*m_um
    m = len(wls)

    no = 2.156#3.39
    ne = 2.108#3.07
    bf = np.ones_like(f)*(no-ne)

    #np.random.seed(1000)
    n = 6

    j = opt(n=n, ret_j=True, x=x6)

    """
    j_rot = np.zeros_like(j)
    for i, freq in enumerate(f):
        j_rot[i] = dot(R(-40*(pi/180)), dot(j[i], R(40*(pi/180))))
    """
    A, B = j[:,0,0], j[:,0,1]
    res_mass = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j)
    retardance = J.parameters.retardance()
    M = Mueller()
    M.from_Jones(J)
    pyp_ = M.parameters.retardance()
    delt = 2*arccos(0.5*np.abs(j[:, 0, 0]+conjugate(j[:, 0, 0])))

    plt.plot(delt, label='wu chipman')
    plt.plot(res_mass, label='masson')
    plt.plot(retardance, label='py-pol')
    plt.plot(pyp_, label='py-pol M')
    plt.legend()
    plt.show()

    #J = jones_matrix.create_Jones_matrices()
    #J.from_matrix(res)
    #J.remove_global_phase()
    #print(J.parameters)

    """
    plt.plot(f, res1, label=f'n={40_1}')
    plt.plot(f, res2, label=f'n={40_2}')
    plt.ylim((0.46, 0.52))
    plt.legend()
    plt.show()
    """
