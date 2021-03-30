import numpy as np
from numpy import power, outer, sqrt, exp, sin, cos, conj, dot, pi, einsum, arctan, array, arccos, conjugate, flip
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from py_pol import jones_matrix
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

s = 500/(2*pi)

def opt(n, x=None, ret_j=False):
    #d = np.ones(n)*204#*4080/n
    #d = flip(np.array([3360, 6730, 6460, 3140, 3330, 8430]), 0)

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

    def erf(x):
        d, angles = x[0:n]*s, x[n:2*n]
        #d, angles = x[0:n], np.deg2rad(x[n:2*n])
        #angles = np.deg2rad(x[0:n])

        #d = flip(array([566.0, 377.4, 377.4, 377.4, 377.4, 377.4]), 0)
        #angles = flip(np.deg2rad(array([12.3, 56.7, 28.1, 356.5, 302.0, 349.2])), 0)

        #d = flip(array([3360, 6730, 6460, 3140, 3330, 8430]), 0)
        #angles = flip(np.deg2rad(array([31.7, 10.4, 118.7, 24.9, 5.1, 69.0])), 0)

        j = np.zeros((m, n, 2, 2), dtype=complex)

        #angles = x[0:n]
        #d = np.ones_like(angles)*204#x[-1]

        delta = pi * outer(bf / wls, d)  # delta/2
        sd = 1j*sin(delta)

        sdca = sd*cos(2 * angles)

        j[:, :, 0, 0] = j[:, :, 1, 1] = cos(delta)
        j[:, :, 0, 1] = j[:, :, 1, 0] = sd*sin(2 * angles)
        j[:, :, 0, 0] += sdca
        j[:, :, 1, 1] -= sdca

        np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

        j = j[:, 0]

        # Masson ret. opt.
        A, B = j[:, 0, 0], j[:, 0, 1]
        delta_equiv = 2 * arctan(sqrt((A.imag ** 2 + B.imag ** 2) / (A.real ** 2 + B.real ** 2)))
        res = (1/m)*np.sum((delta_equiv-pi/2)**2)

        if ret_j:
            return j
        """
        print(np.absolute(j[:, 0, 0]))
        print(np.absolute(j[:, 1, 1]))
        print((1-j[:, 0, 1].imag) ** 2)
        print((1-j[:, 1, 0].imag) ** 2)
        print()
        """
        #res = sum(np.absolute(j[:, 0, 0])**2 + np.absolute(j[:, 1, 1])**2)+\
        #      sum((1-j[:, 0, 1].imag) ** 2 + (1-j[:, 1, 0].imag) ** 2)

        # qwp state opt
        #q = j[:, 0, 0] / j[:, 1, 0]
        #res = sum(q.real ** 2 + (q.imag - 1) ** 2)

        return res

    def print_fun(x, f, accepted):
        print(x, f, accepted)

    #bounds = list(zip([0]*n, [2*pi]*n)) + [(0, 1000)]
    bounds = list(zip([0] * 2*n, [2 * pi] * 2*n))

    minimizer_kwargs = {}

    class MyBounds(object):
        def __init__(self):
            self.xmax = np.ones(2*n)*(2*pi)
            self.xmin = np.ones(2*n)*(0)
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
            #xnew[-1] += np.random.random()*10
            return xnew

    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))

    x0 = np.random.random(2*n)*2*pi
    #x0 = np.concatenate((np.random.random(n) * 2 * pi, np.random.random(n)*500))

    if ret_j:
        return erf(x)

    #return minimize(erf, x0)
    return basinhopping(erf, x0, niter=100, callback=print_fun, take_step=bounded_step, disp=True)


x20 =\
array([4.83958375, 5.46695672, 2.70029202, 1.59720805, 1.38049486,
       0.67858639, 0.47854223, 1.06206799, 4.40208982, 3.92385128,
       0.89218459, 0.38393526, 3.01364689, 3.41504969, 3.57882995,
       4.2264747 , 3.54250995, 0.30855872, 2.85858943, 0.48572504])

x20_1 = \
array([4.83958375, 5.46695672, 2.70029202, 1.59720805, 1.38049486,
       0.67858639, 0.47854223, 1.06206799, 4.40208982, 3.92385128,
       0.89218459, 0.38393526, 3.01364689, 3.41504969, 3.57882995,
       4.2264747 , 3.54250995, 0.30855872, 2.85858943, 0.48572504])

def R(v):
    return array([[cos(v), sin(v)],
                  [-sin(v), cos(v)]])

if __name__ == '__main__':

    f = (np.arange(0.1, 2.0, 0.05)*THz)[::2]

    wls = (c0/f)*m_um
    m = len(wls)

    no = 3.07#2.156#
    ne = 3.39#2.108#
    bf = np.ones_like(f)*(no-ne)

    #np.random.seed(1000)
    n = 12

    ret = opt(n=n)
    print(ret)

    j = opt(n=12, ret_j=True, x=ret.x)
    #plt.plot(j/pi)
    #plt.show()
    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j)
    #J.remove_global_phase()
    #J.set_global_phase(0)
    #J.analysis.retarder(verbose=True)
    """ 
    print(j[:,0,0])
    print(j[:, 0, 1])
    print(j[:, 1, 0])
    print(j[:, 1, 1])
    """
    v1, v2, E1, E2 = J.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()
    #J.parameters.global_phase(verbose=True)
    #Jqwp = jones_matrix.create_Jones_matrices()
    #Jqwp.retarder_linear()

    #print(J.analysis.retarder(verbose=True))
    #E1[11].draw_ellipse()
    #E2[11].draw_ellipse()
    #plt.show()
    #E1.draw_ellipse()
    #E2.draw_ellipse()

    #print(360-np.abs(np.angle(v1[0])-np.angle(v2[0]))*180/pi)
    #E1.parameters.global_phase(verbose=True)
    #(J*E1).parameters.global_phase(verbose=True)
    #E2.remove_global_phase()
    #(J * E2).parameters.global_phase(verbose=True)
    #J.parameters.retardance(verbose=True)
    #print(np.angle(j)[:,0,0]-np.angle(j)[:,0,1])
    #v1, v2, E1, E2 = Jqwp.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()

    #Jhi.remove_global_phase()
    #v1, v2, E1, E2 = Jhi.parameters.eig(as_objects=True)
    #E1.draw_ellipse()
    #E2.draw_ellipse()
    #plt.show()
    #print(Jhi.parameters)
    #J.remove_global_phase()
    #print(J[11].parameters)
    Jqwp=jones_matrix.create_Jones_matrices('qwp_ideal')
    Jqwp.quarter_waveplate(azimuth=pi/4)
    Jin_l = jones_vector.create_Jones_vectors(name='Jin_l')
    Jin_l.linear_light()
    J_out = Jqwp*Jqwp*Jqwp*Jqwp*Jin_l
    #J_out.draw_ellipse()
    #plt.show()

    Jin_c = jones_vector.create_Jones_vectors(name='Jin_c')
    Jin_c.circular_light(kind='l')
    #Jin_c.draw_ellipse()
    #plt.show()
    #Jin.draw_ellipse()
    Jout_l = J * Jin_l
    Jout_c = J * Jin_c
    #Jout = Jhi * Jin
    #Jout[::3].draw_ellipse()
    #Jout.normalize()
    #print(Jout.parameters.delay())
    #print(Jout.parameters)
    #plt.show()
    Jout_l.draw_ellipse()
    plt.show()
    Jout_c.draw_ellipse()
    plt.show()
