import numpy as np
from numpy import power, outer, sqrt, exp, sin, cos, conj, dot, pi, einsum, arctan, array
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
from py_pol_calcs import jones_vector, jones_matrix
import string
import matplotlib.pyplot as plt
import sys

THz = 10**12
m_um = 10**6 # m to um conversion


def opt(n, ret_j=False, x=None):
    d = np.ones(n)*204#*4080/n
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

        if ret_j:
            return j

        q = j[:, 0, 0] / j[:, 1, 0]
        res_int = (1/m)*sum(q.real ** 2 + (q.imag - 1) ** 2)

        return res_int


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

    if ret_j:
        return erf(x)

    #return minimize(erf, x0)
    return basinhopping(erf, x0, niter=200, callback=print_fun, take_step=bounded_step, disp=True, T=1.4)

x5 =\
    array([2.18120575, 2.70390048, 2.27125591, 2.71260592, 3.45363271])
x10 = \
    array([1.08043822, 0.2678766, 0.2797097, 0.97660162, 0.91354882,
           4.94156437, 4.94420145, 4.63796814, 1.53208222, 4.22266831])
x15 = \
    array([4.63083449, 2.21146242, 5.16466218, 5.70099083, 5.57698886,
           3.16877669, 0.67906624, 6.36723395, 5.85967038, 6.10242389,
           5.59340896, -0.37546238, 0.29788886, 6.53909046, 0.111658])
x20 =\
array([4.83958375, 5.46695672, 2.70029202, 1.59720805, 1.38049486,
       0.67858639, 0.47854223, 1.06206799, 4.40208982, 3.92385128,
       0.89218459, 0.38393526, 3.01364689, 3.41504969, 3.57882995,
       4.2264747 , 3.54250995, 0.30855872, 2.85858943, 0.48572504])
x40=\
array([ 1.48526827,  5.68648854,  2.78825831,  2.91651771,  2.81884487,
        0.78918152,  2.07029213,  1.27541674,  5.14464022,  1.5567863 ,
        5.32316252,  4.37550918,  1.28968715,  1.89523375,  6.30634352,
        5.57707961,  2.65804305,  5.82311374,  6.29053   ,  5.73997893,
        3.5926664 ,  0.49653279,  3.54711758,  6.35698866,  6.05193924,
        4.62530237,  6.00679083,  5.0712867 ,  0.06906525, -0.69931257,
        4.03977268,  1.89175715,  0.74023309,  6.26873037,  6.1411231 ,
        1.79606476,  1.1411559 ,  2.52509121,  2.6446789 , -0.0372008 ])

x40_2 =\
array([1.17545914, 4.554194  , 4.07113604, 4.61945512, 5.92630454,
       5.76742732, 3.85703013, 1.01224902, 0.84125755, 3.26299053,
       3.96009646, 2.93249464, 5.9267418 , 4.2577931 , 1.35795942,
       4.11991579, 0.05775351, 5.78405977, 2.7483227 , 0.72560739,
       0.54790961, 1.16589736, 4.69253614, 0.54295811, 0.81419061,
       1.35997666, 3.68956804, 3.04055929, 0.96076034, 2.51693646,
       3.77214371, 5.88433733, 1.55431756, 5.98949997, 5.65430022,
       3.7618517 , 3.30900609, 5.90646438, 1.90639233, 5.73831861])

if __name__ == '__main__':


    f = np.arange(0.2, 2.0, 0.05)*THz

    wls = (c0/f)*m_um
    m = len(wls)
    n = 20

    no = 3.39
    ne = 3.07
    bf = np.ones_like(f)*(no-ne)

    j = opt(20, ret_j=True, x=x20)


    J = jones_matrix.create_Jones_matrices('J_R')
    J.from_matrix(j)
    Jlin = jones_vector.create_Jones_vectors('J_L')
    Jlin.linear_light()

    Jout = J*Jlin

    Jout.draw_ellipse()
    plt.show()

    """
    for _ in range(10):
        res = opt(n=n, res_only=False)
        print(res)
    """

    #np.random.seed(1000)
    """
    for r in range(0, 10):
        res = opt(n=n, res_only=False, x=np.random.random(n))
        print(res)
        print('Break')
    """
    """
    plt.plot(f, res1, label=f'n={40_1}')
    plt.plot(f, res2, label=f'n={40_2}')
    plt.ylim((0.46, 0.52))
    plt.legend()
    plt.show()
    """


