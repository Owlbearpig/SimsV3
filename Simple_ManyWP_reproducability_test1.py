import numpy as np
from numpy import power, outer, sqrt, exp, sin, cos, conj, dot, pi, einsum, arctan
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.linalg import solve
import string
import matplotlib.pyplot as plt

THz = 10**12
m_um = 10**6 # m to um conversion

#np.random.seed(1000)

f = np.arange(0.2, 2.0, 0.05)*THz

wls = (c0/f)*m_um
m = len(wls)

no = 3.39
ne = 3.07
bf = no-ne

n = 20
d = np.ones(n)*204
delta = pi * bf * outer(1 / wls, d)  # delta/2

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

def erf(angles, this=False):
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

    if this:
        return delta_equiv / pi

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

ret = basinhopping(erf, x0, niter=100, callback=print_fun, take_step=bounded_step, disp=True)
print(ret)

#ret = basinhopping(erf, x0, niter=100, callback=print_fun, take_step=bounded_step, disp=True, minimizer_kwargs=minimizer_kwargs)
#ret = minimize(erf, x0, method='cg', options={'disp': True, 'maxiter': 300*n})
#ret = minimize(erf, x0, options={'disp': True, 'maxiter': 300*n})

print(erf(x0))

"""
plt.plot(f, ret)
plt.ylim((0.46, 0.52))
plt.show()
"""

