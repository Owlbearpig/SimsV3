import numpy as np
from numpy import (power, outer, sqrt, exp, sin, cos, conj, dot, pi,
                   einsum, arctan, array, arccos, conjugate, flip, angle, tan, arctan2)
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
from functions import wp_cnt, setup

rad = 180 / np.pi
THz = 10**12
m_um = 10**6 # m to um conversion


def optimize(settings):
    n = wp_cnt(settings)
    bounds = list(zip([0] * n, [2 * pi] * n)) + list(zip([0] * n, [10 ** 4] * n))

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

    minimizer_kwargs = {}

    erf = setup(settings)

    # return minimize(erf, settings['x0'])
    return basinhopping(erf, settings['x'], niter=200, callback=print, take_step=bounded_step, disp=True, T=25)

if __name__ == '__main__':
    # run in terminal and output to .txt file (python optimization.py > somename.txt)
    n = 6
    resolution = 14
    f_range = 0.25 * THz, 1.5 * THz
    bf = 'intrinsic'
    mat_name = ('ceramic_fast', 'ceramic_slow')

    settings = {'resolution': resolution, 'f_range': f_range, 'bf': bf, 'mat_name': mat_name}

    best, best_res = np.inf, None
    for _ in range(10):
        settings['x'] = np.concatenate((np.random.random(n) * 2 * pi, np.random.random(n) * 10 ** 4))
        ret = optimize(settings)
        print(ret)
        print(ret, _)
        if ret.fun < best:
            best = ret.fun
            best_res = ret

    print(best_res)

