import numpy as np
from numpy import power, outer, sqrt, exp, sin, cos, conj, dot, einsum
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

np.random.seed(1234)

path = r"4Eck_D=2042.csv"
um = 1#10 ** -6
THz = 10**-12

sqrt2 = sqrt(2)
pi = np.pi


def read_data_file(file_path):
    data_filepath = Path(PureWindowsPath(file_path))
    df = pandas.read_csv(data_filepath)

    freq_dict_key = [key for key in df.keys() if "freq" in key][0]
    eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]

    frequencies = np.array(df[freq_dict_key])[[100, 101]]#[0:140:3]

    eps_mat_r = np.array(df[eps_mat_r_key])[[100, 101]]#[0:140:3]

    return eps_mat_r, frequencies


eps_mat1, frequencies = read_data_file(path)


eps_mat2 = np.ones_like(eps_mat1)
wls = c0/frequencies
wls *= 10**6

m = len(wls)
n = m

def dn(a, b):
    eps1_p = (outer(eps_mat1, a)+outer(eps_mat2, b)) / (a+b).T
    eps1_s = (outer(eps_mat1*eps_mat2, (a+b)))*(1/(outer(eps_mat1, b)+outer(eps_mat2, a)))

    c = (1 / 3) * power(outer(1 / wls, (a * b * pi) / (a + b)), 2)

    eps2_s = eps1_s + ((c * power(eps1_s, 3) * eps1_p).T * power((1 / eps_mat1 - 1 / eps_mat2), 2)).T
    eps2_p = eps1_p + (c.T * power((eps_mat1 - eps_mat2), 2)).T

    return sqrt(eps2_p)-sqrt(eps2_s)


def erf(x):
    a, b, z = x[0:n], x[n:2*n], x[2*n:3*n]

    bf = dn(a, b)

    return sum((0.5*wls - dot(bf, z) % wls)**2)


a = 100*np.random.random(n)
b = 100*np.random.random(n)
z = 100*np.ones(n)

print(f'freq_cnt): {m}, wp_cnt: {n}, freq_cnt*wp_cnt: {m*n}, var_cnt: {len(a)+len(b)+len(z)}')
print('a (um):', a)
print('b (um):', b)
print('z (um):', z)
print('wavelengths (um)', wls)
print('frequencies (THz)', frequencies*THz, '\n')

x0 = np.concatenate((a, b, z))
print(erf(x0))

#ret = minimize(erf, x0)
#print(ret)

#print(max(a+b), min(0.25*wls))
#local_min_kwargs = {'tol': 10**-9}
#ret = basinhopping(erf, z, niter=0, minimizer_kwargs=local_min_kwargs)

x = solve(dn(a, b) % wls, .5*wls)
print(x)
