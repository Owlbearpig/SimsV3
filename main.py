import numpy as np
from numpy import power, outer, sqrt, exp, sin, cos, conj
import pandas
from pathlib import Path, PureWindowsPath
import scipy
from scipy.constants import c as c0
from scipy.optimize import basinhopping
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution
import string

path = r"E:\MEGA\AG\BFWaveplates\Data\fused silica parameters\4Eck_D=2042.csv"
um = 1#10 ** -6
THz = 10**-12

sqrt2 = sqrt(2)
pi = np.pi


def read_data_file(file_path):
    data_filepath = Path(PureWindowsPath(file_path))
    df = pandas.read_csv(data_filepath)

    freq_dict_key = [key for key in df.keys() if "freq" in key][0]
    eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]

    frequencies = np.array(df[freq_dict_key])[0:10:2]#[0:140:3]

    eps_mat_r = np.array(df[eps_mat_r_key])[0:10:2]#[0:140:3]

    return eps_mat_r, frequencies


eps_mat1, frequencies = read_data_file(path)


eps_mat2 = np.ones_like(eps_mat1)
wls = c0/frequencies
wls *= 10**6

n = m = len(wls)


def dn(a, b):
    eps1_p = (outer(eps_mat1, a)+outer(eps_mat2, b)) / (a+b).T
    eps1_s = (outer(eps_mat1*eps_mat2, (a+b)))*(1/(outer(eps_mat1, b)+outer(eps_mat2, a)))

    return eps1_p, eps1_s


v = 2*pi*np.random.random(n)
a = 10*np.random.random(n)
b = 10*np.random.random(n)
z = 10*np.random.random(n)

print(m, n, m*n, len(a)+len(b)+len(v)+len(z))
print('wavelengths', wls)
print('frequencies', frequencies*THz)

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


def erf(x):
    v, a, b, z = x[0:n], x[n:2*n]*um, x[2*n:3*n]*um, x[3*n:4*n]*um
    eps1_p, eps1_s = dn(a, b)
    bf = eps1_p - eps1_s

    delta = 2*pi*(bf.T*(1/wls)).T*z

    j = np.zeros((m, n, 2, 2), dtype=complex)
    for i in range(0, 2):
        for k in range(0, 2):
            j[:, :, i, k] = sin(delta/2)
    for k in range(0,2):
        j[:, :, k, k] = 1j*j[:, :, k, k]*cos(2*v)
    j[:, :, 0, 1] = 1j*j[:, :, 0, 1] * sin(2 * v)
    j[:, :, 1, 0] = 1j * j[:, :, 1, 0] * sin(2 * v)
    j[:, :, 0, 0] += cos(delta/2)
    j[:, :, 1, 1] = -1*j[:, :, 1, 1] + cos(delta / 2)

    np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

    j = j[:, 0]

    res_int = sum((1 - j[:, 1, 0] * conj(j[:, 1, 0])) + (j[:, 0, 0] * conj(j[:, 0, 0])))

    return res_int.real/(2*m)


def local_min_method(fun, x0, args=(), **unknownoptions):
    return OptimizeResult(x=x0, fun=fun(x0, *args), success=1)

x0 = np.concatenate((v,a,b,z))
for t in range(100):
    continue

print(erf(x0))


local_min_kwargs = {'tol': 10**-14} #None#{'method': local_min_method}


def print_fun(x, f, accepted):
    print(f, accepted)


#ret = basinhopping(erf, x0, niter=0, callback=print_fun, minimizer_kwargs=local_min_kwargs)
bounds = []
bounds.extend([(0,2*pi)]*len(v))
bounds.extend([(0,10)]*len(a))
bounds.extend([(0,10)]*len(b))
bounds.extend([(0,10)]*len(z))

ret = differential_evolution(erf, bounds, tol=10**-8)

print(ret)
