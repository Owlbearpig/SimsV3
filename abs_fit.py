import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.constants import c as c0
import pandas
from scipy import optimize

THz = 10**12
m_um = 10**6
f_min, f_max = 0.2*THz, 2.5*THz

resolution = 1

speed = 'fast' # 'slow' # 'fast'

mat_path = Path('material_data/quartz_m_slow2.csv')

df = pandas.read_csv(mat_path)

freq_dict_key = [key for key in df.keys() if "freq" in key][0]


f = np.array(df[freq_dict_key])

data_slice = np.where((f > f_min) &
                      (f < f_max))
data_slice = data_slice[0][::resolution]

#ref_ind += np.random.random(ref_ind.shape)*0.001
f_full = f[data_slice]

resolution = 1
matpath_slow = Path('material_data/abs_slow_grisch1990.csv')
matpath_fast = Path('material_data/abs_fast_grisch1990.csv')

if speed == 'fast':
    df = pandas.read_csv(matpath_fast)
else:
    df = pandas.read_csv(matpath_slow)

freq_dict_key = [key for key in df.keys() if "freq" in key][0]
alpha_key = [key for key in df.keys() if "alpha" in key][0]

f = np.array(df[freq_dict_key])

data_slice = np.where((f > f_min) &
                      (f < f_max))
data_slice = data_slice[0][::resolution]

alpha_data = np.array(df[alpha_key])[data_slice]

f = f[data_slice]


class Parameter:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def __call__(self):
        return self.value


def fit(function, parameters, y, x=None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = np.arange(y.shape[0])
    p = [param() for param in parameters]
    return optimize.leastsq(f, p, maxfev=10 ** 6)


B1, B2, B3 = Parameter(0.1), Parameter(0.01), Parameter(0.01)
C1, C2, C3 = Parameter(1.1), Parameter(0.001), Parameter(0.01)

def func(x):
    s1 = np.sqrt(x)

    return s1
    # return B1()+B2()*x**1+B3()*x**2+C1()**3+C2()**4+C3()**5

# from dataexport import save
#fit(func, [B1, B2, B3, C1, C2, C3], alpha, x=f)
c = np.polynomial.chebyshev.chebfit(f, np.sqrt(alpha_data), deg=1)
#c = np.polynomial.polynomial.Polynomial.fit(f/10**12, np.sqrt(alpha_data), deg=2)
print(c)
def poly(x):
    s = 0
    for i, coef in enumerate(c):
        print(coef)
        s += coef*x**i
    return s

alpha = poly(f_full)**2

wls = (c0/f_full)*m_um
from sellmeier import ref_ind_fit
ref_ind, ref_ind_data = ref_ind_fit(wls, speed, return_all=True)
kappa = alpha*c0/(4*pi*f_full)
eps_i = 2*ref_ind*kappa*100 # units of alpha are 1/cm

#plt.plot(f, func(f), label='func fit')
plt.plot(f_full, alpha, label='fit')
plt.scatter(f, alpha_data, label='data')

from generate_plotdata import export_csv
"""
export_csv({'freq': f_full, 'wls': wls,
            'alpha_fit': alpha, 'ref_ind_fit': ref_ind, },
           f'{speed}_grisch1990_fit.csv')
"""
plt.legend()
plt.show()

