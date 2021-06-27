import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.constants import c as c0
import pandas
from scipy import optimize


def ref_ind_fit(wl, speed, return_all=False):
    THz = 10**12
    m_um = 10**6
    f_min, f_max = 0.2*THz, 2.5*THz

    resolution = 1
    if speed == 'slow':
        mat_path = Path('material_data/quartz_m_slow2.csv')
    else:
        mat_path = Path('material_data/quartz_m_fast2.csv')

    df = pandas.read_csv(mat_path)

    freq_dict_key = [key for key in df.keys() if "freq" in key][0]
    eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
    eps_mat_i_key = [key for key in df.keys() if "epsilon_i" in key][0]
    ref_ind_key = [key for key in df.keys() if "ref_ind" in key][0]

    f = np.array(df[freq_dict_key])

    data_slice = np.where((f > f_min) &
                          (f < f_max))
    data_slice = data_slice[0][::resolution]

    eps_mat_r = np.array(df[eps_mat_r_key])[data_slice]
    eps_mat_i = np.array(df[eps_mat_i_key])[data_slice]
    ref_ind_data = np.array(df[ref_ind_key])[data_slice]
    #ref_ind += np.random.random(ref_ind.shape)*0.001
    f = f[data_slice]

    wls = (c0/f)*m_um

    #eps_mat1 = (eps_mat_r + eps_mat_i * 1j)


    class Parameter:
        def __init__(self, value):
                self.value = value

        def set(self, value):
                self.value = value

        def __call__(self):
                return self.value

    def fit(function, parameters, y, x = None):
        def f(params):
            i = 0
            for p in parameters:
                p.set(params[i])
                i += 1
            return y - function(x)

        if x is None: x = np.arange(y.shape[0])
        p = [param() for param in parameters]
        return optimize.leastsq(f, p, maxfev=10**6)

    B1, B2, B3 = Parameter(1.1), Parameter(0.001), Parameter(0.01)
    C1, C2, C3 = Parameter(1.1), Parameter(0.001), Parameter(0.01)

    #ref_ind = np.array([1,2,3,4,5,6,7,8,9,10])
    #wls = np.arange(ref_ind.shape[0])

    def sellmeier(x):
        s1 = (B1() * x ** 2) / (x ** 2 - C1())
        s2 = (B2() * x ** 2) / (x ** 2 - C2())
        s3 = (B3() * x ** 2) / (x ** 2 - C3())
        return 1+s1+s2+s3
        #return B1()+B2()*x**1+B3()*x**2+C1()**3+C2()**4+C3()**5

    data = ref_ind_data**2

    #from dataexport import save
    fit(sellmeier, [B1, B2, B3, C1, C2, C3], data, x=wls)
    print(f'(B1, B2, B3, C1, C2, C3)={[param() for param in [B1, B2, B3, C1, C2, C3]]}')
    #print(np.sqrt(sellmeier(wls)))
    #plt.plot(wls, np.sqrt(data), '.')
    #plt.xlim((max(wls)*1.1, min(wls)*0.8))
    #plt.plot(wls, np.sqrt(sellmeier(wls)), '-')
    ref_ind_fit = np.sqrt(sellmeier(wls))

    #save({'freq': f, 'ref_ind': ref_ind_fit, 'epsilon_r': ref_ind_fit**2, 'epsilon_i': np.zeros_like(ref_ind_fit)},
    #     name='sellmeier_quartz_fast')

    #plt.show()
    if return_all:
        return np.sqrt(sellmeier(wl)), ref_ind_data
    return np.sqrt(sellmeier(wl))
