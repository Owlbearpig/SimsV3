from functions import material_values, form_birefringence
import numpy as np
from numpy import array, pi, sqrt, cos, sin
from results import result_GHz
import matplotlib.pyplot as plt

angles_ghz = np.deg2rad(array([99.66, 141.24, 162.78, 168.14])) #+ 1*np.random.random(4) # 2*np.ones(4)#
d_ghz = array([6659.3, 3766.7, 9139.0, 7598.8]) #+ 1000*np.random.random(4)
stripes_ghz = np.array([628, 517.1])
x_ghz = np.concatenate((angles_ghz, d_ghz, stripes_ghz))



eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
stripes = x_ghz[-2], x_ghz[-1]
n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

def retard(d, phi, theta):
    frac = (n_s**2*cos(phi)**2+n_p**2*sin(phi)**2)/n_p**2
    ret = (2*pi/wls)*d*(sqrt(n_s**2-frac*sin(theta)**2) - sqrt(n_p**2-sin(theta)**2))
    return ret

theta = 1*pi/180
ret_sum = 0
for d, azi in zip(d_ghz, angles_ghz):
    ret_sum += retard(d, azi, theta)

plt.plot(ret_sum)
plt.show()