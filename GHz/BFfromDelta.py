import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi, exp
from scipy.constants import c as c0
from results import d_ghz, angles_ghz, result_GHz, stripes_ghz
from functions import get_einsum, setup, material_values, form_birefringence
from py_pol import jones_matrix
from py_pol import jones_vector
from scipy.optimize import curve_fit
from scipy.optimize import minimize

um = 10**6
THz = 10**12


f_measured = np.load('f.npy')
delta_measured = np.load('delta.npy')
a_measured = np.load('a.npy')
b_measured = np.load('b.npy')

plt.plot(f_measured/10**9, delta_measured/np.pi, '.-',label = 'Messung')
plt.plot(f_measured/10**9, f_measured*0+0.5*1.03, 'k--',label='+3%')
plt.plot(f_measured/10**9, f_measured*0+0.5*0.97, 'k--',label='-3%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.xlim([75,110])
plt.ylim([0.0,1.0])
plt.close()
#plt.show()


def func(phi, a,b,delta):
    phi = phi
    return np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)

m, n = len(f_measured), 4
einsum_str, einsum_path = get_einsum(m, n)
f_measured = f_measured.reshape((m, 1))
wls = ((c0/f_measured)*um)

def j_stack(n_s, n_p):
    j = np.zeros((m, n, 2, 2), dtype=complex)

    phi_s, phi_p = (2 * n_s * pi / wls) * d_ghz.T, (2 * n_p * pi / wls) * d_ghz.T

    x, y = 1j * phi_s, 1j * phi_p
    angles = np.tile(angles_ghz, (m, 1))

    j[:, :, 0, 0] = exp(y) * sin(angles) ** 2 + exp(x) * cos(angles) ** 2
    j[:, :, 0, 1] = 0.5 * sin(2 * angles) * (exp(x) - exp(y))
    j[:, :, 1, 0] = j[:, :, 0, 1]
    j[:, :, 1, 1] = exp(x) * sin(angles) ** 2 + exp(y) * cos(angles) ** 2

    np.einsum(einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=einsum_path[0])

    j = j[:, 0]

    return j


Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=0*pi/180)

angles = np.arange(0,370,10)

def calc_delta(n_s, n_p):
    j = j_stack(n_s, n_p)

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_out = J * Jin_l

    delta = np.array(J_out.parameters.delay())

    return delta

def func2(n_s, n_p):
    delta = calc_delta(n_s, n_p)

    return (delta_measured[idx] - delta)**2

n_s_range = np.linspace(1.0, 1.45, 500)#np.linspace(1.256, 1.268, 50)
n_p_range = np.linspace(1.0, 1.45, 500)#np.linspace(1.334, 1.345, 50)

image = np.zeros((len(n_s_range), len(n_p_range)))

f = np.array([])
n_s_arr, n_p_arr = np.array([]), np.array([])
for idx in range(m):
    if idx%50 != 0:
        continue
    if idx == 0:
        continue
    print(idx)
    f = np.append(f, f_measured[idx])

    best_val = 1000
    best_res = None
    for i, n_s in enumerate(n_s_range):
        print(i)
        for j, n_p in enumerate(n_p_range):
            delta = calc_delta(n_s, n_p)
            diff = (delta_measured[idx] - delta)**2
            if diff < best_val:
                best_val = diff
                best_res = n_s, n_p
            image[i,j] = delta
    print(best_val, best_res)
    print(delta_measured[idx])
    np.save(str(idx), image)

    continue
    n_s_arr = np.append(n_s_arr, best_res[0])
    n_p_arr = np.append(n_p_arr, best_res[1])


print(best_val, best_res)
plt.imshow(image)
plt.show()
exit()
#np.save('n_s_arr', n_s_arr)
#np.save('n_p_arr', n_p_arr)

n_s_arr = np.load('n_s_arr.npy')
n_p_arr = np.load('n_p_arr.npy')

plt.plot(f/10**9, n_s_arr, label='n_s_arr')
plt.plot(f/10**9, n_p_arr, label='n_p_arr')
plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$n$")
plt.xlim([75,110])
plt.legend()
plt.show()

delta_calc = np.array([])
for n_s, n_p in zip(n_s_arr, n_p_arr):
    delta_calc = np.append(delta_calc, calc_delta(n_s, n_p))

plt.plot(f_measured/10**9, delta_measured/pi, label='delta_measured')
plt.plot(f/10**9, delta_calc/pi, label='delta_calc')

eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
stripes = stripes_ghz[-2], stripes_ghz[-1]
n_s_arr, n_p_arr, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)
delta_calc = np.array([])
for n_s, n_p in zip(n_s_arr, n_p_arr):
    delta_calc = np.append(delta_calc, calc_delta(n_s, n_p))


plt.plot(f.flatten()/10**9, delta_calc/pi, label='formbirefringence values')

plt.grid(True)
plt.xlabel('$f$ in GHz')
plt.ylabel(r"$\frac{delta}{\pi}$")
plt.xlim([75,110])
#plt.ylim([0,1])
plt.legend()
plt.show()

"""
plt.figure()
f_cut = np.array([])
delta = np.array([])
a, b = np.array([]), np.array([])
for idx in range(len(f)):
    if idx%50 != 0:
        continue
    print(idx)

    f_cut = np.append(f_cut, f[idx])

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_A = jones_matrix.create_Jones_matrices('A')
    J_A.diattenuator_linear(p1=1, p2=0, azimuth=0 * pi / 180)

    phi = np.array([])
    s21 = np.array([])
    for angle in angles:
        J_P = jones_matrix.create_Jones_matrices('P')
        J_P.diattenuator_linear(p1=1, p2=0, azimuth=angle * pi / 180)

        phi = np.append(phi, angle)
        J_out = J_A * J_P * J * Jin_l
        intensity = J_out.parameters.intensity()
        s21 = np.append(s21, np.sqrt(intensity))

    phi = np.deg2rad(phi)
    popt, pcov = curve_fit(func, phi, s21, p0=[0.7, 0.7, pi/2])

    print(*popt)
    a = np.append(a, popt[0])
    b = np.append(b, popt[1])
    delta = np.append(delta, np.abs(popt[2]))

    plt.polar(phi, s21, label='messung')
    plt.plot(phi, func(phi, *popt), label='fit')
    plt.legend()
    plt.show()
"""