import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from results import stripes_ghz, result_GHz, d_ghz, angles_ghz
from functions import material_values, form_birefringence, get_einsum
from scipy.constants import c as c0
from numpy import cos, sin, exp
from py_pol import jones_matrix
from py_pol import jones_vector
from scipy.optimize import curve_fit

um = 10**6
THz = 10**12

delta_measured = np.load('delta.npy')
f_measured = np.load('f.npy')
m, n = len(f_measured), 4
einsum_str, einsum_path = get_einsum(m, n)
wls = ((c0/f_measured)*um).reshape((m, 1))

n_s_range = np.linspace(1.0, 1.45, 500)
n_p_range = np.linspace(1.0, 1.45, 500)

cluster_cnt = 6
cluster_centers = np.empty(shape=(0, cluster_cnt))
f = np.array([])
for idx in range(len(f_measured)):
    if idx%50 != 0:
        continue
    print(idx)
    f = np.append(f, f_measured[idx])
    continue
    delta = np.load(f'{idx}.npy')
    #delta = delta.transpose()
    delta = delta #% pi
    for i in range(500):
        for j in range(500):
            if j > i:
                delta[i, j] = delta[i, j]# + pi

    cond = np.logical_or(delta < 0, delta >= 2*pi)
    delta[cond] = delta[cond] % (2 * np.pi)

    print(f'measured: {delta_measured[idx]}')
    minima = np.where(np.abs(delta - delta_measured[idx]) < 0.013)

    for i, j in zip(minima[0], minima[1]):
        delta[i, j] = 20

    bfs = np.array([])
    for i, j in zip(minima[0], minima[1]):
        if idx < 0:
            print(n_s_range[i], n_p_range[j])
        bfs = np.append(bfs, n_p_range[j] - n_s_range[i])

    if idx == 0:
        plt.imshow(delta, cmap='gray', extent=[1, 1.45, 1.45, 1])
        plt.xlabel('n_p')
        plt.ylabel('n_s')
        plt.colorbar()
        plt.show()

    kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(bfs.reshape(-1,1))
    cluster_centers = np.append(cluster_centers, kmeans.cluster_centers_.flatten().reshape((-1, cluster_cnt)), axis=0)

    print(kmeans.cluster_centers_)
    if idx == 0:
        cluster_center = kmeans.cluster_centers_.flatten()
        print(cluster_center[np.argmin(np.abs(np.abs(cluster_center) - 0.1))])
        plt.scatter(range(len(bfs)), bfs)
        plt.show()



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

def func(phi, a,b,delta):
    phi = phi
    return np.abs(np.cos(phi))*np.sqrt((a*np.cos(phi)+b*np.sin(phi)*np.cos(delta))**2
                               +(b*np.sin(phi)*np.sin(delta))**2)
"""


selected_bf = np.array([])
for cluster_center in cluster_centers:
    chosen_index = np.argmin(np.abs(np.abs(cluster_center) - 0.1))
    selected_bf = np.append(selected_bf, cluster_center[chosen_index])

bf_interp = np.interp(f_measured, f, np.abs(selected_bf))

#np.save('bf_interp_phi0.npy', bf_interp)

plt.plot(f, selected_bf, label='evaluated bf')
plt.plot(f_measured, bf_interp, label='interpolated bf')
plt.legend()
plt.show()

n_s_brute = np.ones_like(bf_interp)*1.0
n_p_brute = np.ones_like(bf_interp)*1.0
n_p_brute += bf_interp

plt.plot(f_measured, n_s_brute, label='n_s_brute')
plt.plot(f_measured, n_p_brute, label='n_p_brute')
plt.legend()
plt.show()
"""

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=pi/180)

def calc_delta(n_s, n_p):
    j = j_stack(n_s, n_p)

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])
    J.rotate(angle=pi/180)

    J_out = J * Jin_l

    delta = np.array(J_out.parameters.delay())

    return delta

def calc_delta_measlike(n_s, n_p):
    j = j_stack(n_s, n_p)
    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_A = jones_matrix.create_Jones_matrices('A')
    J_A.diattenuator_linear(p1=1, p2=0, azimuth=0 * pi / 180)

    angles = np.arange(0, 370, 10)

    phi = np.array([])
    s21 = np.array([])
    for angle in angles:
        J_P = jones_matrix.create_Jones_matrices('P')
        J_P.diattenuator_linear(p1=1, p2=0, azimuth=(angle+9.64) * pi / 180)

        phi = np.append(phi, angle)
        J_out = J_A * J_P * J * Jin_l
        intensity = J_out.parameters.intensity()
        s21 = np.append(s21, np.sqrt(intensity))

    phi = np.deg2rad(phi)
    popt, pcov = curve_fit(func, phi, s21)

    # p1, p2, a, b, delta
    a = popt[0]
    b = popt[1]
    delta = popt[2]

    return delta

n_s_range = np.linspace(1.0, 1.6, 500)
n_p_range = np.linspace(1.0, 1.6, 500)

f = np.array([])
n_s_arr, n_p_arr = np.array([]), np.array([])
for idx in range(m):
    image = np.zeros((len(n_s_range), len(n_p_range)))
    if idx%50 != 0:
        continue
    if idx != 0:
        continue
    print(idx)
    f = np.append(f, f_measured[idx])
    delta_1pnt_pypol, delta_1pnt_measlike = np.array([]), np.array([])

    for i, n_s in enumerate(n_s_range):
        print(i)
        n_p = n_p_range[0]

        delta_1pnt_pypol = np.append(delta_1pnt_pypol, calc_delta(n_s, n_p))
        delta_1pnt_measlike = np.append(delta_1pnt_measlike, calc_delta_measlike(n_s, n_p))


    print(idx)
    print(delta_measured[idx])
    plt.plot(n_s_range, delta_1pnt_pypol, label='pypol')
    plt.plot(n_s_range, delta_1pnt_measlike, label='measlike')
    plt.show()

    #np.save(str(idx)+'1.6x1.6', image)

