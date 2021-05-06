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

delta_measured = np.load('delta_phi4.5.npy')
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

    delta = np.load(f'{idx}.npy')
    #delta = delta.transpose()
    delta = delta % pi

    minima = np.where(np.abs(delta - delta_measured[idx]) < 0.012)

    for i, j in zip(minima[0], minima[1]):
        delta[i, j] = 20

    bfs = np.array([])
    for i, j in zip(minima[0], minima[1]):
        if idx == 0:
            print(n_s_range[i], n_p_range[j])
        bfs = np.append(bfs, n_p_range[j] - n_s_range[i])

    if idx == 1250:
        print(delta_measured[idx])
        plt.imshow(delta, cmap='gray', extent=[1, 1.45, 1.45, 1])
        plt.xlabel('n_p')
        plt.ylabel('n_s')
        plt.colorbar()
        plt.show()

    kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(bfs.reshape(-1,1))
    cluster_centers = np.append(cluster_centers, kmeans.cluster_centers_.flatten().reshape((-1, cluster_cnt)), axis=0)

    print(kmeans.cluster_centers_)
    if idx == 1250:
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

Jin_l = jones_vector.create_Jones_vectors('Jin_l')
Jin_l.linear_light(azimuth=0*pi/180)

selected_bf = np.array([])
for cluster_center in cluster_centers:
    chosen_index = np.argmin(np.abs(np.abs(cluster_center) - 0.1))
    selected_bf = np.append(selected_bf, np.abs(cluster_center[chosen_index]))

bf_interp = np.interp(f_measured, f, selected_bf)

plt.plot(f, selected_bf, label='evaluated bf')
plt.plot(f_measured, bf_interp, label='interpolated bf')
plt.legend()
plt.show()

n_s_brute = np.ones_like(bf_interp)*1.25
n_p_brute = np.ones_like(bf_interp)*1.25
n_p_brute += bf_interp

plt.plot(f_measured, n_s_brute, label='n_s_brute')
plt.plot(f_measured, n_p_brute, label='n_p_brute')
plt.legend()
plt.show()

idx = 0
def calc_delta(n_s, n_p):
    j = j_stack(n_s, n_p)

    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])

    J_out = J * Jin_l

    delta = np.array(J_out.parameters.delay())

    return delta

angles = np.arange(0,370,10)
plt.figure()
f_cut = np.array([])
delta = np.array([])
a, b = np.array([]), np.array([])
for idx in range(len(f_measured)):
    if idx%50 != 0:
        continue
    print(idx)
    #phi_offset = 4.725 + (14.84-4.725)*idx/1400

    f_cut = np.append(f_cut, f_measured[idx])
    n_s, n_p = n_s_brute[idx], n_p_brute[idx]
    j = j_stack(n_s, n_p)
    J = jones_matrix.create_Jones_matrices()
    J.from_matrix(j[idx])
    J.rotate(angle=0 * pi / 180)
    p2 = 0  # p2_100deg[idx]
    p1 = 1 - p2  # p1_190deg[idx]  # 0.5+0.5*(idx/len(f))
    J_A = jones_matrix.create_Jones_matrices('A')
    J_A.diattenuator_linear(p1=p1, p2=p2, azimuth=0 * pi / 180)

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
    popt, pcov = curve_fit(func, phi, s21)

    print(*popt)
    #p1, p2, a, b, delta
    a = np.append(a, popt[0])
    b = np.append(b, popt[1])
    delta = np.append(delta, (popt[2]))

    continue
    plt.polar(phi, s21, label='messung')
    plt.plot(phi, func(phi, *popt), label='fit')
    plt.legend()
    plt.show()

plt.plot(f_cut/10**9, delta, '.-',label = 'Messung')
plt.plot(f_measured/10**9, delta_measured, '.-',label = 'Reale Messung')
#plt.plot(f_cut/10**9, f_cut*0+0.5*1.03, 'k--',label='+3%')
#plt.plot(f_cut/10**9, f_cut*0+0.5*0.97, 'k--',label='-3%')
plt.grid(True)
plt.xlabel('$f$ in GHz')
#plt.ylabel(r"$\frac{\delta}{\pi}$")
plt.ylabel(r"$\delta$")
#plt.ylim([0, 1])
plt.xlim([75,110])
plt.legend()
plt.show()


for i in range(cluster_cnt):
    plt.scatter(f/10**9, np.abs(cluster_centers[:, i]), label='cluster_' + str(i))

plt.plot(f_measured.flatten()/10**9, bf_interp, label='possible(bruteforce) bf')

plt.ylabel('possible birefringence')
plt.xlabel('frequency (GHz)')

eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
stripes = stripes_ghz[-2], stripes_ghz[-1]
n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

plt.plot(f.flatten()/10**9, n_p-n_s, label='FormBF(Rytov)')

plt.legend()
plt.show()







