import numpy as np
from functions import form_birefringence, material_values
from results import result_GHz
import matplotlib.pyplot as plt

def eps_from_bf(phi_offset):
    bf_meas = np.load(f'bf_interp_phi{phi_offset}.npy')
    stripes = np.array([628, 517.1])
    # stripes = np.array([750, 450.1])
    _, _, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
    eps_mat2 = np.ones_like(f).reshape(wls.shape)

    possible_mat_eps_arr = np.linspace(2, 3, len(f)).reshape(wls.shape)

    eps_brute_fit = np.array([])
    for idx in range(len(f)):
        if idx % 50 != 0:
            continue
        bf = bf_meas[idx]
        wl = wls[idx]
        print(f'prog: {idx}/{len(f)}')
        min_diff, best_eps = 100, None
        for possible_mat_eps in possible_mat_eps_arr:
            n_s, n_p, k_s, k_p = form_birefringence(stripes, wl, possible_mat_eps, eps_mat2[idx])
            diff = np.abs(bf - np.abs(n_p - n_s))
            if diff < min_diff:
                min_diff = diff
                best_eps = possible_mat_eps
        eps_brute_fit = np.append(eps_brute_fit, best_eps)

    return eps_brute_fit

if __name__ == '__main__':
    bf_meas = np.load('bf_interp_phi0.npy')
    stripes = np.array([628, 517.1])
    #stripes = np.array([750, 450.1])
    _, _, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
    eps_mat2 = np.ones_like(f).reshape(wls.shape)

    possible_mat_eps_arr = np.linspace(2, 3, len(f)).reshape(wls.shape)
    print(bf_meas)
    plt.plot(np.abs(bf_meas))
    plt.show()
    eps_brute_fit = np.array([])
    for idx in range(len(f)):
        if idx % 50 != 0:
            continue
        bf = bf_meas[idx]
        wl = wls[idx]
        print(f'prog: {idx}/{len(f)}')
        min_diff, best_eps = 100, None
        for possible_mat_eps in possible_mat_eps_arr:
            n_s, n_p, k_s, k_p = form_birefringence(stripes, wl, possible_mat_eps, eps_mat2[idx])
            diff = np.abs(bf - np.abs(n_p-n_s))
            if diff < min_diff:
                min_diff = diff
                best_eps = possible_mat_eps
        eps_brute_fit = np.append(eps_brute_fit, best_eps)

    plt.plot(f.flatten(), eps_brute_fit)
    plt.show()

    #np.save('eps_brute_fit_phi9.64.npy', eps_brute_fit)


