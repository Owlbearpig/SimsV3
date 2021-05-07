from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, array
import pandas
from functions import material_values, form_birefringence


materials = ['HIPS_MUT_1_1', 'HIPS_MUT_1_2', 'HIPS_MUT_1_3', 'HIPS_MUT_2_1', 'HIPS_MUT_2_2', 'HIPS_MUT_2_3']
for mi, material in enumerate(materials):
    result_HIPS_David = {
            'name': '',
            'comments': '',
            'x': '',
            'bf': 'form',
            'mat_name': (material, '')
    }

    eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_HIPS_David, return_vals=True)
    stripes = np.array([628, 517.1])
    n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

    #plt.plot(f.flatten()/10**9, eps_mat1.flatten(), label=material)

    plt.plot(f.flatten(), n_s.flatten()-n_p.flatten(), label=f'bf rytov {material}')

const_eps = 2.4
eps_mat1, eps_mat2 = const_eps*np.ones_like(f), 1*np.ones_like(f)
stripes = np.array([628, 517.1])
n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

plt.plot(f.flatten(), n_s.flatten()-n_p.flatten(), label=f'eps={const_eps}')

plt.legend()
plt.show()
