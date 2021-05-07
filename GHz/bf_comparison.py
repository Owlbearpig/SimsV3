from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, array
import pandas
from functions import material_values, form_birefringence

def load_material_data(path):
    df = pandas.read_csv(path)

    freq_dict_key = [key for key in df.keys() if 'freq' in key][0]
    try:
        ref_ind_key = [key for key in df.keys() if 'ref_ind' in key][0]
        ref_ind = np.array(df[ref_ind_key])
    except IndexError:
        epsilon_r_key = [key for key in df.keys() if 'epsilon_r' in key][0]
        ref_ind = np.sqrt(np.array(df[epsilon_r_key]))
    # dn_key = [key for key in df.keys() if "delta_N" in key][0]

    frequencies = np.array(df[freq_dict_key])

    # dn = np.array(df[dn_key])

    return frequencies, ref_ind

angles_ghz = np.deg2rad(array([99.66, 141.24, 162.78, 168.14]))# + (5*np.random.random(4) - 2.5*np.ones(4)) # 2*np.ones(4)#
d_ghz = array([6659.3, 3766.7, 9139.0, 7598.8])#*(0.85+np.random.random(4)*0.1)
stripes_ghz = np.array([628, 517.1]) #+ (50*np.random.random(2) + 100*np.ones(2))
x_ghz = np.concatenate((angles_ghz, d_ghz, stripes_ghz))

result_HIPS_HHI = {
        'name': 'c_random',
        'comments': '',
        'x': x_ghz,
        'bf': 'form',
        'mat_name': ('HIPS_HHI', '')
}

baseHHI = Path("ResultTeralyzerHHI")
resultfilesHHI = [os.path.join(root, name)
               for root, dirs, files in os.walk(baseHHI)
               for name in files
               if name.endswith('.csv')]

for result in resultfilesHHI:
    frequencies, ref_ind = load_material_data(result)
    plt.plot(frequencies, ref_ind, label=result)

eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_HIPS_HHI, return_vals=True)
stripes = stripes_ghz[-2], stripes_ghz[-1]
n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

plt.plot(f.flatten(), n_s.flatten(), label='n_s rytov')
plt.plot(f.flatten(), n_p.flatten(), label='n_p rytov')
plt.legend()
plt.show()
