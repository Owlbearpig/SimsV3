import os
from pathlib import Path
import numpy as np
from scipy.constants import c as c0
import pandas
import matplotlib.pyplot as plt
from functions import material_values
from functions import form_birefringence

um = 10**6
THz = 10**12

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

angles_ghz = np.deg2rad(np.array([45])) #+ 1*np.random.random(4) # 2*np.ones(4)#
d_ghz = np.array([4000]) #+ 1000*np.random.random(4)
stripes_ghz = np.array([750, 375])
x_sg = np.concatenate((angles_ghz, d_ghz, stripes_ghz))

result_GHz = {
        'name': 'single_grating',
        'comments': '',
        'x': x_sg,
        'bf': 'form',
        'mat_name': ('HIPS_MUT_1_1', '')
}

def fake_fb(f):
    m = len(f)
    stripes = stripes_ghz[-2], stripes_ghz[-1]
    eps_mat2 = (np.ones_like(f)).reshape(m, 1)
    eps_mat1 = (np.ones_like(f)*2.149).reshape(m, 1)
    wls = um*(c0/f).reshape(m, 1)
    n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

    return n_s.flatten(), n_p.flatten()

eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
stripes = stripes_ghz[-2], stripes_ghz[-1]
n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

base = Path('ResultTeralyzerMinFreq')

plt.plot(f.flatten(), n_s, label='n_s, real data')
plt.plot(f.flatten(), n_p, label='n_p, real data')

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv')]

for result in resultfiles:
    frequencies, ref_ind = load_material_data(result)
    plt.plot(frequencies, ref_ind, label=result)

freq = np.linspace(0.100*THz, 1.400*THz, 1000)
n_s, n_p = fake_fb(freq)

plt.plot(freq, n_s, label='n_s, fake')
plt.plot(freq, n_p, label='n_p, fake')

plt.legend()
plt.show()
