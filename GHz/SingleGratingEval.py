import os
from pathlib import Path
import numpy as np
from scipy.constants import c as c0
import pandas
import matplotlib.pyplot as plt
from functions import material_values, form_birefringence
from generate_plotdata import export_csv

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
stripes_ghz = np.array([631, 460])
x_sg = np.concatenate((angles_ghz, d_ghz, stripes_ghz))

result_GHz = {
        'name': 'single_grating',
        'comments': '',
        'x': x_sg,
        'bf': 'form',
        'mat_name': ('HIPS_MUT_1_1', '')
}

def fake_fb(f, eps_const):
    m = len(f)
    stripes = stripes_ghz[-2], stripes_ghz[-1]
    eps_mat2 = (np.ones_like(f)).reshape(m, 1)
    eps_mat1 = (np.ones_like(f)*eps_const).reshape(m, 1)
    wls = um*(c0/f).reshape(m, 1)
    n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

    return n_s.flatten(), n_p.flatten()

eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
stripes = stripes_ghz[-2], stripes_ghz[-1]
n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

#plt.plot(f.flatten(), eps_mat1.flatten(), label=r'$\epsilon \ HIPS \ measured$')
#plt.plot(f.flatten(), np.sqrt(eps_mat1.flatten()), label=r'$n \ HIPS \ measured$')
base = Path('ResultTeralyzerMinFreq')

plt.plot(f.flatten()*10**-12, n_s, label='n_s, (VNA)')
plt.plot(f.flatten()*10**-12, n_p, label='n_p, (VNA)')
"""
export_csv({'freqs': f.flatten(), 'n_s': n_s.flatten(), 'n_p': n_p.flatten(), 'bf':(n_p-n_s).flatten()}, 
           'singlegratingSlim_birefringence_calc_highres.csv')
"""
plt.legend()
plt.show()

plt.plot(f.flatten()*10**-12, n_p-n_s, label='birefringence')
plt.legend()
plt.show()

# find result files
resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv')]

for result in resultfiles:
    frequencies, ref_ind = load_material_data(result)
    plt.plot(frequencies*10**-12, ref_ind, label=result + ' ' + 'SystemLab1')

baseHHI = Path("ResultTeralyzerHHI")
resultfilesHHI = [os.path.join(root, name)
               for root, dirs, files in os.walk(baseHHI)
               for name in files
               if name.endswith('.csv')]

for result in resultfilesHHI:
    frequencies, ref_ind = load_material_data(result)
    plt.plot(frequencies*10**-12, ref_ind, label=result + ' ' + 'HHI' )

freq = np.linspace(0.100*THz, 1.400*THz, 1000)
n_s, n_p = fake_fb(freq, 2.149)

#plt.plot(freq, n_s, label=r'$n_s,\ (\epsilon=2.149)$')
#plt.plot(freq, n_p, label=r'$n_p,\ (\epsilon=2.149)$')

freq = np.linspace(0.100*THz, 1.400*THz, 1000)
n_s, n_p = fake_fb(freq, 2.437)

#plt.plot(freq, n_s, label=r'$n_s,\ (\epsilon=2.437)$')
#plt.plot(freq, n_p, label=r'$n_p,\ (\epsilon=2.437)$')

yeh_te = np.load('TMM/yeh_te.npy')
yeh_tm = np.load('TMM/yeh_tm.npy')

#plt.plot(f.flatten(), yeh_te, label=r'$yeh \ te$')
#plt.plot(f.flatten(), yeh_tm, label=r'$yeh \ tm$')
plt.ylabel('RI')
plt.xlabel('Freq. (THz)')
plt.legend()
plt.show()
