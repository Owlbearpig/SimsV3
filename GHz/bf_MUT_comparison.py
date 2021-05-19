from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, array
import pandas
from results import result_GHz, stripes_ghz
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

base = Path('ResultTeralyzerMinFreq')

resultfiles = [os.path.join(root, name)
               for root, dirs, files in os.walk(base)
               for name in files
               if name.endswith('.csv')]
print(resultfiles)
for result in resultfiles:
    if '90deg' in str(result):
        frequencies, ref_ind_90deg = load_material_data(result)
        plt.plot(frequencies*10**-12, ref_ind_90deg, label=result + ' ' + 'SystemLab1')
    elif '180deg' in str(result):
        frequencies, ref_ind_180deg = load_material_data(result)
        plt.plot(frequencies * 10 ** -12, ref_ind_180deg, label=result + ' ' + 'SystemLab1')
    elif '/0deg' in str(result):
        frequencies, ref_ind_0deg = load_material_data(result)
        plt.plot(frequencies * 10 ** -12, ref_ind_0deg, label=result + ' ' + 'SystemLab1')
    elif '270deg' in str(result):
        frequencies, ref_ind_270deg = load_material_data(result)
        plt.plot(frequencies * 10 ** -12, ref_ind_270deg, label=result + ' ' + 'SystemLab1')
    else:
        frequencies, ref_ind_sample = load_material_data(result)
        plt.plot(frequencies * 10 ** -12, ref_ind_sample, label=result + ' ' + 'SystemLab1')

#plt.plot(frequencies*10**-12, ref_ind_90deg-ref_ind_0deg, label='90deg-0deg' + ' ' + 'SystemLab1')
#plt.plot(frequencies*10**-12, ref_ind_270deg-ref_ind_180deg, label='270deg-180deg' + ' ' + 'SystemLab1')


eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(result_GHz, return_vals=True)
stripes = stripes_ghz[-2], stripes_ghz[-1]
n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)
plt.plot(f.flatten()*10**-12, n_s, label='n_s, (VNA)')
plt.plot(f.flatten()*10**-12, n_p, label='n_p, (VNA)')

#plt.plot(f.flatten()*10**-12, n_p-n_s, label='n_p-n_s, (VNA)')
fakeresult = {
        'name': 'result',
        'comments': '',
        'x': '',
        'bf': 'form',
        'mat_name': ('HIPS_HHI', '')
}

eps_mat1, eps_mat2, _, _, _, _, f, wls, m = material_values(fakeresult, return_vals=True)
stripes = stripes_ghz[-2], stripes_ghz[-1]
n_s_2mm, n_p_2mm, k_s, k_p = form_birefringence(stripes, wls, eps_mat1, eps_mat2)

plt.plot(f.flatten()*10**-12, np.sqrt(eps_mat1), label='HIPS HHI')

plt.plot(f.flatten()*10**-12, n_s_2mm, label='n_s, (HHI)')
plt.plot(f.flatten()*10**-12, n_p_2mm, label='n_p, (HHI)')

plt.legend()
plt.show()

baseHHI = Path("ResultTeralyzerHHI")
resultfilesHHI = [os.path.join(root, name)
               for root, dirs, files in os.walk(baseHHI)
               for name in files
               if name.endswith('.csv')]

for result in resultfilesHHI:
    frequencies, ref_ind = load_material_data(result)
    plt.plot(frequencies*10**-12, ref_ind, label=result + ' ' + 'HHI' )


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
