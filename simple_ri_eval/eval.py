from consts import *
from pathlib import Path
from functions import find_files, fft
import matplotlib.pyplot as plt
import numpy as np

#data_dir = ROOT_DIR / 'GHz' / 'ResultTeralyzerHHI'
data_dir = Path('Y:\MEGA cloud\AG\BFWaveplates\Data\GHz\SingleGratingHHI\SingleGrating')

result_files_90deg = find_files(data_dir, search_str='-90deg')
result_files_ref1 = find_files(data_dir, search_str='ref1')
result_files_0deg = find_files(data_dir, search_str='-0deg')
result_files_ref2 = find_files(data_dir, search_str='ref2')

def avg_e_field_f_space(result_files):
    y_arrs = []
    for res_file in result_files:
        y_arrs.append(np.loadtxt(res_file))

    y_arrs = np.array(y_arrs)

    y_avg = np.mean(y_arrs[:, :, 1], axis=0)
    y_avg = y_avg - np.mean(y_avg[:10])

    #plt.plot(y_avg)
    #plt.show()

    t, a = y_arrs[0, :, 0], y_avg
    f, af = fft(t, a)
    f = f.flatten()
    af = af.flatten()

    plt.plot(f[:len(af)//2], 20*np.log(np.abs(af[:len(af)//2])))
    plt.show()

    return f[:len(af)//2], af[:len(af)//2]

f, avg_E_90deg = avg_e_field_f_space(result_files_90deg)
f_ref1, avg_E_ref1 = avg_e_field_f_space(result_files_ref1)

H = avg_E_90deg / avg_E_ref1
omega = 2*pi*f*THz
d = 4*10**-3

plt.plot(f, 1 + np.angle(H)*(c0/(omega*d)))
plt.xlim((-0.1, 2.5))
plt.show()

