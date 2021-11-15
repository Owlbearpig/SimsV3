import numpy as np
from consts import *
from functions import find_files
from pathlib import Path
import matplotlib.pyplot as plt

path = Path(rf'E:\Projects\SimsV3\simple_ri_eval')

files = find_files(path, file_extension='.txt', search_str='HIPS')

files_new = [files[2], files[3], files[4], files[5], files[0], files[1]]
labels, n1_list, n2_list = [], [], []

for file in files_new:
    data = np.loadtxt(file)
    idx_280GHz = np.argmin(np.abs(data[:, 0]-280*GHz))

    n1, n2 = data[idx_280GHz, 1:]
    n1_list.append(n1), n2_list.append(n2)

    labels.append(str(file.stem).split('_')[2])

    print(str(file.stem).split('_')[2])

plt.plot(labels, n1_list, label='n1 (0deg)')
plt.plot(labels, n2_list, label='n2 (90deg)')
plt.ylabel('refractive index')
plt.legend()
plt.show()

for i, file in enumerate(files_new):
    data = np.loadtxt(file)

    f, n1, n2 = data[:, 0], data[:, 1], data[:, 2]
    f_range = (100*GHz < f)*(f < 400*GHz)
    f, n1, n2 = f[f_range], n1[f_range], n2[f_range]

    plt.figure()
    plt.subplot(2,1,1)
    plt.title(f'birefringence measurement: HIPS grating ' + str(file.stem).split('_')[2])
    plt.plot(f/GHz, n1, label=f'{0} deg')
    plt.plot(f/GHz, n2, label=f'{90} deg')
    plt.legend()
    plt.ylabel('refractive index')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)

    plt.subplot(2, 1, 2)
    plt.plot(f/GHz, n1-n2, label=f'{0} deg - {90} deg')
    plt.legend()
    plt.xlabel('frequency (GHz)')
    plt.ylabel('birefringence')
    #plt.savefig(str(file.stem).split('_')[2] + f'_{i}_.png')
    plt.show()
