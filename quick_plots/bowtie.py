import matplotlib.pyplot as plt
from consts import *
import numpy as np
from pathlib import Path
from functions import find_files, fft

data_dir_old = Path(r'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/BowTieSetup/SiWaferTest')
data_dir_new = Path(r'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/BowTie_v2/Data/2mm HIPS grating 08112021')

data_files_old = find_files(top_dir=data_dir_old, file_extension='.txt', search_str='Ref')
data_files_new = find_files(top_dir=data_dir_new, file_extension='.txt', search_str='ref')


def plot_file(file_path):
    data = np.loadtxt(file_path)
    t,y = data[:, 0], data[:, 1]

    plt.subplot(2, 1, 1)
    plt.plot(t, y, label=file_path.stem)

    f, yf = fft(t, y)
    print(f, yf)
    f = f[:len(f)//2]

    log_yf = -np.log10(np.abs(yf))[:len(f)//2]
    plt.subplot(2, 1, 2)
    plt.plot(f, log_yf, label=file_path.stem)


plot_file(data_files_new[0])

plt.legend()
plt.show()


