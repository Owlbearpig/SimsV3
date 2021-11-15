import matplotlib.pyplot as plt
from consts import *
import numpy as np
from pathlib import Path
from functions import find_files, fft

#data_dir_old = Path(r'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/BowTieSetup/SiWaferTestFlipped')
#data_dir_old = Path(r'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/BowTieSetup/SystemZustandVorUmbau')
data_dir_old = Path(r'/media/alex/sda2/Data/Mockups Images/Image1')

data_dir_new = Path(r'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/BowTie_v2/Data/2mm HIPS grating new - 11112021')

data_files_old = find_files(top_dir=data_dir_old, file_extension='.txt', search_str='X140.000 mm-Y0.000')
data_files_new = find_files(top_dir=data_dir_new, file_extension='.txt', search_str='maxAmpl')

def rem_offset(y):
    return y - np.mean(y)

def normalize(y):
    return y / np.max(y)

def plot_file(file_path):
    data = np.loadtxt(file_path)
    t, y = data[:, 0], data[:, 1]
    if t[0] > 1000:
        t = t - 900
    y = rem_offset(y)
    #y = normalize(y)

    plt.subplot(2, 1, 1)
    plt.plot(t, y, label=file_path.stem)
    plt.xlabel('Time (ps)')
    plt.ylabel('Normalized ampl. (arb. unit)')

    f, yf = fft(t, y)
    f, yf = f[0], yf[0]
    len_f_half = len(f)//2
    f, yf = abs(f[:len_f_half]), yf[:len_f_half]

    log_yf = 10*np.log10(np.abs(yf))
    plt.subplot(2, 1, 2)
    plt.plot(f, log_yf, label=file_path.stem)
    plt.xlim((-0.1, 1.4))
    plt.xlabel('Freq. (THz)')
    plt.ylabel('$10\log_{10}|FFT(y)|$')

for file in data_files_new:
    plot_file(file)

plot_file(data_files_old[0])

plt.legend()
plt.show()


