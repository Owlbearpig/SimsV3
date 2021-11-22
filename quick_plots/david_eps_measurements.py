import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from consts import ROOT_DIR
from functions import find_files
from pathlib import Path

data_dir = Path('/media/alex/sda2/MDrive/AG/BFWaveplates/Data/GHz/Part2/Permittivity')

gratings_files = find_files(data_dir, search_str='Gitter')
full_plates_files = find_files(data_dir, search_str='Platte')

print(pd.read_csv(gratings_files[0], sep=';').keys())

for gitter in gratings_files:
    data = pd.read_csv(gitter, sep=';')
    freq = data['frequency/GHz']
    n_co = data['n:CO']
    n_x = data['n:X']
    plt.plot(freq, n_co, label='n_co' + gitter.stem)
    plt.plot(freq, n_x, label='n_x' + gitter.stem)

plt.legend()
plt.show()

for plate in full_plates_files:
    data = pd.read_csv(plate, sep=';')
    freq = data['frequency/GHz']
    n_co = data['n:CO']
    n_x = data['n:X']
    plt.plot(freq, n_co, label='n_co' + plate.stem)
    plt.plot(freq, n_x, label='n_x' + plate.stem)

plt.legend()
plt.show()