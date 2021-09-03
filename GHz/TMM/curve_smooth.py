import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from generate_plotdata import export_csv
import pandas as pd

GHz = 10**9

f_min, f_max = 75, 110

df = pd.read_csv(r'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/GHz/Part II/part2gratingBF.csv')

f = array(df['frequency/GHz'])*GHz
data_slice = ((f > f_min*GHz) & (f <= f_max*GHz))

f = f[data_slice]
dic = array(df['dichroism'])[data_slice]

#bf = np.load('HIPS_HHI_yeh_BF_mes.npy')
#f = np.load('f.npy')

yhat = savgol_filter(dic, 91, 2)

#export_csv({'freq': f, 'dichroism': yhat}, 'part2gratingBF_dichroism.csv')

plt.plot(f, dic, label='original')
plt.plot(f, yhat, label='filtered')
plt.legend()
plt.show()

