import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from generate_plotdata import export_csv

bf = np.load('HIPS_HHI_yeh_BF_mes.npy')
f = np.load('f.npy')

yhat = savgol_filter(bf, 51, 3)

export_csv({'freq': f, 'bf': yhat}, 'FBF_TMM_HIPS.csv')

plt.plot(bf, label='original')
plt.plot(yhat, label='filtered')
plt.legend()
plt.show()

