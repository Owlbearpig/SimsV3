import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

delta_measured = np.load('delta.npy')

n_s_range = np.linspace(1.0, 1.45, 500)
n_p_range = np.linspace(1.0, 1.45, 500)

delta = np.load('0.npy')
#plt.imshow(delta%pi-delta_measured[0])
#plt.show()

minima = np.where(np.abs(delta%pi-delta_measured[0]) < 0.003)

bfs = []
for i, j in zip(minima[0], minima[1]):
    print(n_s_range[i],n_p_range[j])
    bfs.append(n_s_range[i]-n_p_range[j])

plt.plot(bfs)
plt.show()

