import numpy as np
import matplotlib.pyplot as plt

delta_phi0 = np.load('eps_brute_fit.npy')

plt.plot(delta_phi0)
plt.show()