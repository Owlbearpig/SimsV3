from functions import form_birefringence
import numpy as np
from scipy.constants import c as c0
import matplotlib.pyplot as plt

GHz = 10**9
THz = 10**12
um = 10**-6

freqs = np.linspace(0.25, 2.0, 1000)*THz
wls = c0/freqs
stripes = np.array([300, 700])*um
eps1 = 1.5**2+1j*0.001
eps2 = 1

bf = np.array([])
n1_range = np.linspace(1, 20, 100)
for n1 in n1_range:
    eps1 = n1**2
    n_s, n_p, k_s, k_p = form_birefringence(stripes, wls, eps1, eps2)
    bf = np.append(bf, (n_p-n_s)[200])

plt.plot(n1_range, bf, label='bf')
plt.xlabel('n1')
plt.ylabel('bf at around 500 ghz')
plt.legend()
plt.show()
