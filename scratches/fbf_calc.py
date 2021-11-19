import numpy as np
from numpy import array
from consts import *
from functions import form_birefringence
import matplotlib.pyplot as plt

# a, b = 220*um, 320*um 2021 paper values
a, b = 700*um, 400*um

f = np.linspace(0.15, 0.4, 70) * THz
wls = c0/f

# n1, n2 = 1.53, 1 2021 paper val
n1, n2 = 1.89, 1 # PLA

eps_mat1, eps_mat2 = n1**2*np.ones_like(f), n2*np.ones_like(f)

bf = form_birefringence(array([a, b]), wls, eps_mat1, eps_mat2)

n_f = bf[0,:,0]
n_s = bf[1,:,0]

plt.plot(f/THz, n_s-n_f, label='Birefringence')
plt.xlabel('Frequency (THz)')
plt.title(f'$n_1$:{n1}, $n_2$:{n2}, a (mat.): {round(a/um,1)} um, b (air): {round(b/um,1)} um')
plt.legend()
plt.show()
