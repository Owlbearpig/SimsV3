from numpy import pi, arcsin
import numpy as np
from scipy.constants import c as c0
import matplotlib.pyplot as plt

f = 10**12

mm = 10**-3

x = np.linspace(-5*2.51*mm, 5*2.51*mm, 20)

d, l, h, p = 9*mm, 20*mm, 32*mm, 784

dsigma = 6*x*p*l/(d*h**3)

plt.plot(x/mm, dsigma/10**6, label=r'$d\sigma (x) \cdot 10^{-6}$')
plt.ylabel('$d\sigma (x) \cdot 10^{-6}$')
plt.xlabel('x (mm)')
plt.legend()
plt.show()

A = np.linspace(-0.1, 0.1, 20)

dn = arcsin(A)*c0/(pi*f*d)

plt.plot(A, dn/(10**-3), label=r'$dn(A) \cdot 10^{-3}$')
plt.xlim((0.12, -0.12))
plt.ylabel('$dn(A) \cdot 10^{-3}$')
plt.xlabel('A')
plt.legend()
plt.show()
