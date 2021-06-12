import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from py_pol import jones_matrix
from py_pol import jones_vector

THz = 10**12

f = np.linspace(0.1, 2, 1000)*THz
wl = (c0/f)*10**6

bf = 0.05
L = 32*10**-3



d =  [1.68,   6.68,   6.78,    3.43, 3.34,   10.02]
angles = [51.40,  14.59,  71.46,  157.85,  140.64,  75.91]
angles = [0,  0,  0,  0,  0,  0]

J_in = jones_vector.create_Jones_vectors()
J_in.linear_light(azimuth=pi/4)


J1 = jones_matrix.create_Jones_matrices()
J1.retarder_material(ne=1.55, no=1.5, d=d[0]*10**3, wavelength=wl, azimuth=angles[0]*pi/180)

J2 = jones_matrix.create_Jones_matrices()
J2.retarder_material(ne=1.55, no=1.5, d=d[1]*10**3, wavelength=wl, azimuth=angles[1]*pi/180)

J3 = jones_matrix.create_Jones_matrices()
J3.retarder_material(ne=1.55, no=1.5, d=d[2]*10**3, wavelength=wl, azimuth=angles[2]*pi/180)

J4 = jones_matrix.create_Jones_matrices()
J4.retarder_material(ne=1.55, no=1.5, d=d[3]*10**3, wavelength=wl, azimuth=angles[3]*pi/180)

J5 = jones_matrix.create_Jones_matrices()
J5.retarder_material(ne=1.55, no=1.5, d=d[4]*10**3, wavelength=wl, azimuth=angles[4]*pi/180)

J6 = jones_matrix.create_Jones_matrices()
J6.retarder_material(ne=1.55, no=1.5, d=d[5]*10**3, wavelength=wl, azimuth=angles[5]*pi/180)

J = J6*J5*J4*J3*J2*J1

Jqwp = jones_matrix.create_Jones_matrices()
Jqwp.quarter_waveplate(azimuth=pi/4)

plt.plot(f, J.parameters.retardance())
plt.show()

plt.plot(f, (J*J_in).parameters.delay())
plt.show()

