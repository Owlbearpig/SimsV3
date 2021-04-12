from py_pol import jones_vector, jones_matrix
import matplotlib.pyplot as plt
from numpy import pi, sqrt, array

lin_90 = jones_vector.create_Jones_vectors('in lin_90')
lin_90 = lin_90.linear_light(azimuth=pi/2)
c_l = jones_vector.create_Jones_vectors('c_left')
c_l.circular_light(kind='l')
c_r = jones_vector.create_Jones_vectors('c_right')
c_r.circular_light(kind='r')

l4_wp = jones_matrix.create_Jones_matrices('l4_wp')
l4_wp.quarter_waveplate(azimuth=pi/4)

print(l4_wp)

