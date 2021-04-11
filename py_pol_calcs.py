from py_pol import jones_vector, jones_matrix
import matplotlib.pyplot as plt
from numpy import pi, sqrt, array

lin_90 = jones_vector.create_Jones_vectors('in lin_90')
lin_90 = lin_90.linear_light(azimuth=pi/2)
c_l = jones_vector.create_Jones_vectors('c_left')
c_l.circular_light(kind='l')
c_r = jones_vector.create_Jones_vectors('c_right')
c_r.circular_light(kind='r')

l2_wp = jones_matrix.create_Jones_matrices('l2_wp')
l2_wp.half_waveplate()

#lin_45.draw_ellipse()
#plt.show()
print(l2_wp.rotate(angle=pi/1))
out = l2_wp*lin_90
#out.draw_ellipse()
print(out)
#plt.show()

lin_neg = jones_vector.create_Jones_vectors('lin_neg')
lin_neg.from_matrix(sqrt(1/2)*array([1+1j,1-1j]))
print(lin_neg)
lin_neg.draw_ellipse()
plt.show()



rcp = jones_vector.create_Jones_vectors('rcp')
rcp.from_matrix(array([1-1j,1+1j]))
#rcp.circular_light(kind='r')
rcp.draw_ellipse()
plt.show()
