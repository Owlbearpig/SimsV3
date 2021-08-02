from generate_plotdata import export_csv
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

um = 10**-6
THz = 10**12
GHz = 10**9

plt.style.use('fast')

plt.figure()

r = np.linspace(0.1, 15, 1000)

dn = 0.3
single_wp_delta = (2*pi*dn*r) % (2*pi)

s4 = []
for i in range(0, 10):
    if round(i/(4*dn)) > 6:
        break
    s4.append( (round((i*4+1)/(4*dn), 3), pi/2) )

s2 = []
for i in range(0, 10):
    if round(i/(2*dn)) > 6:
        break
    s2.append( (round((i*2+1)/(2*dn), 3), pi) )

new_r_array = np.array([])
new_delta_array = np.array([])
prev_val = single_wp_delta[0]
for i, (r_val, val) in enumerate(zip(r, single_wp_delta)):
    if np.abs(val-prev_val) > 1:
        new_delta_array = np.append(new_delta_array, 'nan')
        new_r_array = np.append(new_r_array, (r[i]+r[i+1])/2)
    new_delta_array = np.append(new_delta_array, val)
    new_r_array = np.append(new_r_array, r_val)
    prev_val = val



export_csv({'x': new_r_array, 'delta': new_delta_array}, 'monochromaticwaveplates_wJumps.csv')
export_csv({'x': r, 'delta': single_wp_delta}, 'monochromaticwaveplates.csv')

print(s4)
print(s2)

plt.plot(r, single_wp_delta, label='Single wp')

plt.scatter([t[0] for t in s4], [t[1] for t in s4])
plt.scatter([t[0] for t in s2], [t[1] for t in s2])
#plt.plot([(i+1)*pi/2 for i in range(10)], '.', label='i*pi/2')
#plt.gca().invert_xaxis()
plt.grid(True)
plt.xlabel(f'thickness/wavelength')
plt.ylabel(r'delta')
#plt.xlim([75, 110])
#plt.ylim([0.3, 0.6])
plt.legend()
plt.show()
