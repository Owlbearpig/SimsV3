import numpy as np
from consts import *
import pickle
import matplotlib.pyplot as plt

results = pickle.load(open('polOffset_results_lowRes.p', 'rb'))
var_names = ['f', 'delta', 'b/a', 'eta', 'var_a', 'var_b', 'var_delta']
"""
item: [f, delta, rel, eta, var1, var2, var3]
var1, var2, var3 = pcov[0,0], pcov[1,1], pcov[2,2] # a, b, delta
"""

var_idx = 5

angles, min_var_a, min_var_b, min_var_delta = [], [], [], []
for key, item in results.items():
    angles.append(float(key)), min_var_a.append(min(item[4]))
    min_var_b.append(min(item[5])), min_var_delta.append(min(item[6]))

    if not float(key).is_integer():
        continue

    f = item[0]
    if var_idx == 1:
        plt.plot(f / GHz, item[var_idx] / np.pi, label=f'polOffset: {key}')
        plt.ylabel(f'{var_names[var_idx]}/pi')
    else:
        plt.plot(f / GHz, item[var_idx], label=f'polOffset: {key}')
        plt.ylabel(f'{var_names[var_idx]}')

if var_idx == 1:
    plt.plot(f / 10 ** 9, f * 0 + 0.5 * 1.1, 'k--', label='0,5+10%')
    plt.plot(f / 10 ** 9, f * 0 + 0.5 * 0.9, 'k--', label='0,5-10%')

plt.legend()
plt.grid(True)
plt.xlabel('$f$ in GHz')

plt.xlim([75, 110])
plt.title('Measurement eval. for different polarizer angle offsets')
plt.show()

plt.plot(angles, min_var_a, label='min variance of a with frequency')
plt.plot(angles, min_var_b, label='min variance of b with frequency')
plt.plot(angles, min_var_delta, label='min variance of delta with frequency')
plt.legend()
plt.xlabel('Polarizer offset in deg.')
plt.ylabel('min(var(freq))')
plt.title('Minimum variance with frequency as a function of the polarizer offset.')
plt.show()


