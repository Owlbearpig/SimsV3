import numpy as np
import matplotlib.pyplot as plt


lines = []
with open('hwp_state_opt_v1_10k_local_opt.txt') as file:
    for line in file.readlines():
        lines.append(str(line).replace(r',', ''))
b = []
for line in lines:
    b.append(float(line))

b = np.array(b)
print(b)
print(len(b[b>0.5]))
print(len(b[b<0.5]))
plt.scatter(np.arange(len(b)), b)
#plt.scatter(np.arange(len(np.diff(b))), np.abs(np.diff(b)))
plt.show()

from ast import literal_eval

lines = []
with open('min_opt_x_ad.txt') as file:
    for line in file.readlines():
        lines.append(np.array(literal_eval(line)))

dists = []
for i in range(len(lines)):
    if not i == len(lines)-1:
        dists.append(np.abs(lines[i]-lines[i+1]))

dists = np.array(dists)
for i in range(6):
    print(np.average(dists[:, i]))


