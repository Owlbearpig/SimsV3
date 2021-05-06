import numpy as np
import matplotlib.pyplot as plt

n_s_range = np.linspace(1.0, 1.45, 500)
n_p_range = np.linspace(1.0, 1.45, 400)

image = np.zeros((len(n_s_range), len(n_p_range)))

for i, n_s in enumerate(n_s_range):
    print(i)
    for j, n_p in enumerate(n_p_range):
        image[i,j] = n_s*n_p

plt.imshow(image, extent=[1,1.45,1.45,1])
plt.xlabel('n_p')
plt.ylabel('n_s')
plt.show()

