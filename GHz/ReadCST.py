import skrf as rf
import matplotlib.pyplot as plt

path = r'C:\Users\POTATO\Desktop\Test\HIPS_d_sweep_110.s4p'
hips_waveguide = rf.Network(path)

print(hips_waveguide)

hips_waveguide.plot_s_db()
plt.show()
