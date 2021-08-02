import numpy as np
import matplotlib.pyplot as plt

w, h = 100, 100
two_state_grid = np.zeros((w, h))

image = np.array([[1,1,1], [0,1,0], [1,0,1]])

fig,ax = plt.subplots(1,1)
im = ax.imshow(two_state_grid, cmap='binary', vmin=0, vmax=1)

while True:
    two_state_grid = np.random.random(two_state_grid.shape)
    two_state_grid[np.where(two_state_grid > 0.5)] = 1
    two_state_grid[np.where(two_state_grid < 0.5)] = 0
    print(np.sum(two_state_grid))
    im.set_data(two_state_grid)

    fig.canvas.draw_idle()
    plt.pause(0.01)
