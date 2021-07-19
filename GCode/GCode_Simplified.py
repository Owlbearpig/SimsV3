import datetime
import numpy as np
from numpy import sqrt, round
import matplotlib.pyplot as plt


filename = "2mm_plate"

c = 0.028206675277192242 # extrusion speed: E[distance*c]

def speed(p1, p2):
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    return round(sqrt((x1-x2)**2+(y1-y2)**2)*c, 3)

def writeline(open_file, s):
    open_file.write('\n'+s)

def sign(i):
    if (i % 4) == 0:
        return -1
    if (i % 4) == 2:
        return 1
    return 0

def layer_transition(open_file, z, cur_pos, start_pos):
    layer_transition = open('layer_transition', 'r')
    for line in layer_transition:
        line = line.replace('z', str(z))
        line = line.replace('h', str(z + 0.8))

        line = line.replace('cur_pos', f'X{cur_pos[0]} Y{cur_pos[1]}')
        line = line.replace('start_pos', f'X{start_pos[0]} Y{start_pos[1]}')

        open_file.write(line)

thickness = 2
layer_cnt = int(thickness / 0.200)

dy = 48.459 #dy = 48.609
dx = 0.377
center_width = 149.413-100.587
line_count =  int(center_width / dx)

# Start: G1 X100.973 Y129.300
p0 = np.array([100.973, 129.300])

with open(filename + '.gcode', 'a+', newline='') as file:
    # Header # constant
    header_content = open('header', 'r')
    file.write(header_content.read())

    for layer_indx in range(layer_cnt):

        prev_pos = p0
        for i in range(2*line_count-1):
            dp = prev_pos + np.array([(i%2)*dx, sign(i)*dy])
            writeline(file, f'G1 X{dp[0]} Y{dp[1]} E{speed(prev_pos, dp)}')

            prev_pos = dp

        layer_transition(file, layer_indx*0.200, dp, p0)

    file.close()
