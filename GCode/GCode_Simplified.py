import numpy as np
from numpy import sqrt, round

filename = "2mm_plate"

c = 0.028206675277192242 # extrusion speed: E'distance*c'

def speed(p1, p2):
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    return round(sqrt((x1-x2)**2+(y1-y2)**2)*c, 3)

def writeline(open_file, s):
    open_file.write('\n'+s)

def sign(i):
    if i % 4 == 0:
        return -1
    if i % 4 == 2:
        return 1
    return 0

def layer_transition(open_file, z, cur_pos, start_pos):
    layer_transition = open('layer_transition', 'r')
    z, h = str(z), str(round(z+0.800, 3))
    for line in layer_transition:
        line = line.replace('z', z)
        line = line.replace('h', h)

        line = line.replace('cur_pos', f'X{cur_pos[0]} Y{cur_pos[1]}')
        line = line.replace('start_pos', f'X{start_pos[0]} Y{start_pos[1]}')

        open_file.write(line)


def end(open_file, z):
    end = open('end', 'r')
    z = round(z+0.600, 3)
    for line in end:
        line = line.replace('z', f'{z}')

        open_file.write(line)

thickness = 2
layer_cnt = int(thickness / 0.200)

dy = 48.459 #dy = 48.609
dx = 0.377
center_width = 149.413-100.587
horiz_segment_cnt = int(center_width / dx)
segment_cnt = 2*horiz_segment_cnt # vertical line + horiz line moves one dx of total width.

total_segment_cnt = layer_cnt*segment_cnt

# Start: G1 X100.973 Y129.300
p0 = np.array([100.973, 129.300])

with open(filename + '.gcode', 'a+', newline='') as file:
    # Header # constant
    header_content = open('header', 'r')
    file.write(header_content.read())

    for layer_indx in range(layer_cnt):
        z = round(0.200+(layer_indx+1)*0.200, 3) # first layer height 0.200 set in header -> z offset
        prev_pos = p0
        for i in range(segment_cnt-1):
            dp = round(prev_pos + np.array([(i%2)*dx, sign(i)*dy]), 3)
            writeline(file, f'G1 X{dp[0]} Y{dp[1]} E{speed(prev_pos, dp)}')

            prev_pos = dp

            prog = 100*((layer_indx*segment_cnt+(i+1))/total_segment_cnt)

            if (layer_indx*segment_cnt+i) % 385 == 0:
                writeline(file, f'M73 P{int(round(prog))}')

        if layer_indx != layer_cnt-1:
            layer_transition(file, z, dp, p0)

    # end
    end(file, z)

    file.close()
