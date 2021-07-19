import datetime
import numpy as np
from numpy import sqrt, round
import matplotlib.pyplot as plt


filename = "2mm_plate"

F = 1200
c = 0.028206675277192242 # extrusion speed: E[distance*c]
delt = 0.377


def speed(p1, p2):
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    return round(sqrt((x1-x2)**2+(y1-y2)**2)*c, 3)


def writeline(open_file, s):
    open_file.write('\n'+s)


with open(filename + '.gcode', 'a+', newline='') as file:
    # Header
    header_content = open('header', 'r')
    file.write(header_content.read())
    # Start: G1 X100.973 Y129.300

    writeline(file, f'G1 X100.973 Y80.691 E{speed([100.973, 129.300],[100.973, 80.691])}')

    file.close()
