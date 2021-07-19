import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv

filename = "2mm_plate"

F = 1200
c = 0.028206675277192242 # extrusion speed: E[distance*c]


def writeline(open_file, s):
    open_file.write('\n' + s)


with open(filename + '.gcode', 'a+', newline='') as file:
    # Header
    header_content = open('header', 'r')
    file.write(header_content.read())

    writeline(file, 'abe')

    file.close()
