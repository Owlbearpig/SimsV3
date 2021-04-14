import pandas as pd
import numpy as np
from plotting import draw_ellipse

THz = 10**12

"""
data format: {'name_array1': array1, 'name_array2': array2, ...}
"""
def save(data, name):
    df = pd.DataFrame(data=data)
    df.to_csv(name + '.csv')


def pe_export(f, jones_vec, name):
    Ex, Ey = draw_ellipse(jones_vec, return_values=True)
    data = {}
    for ind, freq in enumerate(f.flatten()):
        col_name = str(round((1/THz)*freq, 2))
        data[col_name + '_X'] = Ex[ind, :]
        data[col_name + '_Y'] = Ey[ind, :]

    save(data, name=name+'_pe')


if __name__ == '__main__':
    points = {'name_array1': np.random.random(10), 'name_array2': np.random.random(10)}
    save(data=points, name='test')
