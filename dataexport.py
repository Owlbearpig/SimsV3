import pandas as pd
import numpy as np


"""
data format: {'name_array1': array1, 'name_array2': array2, ...}
"""
def save(data, name):
    df = pd.DataFrame(data=data)
    df.to_csv(name + '.csv')


if __name__ == '__main__':
    points = {'name_array1': np.random.random(10), 'name_array2': np.random.random(10)}
    save(data=points, name='test')
