import matplotlib.pyplot as plt
import numpy as np

from generate_plotdata import export_csv

output_file = 'opt_out_02_15_n6_1.txt'

with open(output_file, 'r') as file:
    content = file.readlines()
    file.close()

test = []
vals, prev_step = [], -1
for i, line in enumerate(content):
    if ' lowest_optimization_result:' in line:
        break

    if 'basinhopping step' in line:
        splits = line.split(' ')
        cur_step = int(splits[2].replace(':', ''))
        if cur_step == 0:
            step_value = float(splits[4])
        else:
            step_value = float(splits[11].replace('\n', ''))

        test.append((cur_step, step_value))
        """
        if prev_step > cur_step:
            break
        prev_step = cur_step
        """
        vals.append(step_value)

print(len(np.arange(0, 200)), len(vals))
export_csv({'step': np.arange(0, 201), 'min_f': vals}, 'convergence.csv')
plt.plot(vals)
plt.show()