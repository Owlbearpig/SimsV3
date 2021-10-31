import matplotlib.pyplot as plt
from consts import *
from functions import find_files, parse_teralyzer_csv

result_paths = find_files(file_extension='.csv', search_str='COC')
print(result_paths)

res_names, ref_inds, freq_axes = [], [], []
for result_path in result_paths:
    teralyzer_data = parse_teralyzer_csv(result_path)

    res_name = result_path.stem
    f, ref_ind = teralyzer_data['freq'], teralyzer_data['ref_ind']

    res_names.append(res_name), freq_axes.append(f), ref_inds.append(ref_ind)

    plt.plot(f / THz, ref_ind, label=f'{res_name}')

plt.xlabel('Frequency (THz)')
plt.ylabel('Ref. ind.')
plt.legend()
plt.show()

plt.plot(freq_axes[0]/THz, ref_inds[2]-ref_inds[1], label=f'{res_names[2]}-{res_names[1]}')
plt.xlabel('Frequency (THz)')
plt.ylabel('Ref. ind. difference')
plt.legend()
plt.show()

