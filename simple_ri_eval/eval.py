from consts import *
from pathlib import Path
from functions import find_files, fft
import matplotlib.pyplot as plt
import numpy as np

#data_dir = ROOT_DIR / 'GHz' / 'ResultTeralyzerHHI'
#data_dir = Path('Y:\MEGA cloud\AG\BFWaveplates\Data\GHz\SingleGratingHHI\SingleGrating')
d = 4 # thickness mm

data_dir = Path(fr'Y:\MEGA cloud\AG\BFWaveplates\Data\BowTie_v2\Data\HIPS gratings new setup adjustment - 11112021\{d}mm')

d = d*10**-3
angle1, angle2 = '0', '90'

data_files_1 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle1}deg')
data_files_2 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle2}deg')
data_files_ref = find_files(data_dir, file_extension='.txt', search_str='-refEnd')
#result_files_ref2 = find_files(data_dir, search_str='ref2')

print(data_files_1)
print(data_files_2)
print(data_files_ref)

def avg_e_field(result_files):
    y_arrs = []
    for res_file in result_files:
        y_arrs.append(np.loadtxt(res_file))

    y_arrs = np.array(y_arrs)

    y_avg = np.mean(y_arrs[:, :, 1], axis=0)
    y_avg = y_avg - np.mean(y_avg[:10])

    t, a = y_arrs[0, :, 0], y_avg

    plt.plot(t, y_avg)
    plt.show()

    return t, a


def format_fft(t, a):
    f, af = fft(t, a)
    f = f.flatten()
    af = af.flatten()

    f, af = f[:len(af) // 2], af[:len(af) // 2]

    # plt.plot(f, 20*np.log(np.abs(af)))
    # plt.show()

    return f, af


def unwrapped_phase(E):
    phase = np.arctan(E.imag / E.real)
    phase_unwrapped = -np.unwrap(2 * phase) / 2
    phase_unwrapped = phase_unwrapped - phase_unwrapped[0]

    return phase_unwrapped


def calc_ref_ind(f, sample, ref, d):
    phase_s, phase_ref = unwrapped_phase(sample), unwrapped_phase(ref)
    #T = sample / ref
    #phase_diff = unwrapped_phase(T)
    phase_diff = phase_s - phase_ref

    omega = 2 * pi * f * THz

    return 1 + phase_diff * (c0/(omega*d))

t, a_avg_1 = avg_e_field(data_files_1)
_, a_avg_2 = avg_e_field(data_files_2)
_, a_avg_ref1 = avg_e_field(data_files_ref)

t_offset_1 = t[np.argmin(a_avg_1)]-t[np.argmin(a_avg_ref1)]
t_offset_2 = t[np.argmin(a_avg_2)]-t[np.argmin(a_avg_ref1)]

f, af_avg_1 = format_fft(t, a_avg_1)
_, af_avg_2 = format_fft(t, a_avg_2)
_, af_avg_ref1 = format_fft(t, a_avg_ref1)

data_range_min, data_range_max = f > 0.280, f < 0.38
range_mask = data_range_min*data_range_max

f = f[range_mask]
af_avg_1, af_avg_2, af_avg_ref1 = af_avg_1[range_mask], af_avg_2[range_mask], af_avg_ref1[range_mask]

n_1 = calc_ref_ind(f, af_avg_1, af_avg_ref1, d)
n_2 = calc_ref_ind(f, af_avg_2, af_avg_ref1, d)

phase_unwrapped_ref, phase_unwrapped_sam_1 = unwrapped_phase(af_avg_ref1), unwrapped_phase(af_avg_1)
phase_unwrapped_sam_2 = unwrapped_phase(af_avg_2)

plt.plot(f*1000, phase_unwrapped_ref, label='phase_unwrapped_ref')
plt.plot(f*1000, phase_unwrapped_sam_1, label=f'phase_unwrapped_{angle1}deg')
plt.plot(f*1000, phase_unwrapped_sam_2, label=f'phase_unwrapped_{angle2}deg')
plt.title('phase_unwrapped')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Phase (rad)')
plt.legend()
plt.show()

plt.plot(f*1000, n_1, label=f'{angle1}deg')
plt.plot(f*1000, n_2, label=f'{angle2}deg')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Refractive index')
plt.legend()
plt.show()

plt.plot(f*1000, n_1-n_2, label=f'n_1({angle1} deg)-n_2({angle2} deg)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Birefringence')
plt.legend()
plt.show()