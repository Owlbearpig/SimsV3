import os
from consts import *
from pathlib import Path
from functions import find_files, fft
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, angle
import numpy.polynomial.polynomial as poly


def discard_first_meas(file_list):
    if len(file_list) > 1:
        return file_list[1:]
    else:
        return file_list

def avg_e_field(result_files):
    """
    calc avg of multiple measurements
    result_files : list of datafile paths
    """
    #result_files = discard_first_meas(result_files)

    y_arrs = []
    for res_file in result_files:
        y_arrs.append(np.loadtxt(res_file))

    y_arrs = np.array(y_arrs)

    y_avg = np.mean(y_arrs[:, :, 1], axis=0)
    y_avg = y_avg - np.mean(y_avg[:10])

    t, a = y_arrs[0, :, 0], y_avg

    return t, a


def calc_fft(t, a):
    f, af = fft(t, a)
    f = f.flatten()
    af = af.flatten()

    f, af = np.abs(f[:len(af) // 2]), af[:len(af) // 2]

    return f, af


def phase_unwrap(phase):
    # "old" unwrap from jepsen paper, seems to be same result as np.unwrap, discont=pi
    """
    threshold = pi
    n = len(phase)
    for i in range(0, n-1):
        if (phase[i+1]-phase[i]) > threshold:
            for j in range(i+1, n):
                phase[j] = phase[j] - 2*pi
        else:
            if (phase[i+1]-phase[i]) < -threshold:
                for j in range(i+1, n):
                    phase[j] = phase[j] + 2*pi
    return phase
    """
    return np.unwrap(phase)


def phase_linear_fit(phase, f, f_min, f_max, confused):
    """
    ax+b fit to some freq. region. Subtracts b(offset in rad) from phase.
    """
    f_mask = (f > f_min)*(f < f_max)
    f_region, trusted_phase = f[f_mask], phase[f_mask]

    coefs = poly.polyfit(f_region, trusted_phase, 1)
    b, a = coefs

    phase_fit = poly.polyval(f, coefs)

    if confused:
        plt.figure()
        plt.plot(f, phase, label='original phase')
        plt.plot(f, phase_fit, label='linear fit full region')
        plt.plot(f_region, trusted_phase, label='trusted phase')
        plt.xlim((-0.1, 0.9))
        plt.xlabel('frequency (THz)')
        plt.ylabel('phase (rad)')
        plt.legend()
        plt.show()

    print(f'fit y-intercept: {b}')
    print(f'phase offset by -2pi*{round(b/(2*pi), 0)}\n')

    return phase - 2*pi*round(b/(2*pi), 0)


def algo(data_files_ref, data_files_sample, settings, correct_phase_offset=True):
    """
    load data -> calc. fft -> unwrap phase + rem. phase offset -> return f, abs(phase_sam - phase_ref)
    """
    confused, data_range = settings['enable_plots'], settings['data_range']
    t_ref, a_ref = avg_e_field(data_files_ref)
    t_s, a_s = avg_e_field(data_files_sample)

    if confused:
        plt.figure()
        plt.plot(t_ref, a_ref, label='ref')
        plt.plot(t_s, a_s, label='sam')
        plt.xlabel('time (ps)')
        plt.legend()
        plt.show()

    f_ref, af_ref = calc_fft(t_ref, a_ref)
    f_s, af_s = calc_fft(t_s, a_s)

    data_mask = (data_range[0] < f_ref)*(f_ref < data_range[1])
    f_ref, af_ref, f_s, af_s = f_ref[data_mask], af_ref[data_mask], f_s[data_mask], af_s[data_mask]

    if confused:
        plt.figure()
        plt.plot(f_ref, 10*np.log10(np.abs(af_ref)/max(np.abs(af_ref))), label='ref')
        plt.plot(f_s, 10*np.log10(np.abs(af_s)/max(np.abs(af_ref))), label='sam')
        plt.xlabel('frequency (THz)')
        plt.ylabel(r'$10\log_{10}\frac{|FFT(y)|}{max(|FFT(y)|)}$')
        plt.legend()
        plt.show()

    raw_phase_ref = angle(af_ref)
    raw_phase_s = angle(af_s)

    if confused:
        plt.figure()
        plt.plot(f_ref, raw_phase_ref, label='ref')
        plt.plot(f_s, raw_phase_s, label='sam')
        plt.xlabel('frequency (THz)')
        plt.ylabel('raw phases (rad)')
        plt.legend()
        plt.show()

    unwrapped_raw_phase_ref, unwrapped_raw_phase_s = phase_unwrap(raw_phase_ref), phase_unwrap(raw_phase_s)

    if confused:
        plt.figure()
        plt.plot(f_ref, unwrapped_raw_phase_ref, label='ref')
        plt.plot(f_s, unwrapped_raw_phase_s, label='sam')
        plt.xlabel('frequency (THz)')
        plt.ylabel('unwrapped raw phases (rad)')
        plt.legend()
        plt.show()

    if correct_phase_offset:
        f_min, f_max = settings['trusted_region']
        unwrapped_raw_phase_ref = phase_linear_fit(unwrapped_raw_phase_ref, f_ref, f_min, f_max, confused)
        unwrapped_raw_phase_s = phase_linear_fit(unwrapped_raw_phase_s, f_ref, f_min, f_max, confused)

    delta_phi0 = np.abs(unwrapped_raw_phase_s - unwrapped_raw_phase_ref)  # why do both phases have negative slope?

    if confused:
        plt.figure()
        plt.plot(f_ref, delta_phi0, label='$\Delta\phi$')
        plt.xlabel('frequency (THz)')
        plt.ylabel('phase difference (rad)')
        plt.legend()
        plt.show()

    return f_ref, delta_phi0


def calc_refractive_index(f, phase_diff, d):
    omega = 2 * pi * f * THz
    return 1 + phase_diff * (c0 / (omega * d))


if __name__ == '__main__':
    d = 4  # thickness mm

    if os.name == 'posix':
        base_path = Path(fr'/media/alex/sda2/MDrive/AG/BFWaveplates/Data/BowTie_v2/Data/HIPS gratings new setup adjustment - 11112021')
    else:
        base_path = Path(fr'Y:\MEGA cloud\AG\BFWaveplates\Data\BowTie_v2\Data\HIPS gratings new setup adjustment - 11112021')

    sam_type = 'grating'
    material_type = 'PLA'

    data_dir = base_path / f'{sam_type}s' / material_type / f'{d}mm' # fullplates / gratings

    d = d * 10 ** -3
    angle1, angle2 = '0', '90'

    save_result = False

    settings = {
        'data_range': (0.06, 0.8), # THz,
        'enable_plots': False,
        'trusted_region': (0.05, 0.1), # THz,
    }

    data_files_1 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle1}deg')
    data_files_2 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle2}deg')
    data_files_ref = find_files(data_dir, file_extension='.txt', search_str='-ref') # -refEnd for 4mm

    print('sam1. files:\n', data_files_1)
    print('sam2. files:\n', data_files_2)
    print('ref. files:\n', data_files_ref, '\n')

    f, phase_diff1 = algo(data_files_ref, data_files_1, settings)
    _, phase_diff2 = algo(data_files_ref, data_files_2, settings, correct_phase_offset=True)

    n1 = calc_refractive_index(f, phase_diff1, d)
    n2 = calc_refractive_index(f, phase_diff2, d)

    plt.figure()
    plt.subplot(2,1,1)
    plt.title(f'birefringence measurement: {material_type} {sam_type} {d*10**3} mm')
    plt.plot(f, n1, label=f'{angle1} deg')
    plt.plot(f, n2, label=f'{angle2} deg')
    plt.legend()
    plt.ylabel('refractive index')

    plt.subplot(2, 1, 2)
    plt.plot(f, n1-n2, label=f'n1({angle1} deg) - n2({angle2} deg)')
    plt.legend()
    plt.xlabel('frequency (THz)')
    plt.ylabel('birefringence')
    plt.show()

    if save_result:
        save_data = np.array([f*THz, n1, n2]).transpose()
        np.savetxt(f'{material_type}_{sam_type}_{d*10**3}mm_freq_n{angle1}deg_n{angle2}deg.txt', save_data)
