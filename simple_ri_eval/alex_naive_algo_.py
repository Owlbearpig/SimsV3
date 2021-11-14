from consts import *
from pathlib import Path
from functions import find_files, fft
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, angle
import numpy.polynomial.polynomial as poly


def avg_e_field(result_files):
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
    # "old" unwrap from jepsen paper, same result as np.unwrap, discont=pi
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
    f_mask = (f > f_min)*(f < f_max)
    f_region, trusted_phase = f[f_mask], phase[f_mask]

    coefs = poly.polyfit(f_region, trusted_phase, 1)
    b, a = coefs

    phase_fit = poly.polyval(f, coefs)

    if confused:
        plt.figure()
        plt.plot(f, phase, label='|sam-ref| phase diff')
        plt.plot(f, phase_fit, label='linear fit full region')
        plt.plot(f_region, trusted_phase, label='trusted phase')
        plt.xlabel('frequency (THz)')
        plt.ylabel('phase (rad)')
        plt.legend()
        plt.show()

    print(f'fit y-intercept: {b}')
    print(f'phase offset by {2*pi*round(b/(2*pi), 0)}\n')

    return phase-2*pi*round(b/(2*pi), 0)


def algo(data_files_ref, data_files_sample, data_range, confused=True):
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

    delta_phi_star = np.abs(unwrapped_raw_phase_s - unwrapped_raw_phase_ref)  # why do both phases have negative slope?

    f_min, f_max = 0.09, 0.22
    delta_phi0 = phase_linear_fit(delta_phi_star, f_ref, f_min, f_max, confused)

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

    data_dir = Path(fr'Y:\MEGA cloud\AG\BFWaveplates\Data\BowTie_v2\Data\HIPS gratings new setup adjustment - 11112021\{d}mm')

    d = d * 10 ** -3
    angle1, angle2 = '0', '90'
    data_range = (0.06, 0.8)

    confused = False  # True, should plot every step (verbose I guess...)
    save_result = False

    data_files_1 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle1}deg')
    data_files_2 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle2}deg')
    data_files_ref = find_files(data_dir, file_extension='.txt', search_str='-refEnd')

    print('sam1. files:\n', data_files_1)
    print('sam2. files:\n', data_files_2)
    print('ref. files:\n', data_files_ref, '\n')

    f, phase_diff1 = algo(data_files_ref, data_files_1, data_range, confused)
    _, phase_diff2 = algo(data_files_ref, data_files_2, data_range, confused)

    n1 = calc_refractive_index(f, phase_diff1, d)
    n2 = calc_refractive_index(f, phase_diff2, d)

    plt.figure()
    plt.subplot(2,1,1)
    plt.title(f'birefringence measurement: HIPS grating {d*10**3} mm')
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
        np.savetxt(f'freq_n{angle1}deg_n{angle2}deg.txt', save_data)
