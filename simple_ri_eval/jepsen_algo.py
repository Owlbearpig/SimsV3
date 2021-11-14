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

    # plt.plot(t, y_avg)
    # plt.show()

    return t, a

def signal_peak_position(t, a):
    return t[np.argmin(a)]

def find_t_offset(t, a_ref, a_sam):
    return np.abs(signal_peak_position(t, a_ref) - signal_peak_position(t, a_sam))

def calc_fft(t, a):
    f, af = fft(t, a)
    f = f.flatten()
    af = af.flatten()

    f, af = f[:len(af) // 2], af[:len(af) // 2]

    return f, af

def fft_phase(f, t0):
    return 2*pi*f*t0

def calc_reduced_phase(af, fft_phase):
    return angle(af*exp(-1j*fft_phase))

def phase_unwrap(phase):
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
    #return np.unwrap(2 * phase) / 2

def phase_linear_fit(phase, f, f_min, f_max):
    f_mask = (f > f_min)*(f < f_max)
    f_region, trusted_phase = f[f_mask], phase[f_mask]

    coefs = poly.polyfit(f_region, trusted_phase, 1)
    b, a = coefs

    phase_fit = poly.polyval(f, coefs)

    plt.plot(f, phase, label='original phase')
    plt.plot(f, phase_fit, label='linear fit full region')
    plt.plot(f_region, trusted_phase, label='trusted phase')
    plt.legend()
    plt.show()

    return phase_fit-2*pi*round(b/(2*pi), 0)

def unwrap_correction(f, phase):
    return phase + 2*pi*f/np.mean(f)

def algo(data_files_ref, data_files_sample):
    t_ref, a_ref = avg_e_field(data_files_ref)
    t_s, a_s = avg_e_field(data_files_sample)

    plt.plot(t_ref, a_ref, label='ref')
    plt.plot(t_s, a_s, label='sample')
    plt.xlabel('time (ps)')
    plt.legend()
    plt.show()

    f_ref, af_ref = calc_fft(t_ref, a_ref)
    f_s, af_s = calc_fft(t_s, a_s)

    data_mask = (0.0 < f_ref)*(f_ref < 1.1)
    f_ref, af_ref, f_s, af_s = f_ref[data_mask], af_ref[data_mask], f_s[data_mask], af_s[data_mask]

    plt.plot(f_ref, 20*np.log10(np.abs(af_ref)), label='ref')
    plt.plot(f_s, 20*np.log10(np.abs(af_s)), label='sample')
    plt.xlabel('frequency (THz)')
    plt.ylabel('$20\log_{10}|FFT(y)|$')
    plt.legend()
    plt.show()

    t0_ref, t0_s = signal_peak_position(t_ref, a_ref), signal_peak_position(t_s, a_s)
    t_offset = t0_s-t0_ref
    print(f'\nref. peak @ {t0_ref} ps, sam. peak @ {t0_s} ps, diff. {t_offset} ps')

    phi0_ref, phi0_s = fft_phase(f_ref, t0_ref), fft_phase(f_s, t0_s)

    plt.plot(f_ref, phi0_ref, label='$\phi_0$ ref')
    plt.plot(f_s, phi0_s, label='$\phi_0$ sam')
    plt.xlabel('frequency (THz)')
    plt.ylabel('fft phase (rad)')
    plt.legend()
    plt.show()

    phi_red_ref, phi_red_s = calc_reduced_phase(af_ref, phi0_ref), calc_reduced_phase(af_s, phi0_s)

    plt.plot(f_ref, phi_red_ref, label='$\phi_{ref}^{red}$')
    plt.plot(f_s, phi_red_s, label='$\phi_{sam}^{red}$')
    plt.xlabel('frequency (THz)')
    plt.ylabel('reduced phases (rad)')
    plt.legend()
    plt.show()

    delta_phi0_star = phase_unwrap(phi_red_s-phi_red_ref)

    plt.plot(f_ref, delta_phi0_star, label='$\Delta\phi_0^*$ ref')
    plt.xlabel('frequency (THz)')
    plt.ylabel('unwrapped reduced phase difference (rad)')
    plt.legend()
    plt.show()

    f_min, f_max = 0.09, 0.22
    delta_phi0 = phase_linear_fit(delta_phi0_star, f_ref, f_min, f_max)

    plt.plot(f_ref, delta_phi0_star, label='$\Delta\phi_0^*$ ref')
    plt.plot(f_ref, delta_phi0, label='$\Delta\phi_0$ ref')
    plt.xlabel('frequency (THz)')
    plt.ylabel('linear fitted reduced phase difference (rad)')
    plt.legend()
    plt.show()

    phi_offset = 2*pi*f_ref*t_offset

    delta_phi = delta_phi0 - phi0_ref + phi0_s + phi_offset

    plt.plot(f_ref, delta_phi, label='$\Delta\phi$ full')
    plt.xlabel('frequency (THz)')
    plt.ylabel('full phase difference (rad)')
    plt.legend()
    plt.show()

    return f_ref, delta_phi

def calc_refractive_index(f, phase_diff, d):
    omega = 2 * pi * f * THz
    return 1 + phase_diff * (c0 / (omega * d))

if __name__ == '__main__':
    d = 4  # thickness mm

    data_dir = Path(fr'Y:\MEGA cloud\AG\BFWaveplates\Data\BowTie_v2\Data\HIPS gratings new setup adjustment - 11112021\{d}mm')

    d = d * 10 ** -3
    angle1, angle2 = '0', '90'

    data_files_1 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle1}deg')
    # data_files_2 = find_files(data_dir, file_extension='.txt', search_str=f'-{angle2}deg')
    data_files_ref = find_files(data_dir, file_extension='.txt', search_str='-refEnd')

    print(data_files_1)
    # print(data_files_2)
    print(data_files_ref)

    f, phase_diff = algo(data_files_ref, data_files_1)

    n = calc_refractive_index(f, phase_diff, d)

    plt.plot(f, n)
    plt.xlabel('frequency (THz)')
    plt.ylabel('refractive index')
    plt.show()
