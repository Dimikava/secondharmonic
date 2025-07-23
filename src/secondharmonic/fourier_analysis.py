import numpy as np
from scipy.fft import rfft

def theoretical(a_0,a_4,phi,alpha):
    return a_0 + a_4 * np.cos(2*(3*phi - 2 * alpha) )

def theoretical2(a_0,a_4,phi,alpha):
    return a_0 + a_4 * np.cos(2*(3*phi + 2 * alpha))

def r_squared(observed, predicted):
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - ss_res / ss_tot

def analyze_pixel_signal(signal, t, r2_thresh=0.8, mag_rel_thresh=0.12):
    fft_result = rfft(signal)
    fft_phases = np.angle(fft_result)
    a_0 = np.abs(fft_result[0]) / len(t)
    a_4 = 2 * np.abs(fft_result[4]) / len(t)

    phase = np.mod(fft_phases[4], 2 * np.pi) / 6
    fit1 = theoretical(a_0, a_4, phase, t)
    r1 = r_squared(signal, fit1)
    fit2 = theoretical2(a_0, a_4, phase, t)
    r2 = r_squared(signal, fit2)

    if r2 > r1:
        phase = np.abs(np.mod(fft_phases[4], 2 * np.pi) - 2 * np.pi) / 6
        fit = theoretical(a_0, a_4, phase, t)
        r = r_squared(signal, fit)
    else:
        fit = fit1
        r = r1

    magnitudes = np.abs(fft_result[1:])
    sorted_indices = np.argsort(magnitudes)[::-1]
    dom = sorted_indices[0] + 1
    dom2 = sorted_indices[1] + 1 if len(sorted_indices) > 1 else None

    mag_dom2 = 2 * np.abs(fft_result[dom2]) / len(t) if dom2 else 0
    mag_rel = mag_dom2 / a_4 if a_4 != 0 else 0
    mag_arm = np.mod(fft_phases[8], 2 * np.pi) / 12 if r > r2_thresh and mag_rel >= mag_rel_thresh else None

    return {
        "r2": r,
        "phase": np.degrees(phase) if r > r2_thresh else None,
        "dominant": dom,
        "mag_rel": mag_rel,
        "arm": mag_arm
    }
