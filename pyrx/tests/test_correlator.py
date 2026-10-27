import numpy as np

from pyrx.codes import CA_CHIP_RATE, gps_ca, sample_code
from pyrx.correlator import correlate

FS = 4e6
N = 4000
TAPS = np.array([0.0, 0.25, -0.25, -10.0])  # prompt, early, late, noise


def _clean(prn=7, f=1234.0, phase=0.0, code_phase=0.0):
    k = np.arange(N)
    code = sample_code(gps_ca(prn), CA_CHIP_RATE, FS, N, phase_chips=code_phase)
    return (code * np.exp(2j * np.pi * (f * k / FS + phase))).astype(np.complex64)


def test_correlate_aligned_replica():
    c = correlate(_clean(), FS, 1234.0, 0.0, gps_ca(7), CA_CHIP_RATE, 0.0, TAPS)
    assert c.shape == (4,)
    assert abs(c[0] - N) < 0.02 * N  # prompt: full power, real
    assert abs(abs(c[1]) / N - 0.75) < 0.05  # early, +0.25 chip
    assert abs(abs(c[2]) / N - 0.75) < 0.05  # late, -0.25 chip
    assert abs(c[3]) < 0.1 * N  # noise tap


def test_correlate_carrier_phase_rotates_prompt():
    c = correlate(_clean(phase=0.1), FS, 1234.0, 0.0, gps_ca(7), CA_CHIP_RATE, 0.0, TAPS)
    assert abs(np.angle(c[0]) / (2 * np.pi) - 0.1) < 0.01


def test_correlate_code_phase_sign():
    # Signal code phase is 0.1 chip behind the replica (signal arrived late): late tap wins.
    c = correlate(_clean(code_phase=-0.1), FS, 1234.0, 0.0, gps_ca(7), CA_CHIP_RATE, 0.0, TAPS)
    assert abs(c[2]) > abs(c[1])


def test_correlate_real_input_with_if():
    fs, n, f_if = 24e6, 24000, 6.0e6
    k = np.arange(n)
    code = sample_code(gps_ca(3), CA_CHIP_RATE, fs, n)
    x = (code * np.cos(2 * np.pi * (f_if - 2000.0) * k / fs)).astype(np.complex64)
    c = correlate(x, fs, f_if - 2000.0, 0.0, gps_ca(3), CA_CHIP_RATE, 0.0, TAPS)
    assert abs(c[0] - n / 2) < 0.03 * n  # real input: half the power, real
