"""Acquisition compared with Pocket SDR's pocket_acq.py on its own recordings.

Pocket SDR integrates the first 10 ms of the file by default, so the same window
is used here. The 24 MHz recording has sample gaps near 40 ms and 90 ms (the code
offset of every satellite jumps there), so integrating the whole file would split
each peak in two; the first 10 ms are clean.
"""

WINDOW = 0.010  # s, Pocket SDR's default acquisition integration time

from conftest import capture_file

from pyrx.acquisition import acquire_l1ca
from pyrx.samples import read_samples

# prn: (code offset s, doppler Hz, cn0 dB-Hz) reported by Pocket SDR on the same file
REF_4MHZ = {
    16: (0.98950e-3, 2568, 44.0),
    26: (0.89975e-3, 610, 47.4),
    29: (0.41325e-3, -2206, 44.1),
    31: (0.28975e-3, -246, 46.7),
    32: (0.69150e-3, -3210, 40.8),
}
WEAK_4MHZ = set(range(1, 33)) - set(REF_4MHZ) - {18}  # Pocket SDR below 36 dB-Hz

REF_24MHZ = {
    10: (0.85150e-3, -2022, 45.1),
    12: (0.15083e-3, -1915, 47.6),
    25: (0.66950e-3, 399, 47.7),
    32: (0.06479e-3, 2092, 49.6),
    31: (0.44771e-3, 2513, 40.4),
}
WEAK_24MHZ = set(range(1, 33)) - set(REF_24MHZ) - {29}


def _check(results, ref, weak, fs):
    for prn, (coff, dop, cn0) in ref.items():
        r = results[prn]
        assert r.detected, prn
        assert abs(r.code_offset - coff) * fs <= 1.0, prn
        assert abs(r.doppler - dop) <= 500.0, prn
        assert abs(r.cn0 - cn0) <= 2.0, prn
    for prn in weak:
        assert not results[prn].detected and results[prn].cn0 < 40.0, prn


def test_reference_l1_4mhz_iq(captures):
    fs = 4e6
    x = read_samples(
        capture_file(captures, "L1_20211202_084700_4MHz_IQ.bin"), fs, iq=True, duration=WINDOW
    )
    results = {prn: acquire_l1ca(x, fs, 0.0, prn) for prn in range(1, 33)}
    _check(results, REF_4MHZ, WEAK_4MHZ, fs)


def test_reference_l1_24mhz_real_if(captures):
    fs = 24e6
    x = read_samples(
        capture_file(captures, "L1_20211201_054600_24MHz_I.bin"), fs, iq=False, duration=WINDOW
    )
    results = {prn: acquire_l1ca(x, fs, 6.0e6, prn) for prn in range(1, 33)}
    _check(results, REF_24MHZ, WEAK_24MHZ, fs)
