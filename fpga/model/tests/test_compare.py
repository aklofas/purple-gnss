import numpy as np
import pytest
from pyrx.acquisition import acquire_l1ca
from pyrx.samples import read_samples
from pyrx.synth import synth_l1ca

from fpgamodel.compare import loss_db, prompt_snr_model, prompt_snr_pyrx, start_phase
from fpgamodel.quantize import to_sign_mag

FS, F_IF = 24e6, 6.0e6
N_MS = 100
RECORDING = "L1_20211226_082401_24MHz_I.bin"


def test_start_phase_splits_quarter_chips():
    chip_q, frac = start_phase(0.0, 1.023e6)
    assert (chip_q, frac) == (0, 0)
    chip_q, frac = start_phase(-1e-3 / 1023 / 8, 1.023e6)  # an eighth of a chip late
    assert (chip_q, frac) == (0, 1 << 31)
    chip_q, frac = start_phase(1e-3 / 1023, 1.023e6)  # one chip early: wraps to 4088
    assert (chip_q, frac) == (4088, 0)


def test_loss_db():
    assert loss_db(2.0, 1.0) == pytest.approx(3.0103, abs=1e-3)


def test_total_loss_on_synthetic_signal():
    prn, wrong, doppler = 5, list(range(6, 14)), 1500.0
    x = synth_l1ca(prn, FS, F_IF, doppler, 0, round(FS * N_MS * 1e-3), 45.0, iq=False).real
    snr_ref = prompt_snr_pyrx(x, FS, F_IF, doppler, 0.0, prn, wrong, N_MS)
    snr_model = prompt_snr_model(to_sign_mag(x), FS, F_IF, doppler, 0.0, prn, wrong, N_MS)
    loss = loss_db(snr_ref, snr_model)
    print(f"total loss on synthetic signal {loss:.2f} dB")
    assert 0.0 < loss <= 1.5


def test_lo_loss_on_recording(captures):
    path = captures / RECORDING
    if not path.is_file():
        pytest.skip(f"{RECORDING} not found")
    x = read_samples(path, FS, iq=False, duration=N_MS * 1e-3).real
    results = [acquire_l1ca(x[: round(FS * 0.01)], FS, F_IF, prn) for prn in range(1, 33)]
    best = max(results, key=lambda a: a.cn0 if a.detected else -1.0)
    assert best.detected
    absent = sorted((a for a in results if not a.detected), key=lambda a: a.cn0)
    wrong = [a.prn for a in absent[:8]]  # the eight least signal-like PRNs stand in for noise
    snr_ref = prompt_snr_pyrx(x, FS, F_IF, best.doppler, best.code_offset, best.prn, wrong, N_MS)
    codes = to_sign_mag(x)
    assert np.array_equal(np.abs(x), np.where(codes & 1, 3, 1))  # the recording is 2-bit already
    snr_model = prompt_snr_model(codes, FS, F_IF, best.doppler, best.code_offset, best.prn, wrong, N_MS)
    loss = loss_db(snr_ref, snr_model)
    print(f"PRN {best.prn} at {best.cn0:.1f} dB-Hz: LO loss on the recording {loss:.2f} dB")
    assert 0.0 <= loss <= 0.8
