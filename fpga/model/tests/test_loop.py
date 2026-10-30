import numpy as np
import pytest
from pyrx.acquisition import acquire_l1ca
from pyrx.codes import CA_CHIP_RATE, gps_ca
from pyrx.samples import SampleFile, read_samples
from pyrx.signals import L1_FREQ
from pyrx.tracking import track_l1ca

from fpgamodel.channel import Channel, Dump
from fpgamodel.compare import start_phase
from fpgamodel.loop import Tracker, compare_with_pyrx, track_model
from fpgamodel.quantize import code_bits, to_sign_mag

FS, F_IF = 24e6, 6.0e6
RECORDING = "L1_20211226_082401_24MHz_I.bin"
DURATION = 3.0


def test_tracker_steps_follow_doppler():
    t = Tracker(FS, F_IF, doppler=1000.0)
    carr, code = t.steps()
    assert carr == round((F_IF + 1000.0) / FS * 2**32)
    assert code == round(4 * CA_CHIP_RATE * (1 + 1000.0 / L1_FREQ) / FS * 2**32)


def test_pll_pulls_phase_toward_zero():
    t = Tracker(FS, F_IF, doppler=0.0)
    t.epoch = 1001  # past pull-in: Costas PLL
    fd0 = t.doppler
    t.update(Dump(1002, 0, 0, 1000, 300, 0, 0, 0))  # prompt leads: positive phase error
    assert t.doppler > fd0


def test_dll_speeds_the_code_when_early_wins():
    t = Tracker(FS, F_IF, doppler=0.0)
    base = t.steps()[1]
    for k in range(1, 11):
        t.update(Dump(k, 900, 0, 1000, 0, 700, 0, 0))  # early > late every epoch
    assert t.steps()[1] > base


def test_model_loop_tracks_like_pyrx(captures):
    path = captures / RECORDING
    if not path.is_file():
        pytest.skip(f"{RECORDING} not found")
    head = read_samples(path, FS, iq=False, duration=0.01)
    results = [acquire_l1ca(head, FS, F_IF, prn) for prn in range(1, 33)]
    best = max(results, key=lambda a: a.cn0 if a.detected else -1.0)
    assert best.detected

    x = read_samples(path, FS, iq=False, duration=DURATION).real
    codes = to_sign_mag(x)
    chip_rate = CA_CHIP_RATE * (1 + best.doppler / L1_FREQ)
    chip_q, frac = start_phase(best.code_offset, chip_rate)
    ch = Channel(code_bits(gps_ca(best.prn)))
    tracker = Tracker(FS, F_IF, best.doppler, best.cn0)
    ch.set_steps(*tracker.steps())
    ch.set_phase(0, frac, chip_q)
    ch.enabled = True
    history = track_model(ch, tracker, codes)
    assert tracker.locked
    assert len(history) >= 2990

    ref = track_l1ca(SampleFile(path, FS, iq=False), F_IF, best.prn, best.code_offset,
                     best.doppler, best.cn0, duration=DURATION)
    assert ref.locked
    dopp_rms, code_max_chips = compare_with_pyrx(history, ref, FS, t_from=1.0)
    print(f"PRN {best.prn}: Doppler RMS {dopp_rms:.3f} Hz, code max {code_max_chips:.4f} chips")
    assert dopp_rms < 1.0
    assert code_max_chips < 0.25
    assert np.isfinite(tracker.cn0) and tracker.cn0 > 40.0


def test_track_model_does_not_depend_on_block_size():
    # at 8 MHz a period is 8000 samples, so a 12 000-sample block can hold two dumps; the
    # feedback after each dump must still apply before the next one
    from pyrx.codes import gps_ca
    from pyrx.synth import synth_l1ca

    from fpgamodel.quantize import code_bits, to_sign_mag

    fs, f_if, doppler = 8e6, 2e6, 1200.0
    codes = to_sign_mag(synth_l1ca(7, fs, f_if, doppler, 0, 400_000, 45.0, iq=False).real)
    runs = []
    for block in (12_000, 3_000, 50_000):
        ch = Channel(code_bits(gps_ca(7)))
        tracker = Tracker(fs, f_if, doppler + 40.0)  # start 40 Hz off so the loop has to move
        ch.set_steps(*tracker.steps())
        ch.enabled = True
        runs.append(track_model(ch, tracker, codes, block=block))
    assert len(runs[0]) == 50
    assert runs[0] == runs[1] == runs[2]
