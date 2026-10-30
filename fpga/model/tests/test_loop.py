import pytest
from pyrx.acquisition import acquire_l1ca
from pyrx.codes import CA_CHIP_RATE, gps_ca
from pyrx.samples import ArraySamples, SampleFile, read_samples
from pyrx.signals import L1_FREQ
from pyrx.synth import synth_l1ca
from pyrx.tracking import track_l1ca

from fpgamodel.channel import Channel, Dump
from fpgamodel.compare import start_phase
from fpgamodel.loop import Tracker, compare_with_pyrx, track_model
from fpgamodel.quantize import code_bits, to_sign_mag

FS, F_IF = 24e6, 6.0e6
FS_SYN, F_IF_SYN, DOPPLER_SYN = 5e6, 1.25e6, 1200.0
RECORDING = "L1_20211226_082401_24MHz_I.bin"
DURATION = 3.0


def test_tracker_steps_follow_doppler():
    t = Tracker(FS, F_IF, doppler=1000.0)
    carr, code = t.steps()
    assert carr == round((F_IF + 1000.0) / FS * 2**32)
    assert code == round(4 * CA_CHIP_RATE * (1 + 1000.0 / L1_FREQ) / FS * 2**32)


def _synthetic(cn0, seed=0, duration=2.2, data_bits=True):
    """A data-modulated L1 C/A signal at 5 MHz (1.25 MHz IF), the smallest rate the quarter-chip
    code NCO allows; float samples for pyrx and 2-bit codes for the channel."""
    n = round(FS_SYN * duration)
    x = synth_l1ca(7, FS_SYN, F_IF_SYN, DOPPLER_SYN, 0, n, cn0, iq=False, seed=seed, data_bits=data_bits)
    return x, to_sign_mag(x.real)


def _track_synthetic(codes, doppler_seed, cn0_seed):
    ch = Channel(code_bits(gps_ca(7)))
    tracker = Tracker(FS_SYN, F_IF_SYN, doppler_seed, cn0_seed)
    ch.set_steps(*tracker.steps())
    ch.enabled = True
    track_model(ch, tracker, codes)
    return tracker


def test_tracker_matches_pyrx_on_its_correlator_outputs():
    # fed pyrx's own prompt, early and late sums epoch by epoch, the tracker's carrier loop
    # must reproduce pyrx's Doppler exactly (the same arithmetic in the same order), and its
    # C/N0 estimate, built without pyrx's noise tap, must agree with pyrx's to the dB
    x, _ = _synthetic(40.0)
    ref = track_l1ca(ArraySamples(x, FS_SYN), F_IF_SYN, 7, 0.0, DOPPLER_SYN + 5.0, 45.0)
    assert ref.locked and len(ref.t) >= 2000
    tracker = Tracker(FS_SYN, F_IF_SYN, DOPPLER_SYN + 5.0, 45.0)
    for k in range(len(ref.t)):
        prompt, early, late = ref.corr[k][:3]
        sums = (early.real, early.imag, prompt.real, prompt.imag, late.real, late.imag)
        tracker.update(Dump(k + 1, *sums, 0))
        assert tracker.doppler == ref.doppler[k], k
    assert abs(tracker.cn0 - ref.cn0[-1]) < 1.0
    assert tracker.locked


@pytest.mark.parametrize("cn0", [35.0, 40.0, 45.0])
def test_cn0_estimate_holds_lock_through_data_bits(cn0):
    # seeded 5 Hz and 5 dB off, through pull-in and data bits: the estimate settles within
    # 1.5 dB of the truth and lock is never declared lost
    _, codes = _synthetic(cn0)
    tracker = _track_synthetic(codes, DOPPLER_SYN + 5.0, cn0 + 5.0)
    assert tracker.locked
    assert abs(tracker.cn0 - cn0) < 1.5, tracker.cn0


def test_noise_alone_loses_lock_after_pull_in():
    _, codes = _synthetic(5.0)  # 5 dB-Hz: the signal is 20 dB under the noise per epoch
    tracker = _track_synthetic(codes, DOPPLER_SYN, 45.0)
    assert not tracker.locked
    assert tracker.cn0 < 25.0


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
    tracker = Tracker(FS, F_IF, best.doppler, best.cn0, skip_first=True)  # starts mid-period
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
    # measured 0.41 Hz and 0.0006 chips; a doubled FLL gain gives 0.76 Hz and a dead DLL
    # 0.0057 chips (the carrier arithmetic itself is pinned by the pyrx equality test)
    assert dopp_rms < 0.5
    assert code_max_chips < 0.002
    assert abs(tracker.cn0 - ref.cn0[-1]) < 2.0


def test_track_model_does_not_depend_on_block_size():
    # a block may end anywhere in a period (3 000 samples at an 8 000-sample period) or hold
    # several (50 000); the feedback after each dump must still apply before the next sample
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
