"""The Galileo E1 pilot tracker on recordings, against Pocket SDR's C tracker.

The L1 band recording of 2021-12-26 holds five Galileo satellites above 41 dB-Hz on E1. The
reference tracks the data component E1-B and the pilot E1-C of each as separate channels, one
started after another, so its E1-C channels cover the last seconds of the file only and two
satellites have none. The parked car-roof capture holds seven for its whole length; the
reference tracked E1-B there.
"""

import numpy as np
import pytest
from conftest import capture_file
from pocketsdr_ctrk import load_ctrk_ch

from pyrx.acquisition import acquire_signal
from pyrx.channels import track_prns
from pyrx.codes import CA_PERIOD
from pyrx.observables import code_starts
from pyrx.samples import read_samples
from pyrx.signalspec import E1
from pyrx.tracking import track_channels

RECORDING = "L1_20211226_082401_24MHz_I.bin"
REFERENCE = "pocketsdr-trk/L1_20211226_082401_24MHz_I.e1.ctrk.log"
CAPTURE = "sky/20261103_030551_carroof-static"
FS = 24e6
F_IF = 6e6
MUST_PASS = [2, 7, 8, 30, 36]  # 41 to 47 dB-Hz; E11 and E27, at 35, are below the search
PARKED = [13, 15, 18, 21, 27, 30, 34]
ABSENT = 1
COLUMNS = ("time", "lock", "cn0", "coff_ms", "dop")
INAV_SYNC = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])  # Galileo OS SIS ICD 4.3.2.1
INAV_PAGE = 250  # symbols: one page part a second
PERIOD = E1.epochs * CA_PERIOD  # of the code, s


def _at_log_times(r, rows):
    """The reference rows taken 4 s or more into both channels, the epoch holding each row's
    time, and the time from the row to our next start of a code period."""
    starts = code_starts(r)
    last = len(starts) - 1 - (len(starts) - 1) % E1.epochs  # the last epoch beginning a period
    rows = rows[(rows[:, 1] >= 4.0) & (rows[:, 0] >= starts[0] + 4.0) & (rows[:, 0] < starts[last])]
    k = np.searchsorted(starts, rows[:, 0], side="right") - 1
    return rows, k, starts[k + E1.epochs - k % E1.epochs] - rows[:, 0]


def _code_difference(to_code_start, rows):
    """Our code start minus the reference's, modulo the 4 ms code period, in seconds."""
    return (to_code_start - rows[:, 3] * 1e-3 + PERIOD / 2) % PERIOD - PERIOD / 2


def _sync_pattern_offsets(symbols, inverted):
    """Offsets within a page part at which every part of the stream starts with the I/NAV
    sync pattern."""
    bits = (symbols < 0).astype(int) ^ int(inverted)
    out = []
    for offset in range(INAV_PAGE):
        starts = np.arange(offset, len(bits) - len(INAV_SYNC) + 1, INAV_PAGE)
        if len(starts) >= 2 and all(np.array_equal(bits[s : s + 10], INAV_SYNC) for s in starts):
            out.append(offset)
    return out


def _check_channel(r, seconds):
    """What holds of every E1 channel on a satellite in clear view."""
    who = r.prn
    assert r.locked and not r.side_peak and r.t[-1] - r.t[0] > seconds, who
    assert r.sec_sync is not None and r.t[r.sec_sync] - r.t[0] <= 2.0, who
    # the secondary code is found at the end of a code period and counted in whole periods
    assert (r.sec_sync + 1) % E1.epochs == 0 and r.sec_origin % E1.epochs == 0, who

    # every complete window after the sync is phase-locked (0.96 the lowest)
    ends = np.arange(999, len(r.t), 1000)
    windows = r.phase_lock[ends[ends - 999 > r.sec_sync]]
    assert len(windows) >= seconds - 3 and windows.min() > 0.9, (who, windows.min())

    # E1-B: a quarter decibel above the pilot on every satellite of both recordings (+0.23
    # to +0.27 dB), and on the pilot's axis with the opposite sign
    e = np.arange(r.symbol_epoch, len(r.t))
    ratio = np.mean(np.abs(r.data_prompt[e]) ** 2) / np.mean(np.abs(r.corr[e, 0]) ** 2)
    assert abs(10 * np.log10(ratio)) < 0.5, (who, ratio)
    sec = E1.secondary(r.prn)
    data = r.data_prompt[e] * sec[(e - r.sec_origin) % len(sec)] * r.sec_polarity
    along = (data * np.conj(E1.data_axis)).real
    assert np.sum(along**2) >= 0.9 * np.sum(np.abs(data) ** 2), who
    # a symbol is the four epochs of a code period: they add up coherently (3.95 of 4 or
    # more; any other alignment splits a symbol)
    count = len(r.symbols)
    sums = data[: count * E1.symbol_epochs].reshape(count, E1.symbol_epochs).sum(axis=1)
    gain = np.mean(np.abs(sums)) / np.mean(np.abs(data))
    assert gain >= 0.9 * E1.symbol_epochs, (who, gain)

    # every I/NAV page part starts with an uncoded sync pattern. Found as sent, whichever
    # polarity the pilot synced with, it confirms the data component's sign and the symbol
    # edges: no other alignment and no inversion shows it
    assert len(_sync_pattern_offsets(r.symbols, inverted=False)) == 1, who
    assert _sync_pattern_offsets(r.symbols, inverted=True) == [], who


def test_e1_pilots_match_the_reference(captures):
    path = capture_file(captures, RECORDING)
    log = capture_file(captures, REFERENCE)
    head = read_samples(path, FS, False, 0.0, 0.020)
    seeds = []
    for prn in [*MUST_PASS, ABSENT]:
        acq = acquire_signal(head, FS, F_IF, E1, prn)
        assert acq.detected == (prn != ABSENT), (prn, acq.cn0)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
    results = track_channels(path, FS, False, F_IF, seeds, signal="E1")
    data_ref = load_ctrk_ch(log, "E1B", COLUMNS)
    pilot_ref = load_ctrk_ch(log, "E1C", COLUMNS)
    assert {r.sec_polarity for r in results} == {1, -1}  # both polarities occur
    for r in results:
        _check_channel(r, 30.0)
        # Doppler, C/N0 and the code start against the reference's E1-B channel at each of
        # its log lines' own time: 0.61 Hz, 0.41 dB and 22 ns the largest. Its second-order
        # code loop is still ringing at the end of the file, and its channels stand 6 to 15 ns
        # from ours in the mean; half a chip, the side peak, is 489 ns
        rows, k, to_code_start = _at_log_times(r, data_ref[r.prn])
        assert len(rows) >= 5, r.prn
        assert np.max(np.abs(r.doppler[k] - rows[:, 4])) < 1.0, r.prn
        assert abs(np.mean(r.cn0[k]) - np.mean(rows[:, 2])) < 2.0, r.prn
        d_code = _code_difference(to_code_start, rows)
        assert np.max(np.abs(d_code)) < 30e-9, (r.prn, np.max(np.abs(d_code)))
        # and against its E1-C channel where it has run 4 s (three satellites, 2 to 5 lines)
        if r.prn in pilot_ref:
            rows, k, to_code_start = _at_log_times(r, pilot_ref[r.prn])
            assert len(rows) >= 2, r.prn
            assert np.max(np.abs(r.doppler[k] - rows[:, 4])) < 1.0, r.prn
            assert abs(np.mean(r.cn0[k]) - np.mean(rows[:, 2])) < 2.0, r.prn
            d_code = _code_difference(to_code_start, rows)
            assert np.max(np.abs(d_code)) < 30e-9, (r.prn, np.max(np.abs(d_code)))
    assert sum(r.prn in pilot_ref for r in results) == 3


@pytest.mark.slow
def test_e1_walk_matches_the_reference_on_the_parked_capture(captures):
    path = capture_file(captures, f"{CAPTURE}/ch1.bin")
    ref = load_ctrk_ch(capture_file(captures, f"{CAPTURE}/trk_e1b.log"), "E1B", COLUMNS)
    rejected = []
    segments = track_prns(
        path, FS, False, F_IF, [*PARKED, ABSENT],
        t_start=80.0, duration=60.0, rejected=rejected, signal="E1",
    )  # fmt: skip
    # one segment per satellite from the first probe to the end of the span, nothing turned
    # away, nothing of the absent PRN
    assert [r.prn for r in segments] == PARKED and rejected == []
    offsets = []
    for r in segments:
        _check_channel(r, 59.0)
        rows, k, to_code_start = _at_log_times(r, ref[r.prn])
        assert len(rows) >= 50, r.prn
        # 0.98 Hz on the weakest (its channel is a 1 ms Costas loop on the data component)
        assert np.max(np.abs(r.doppler[k] - rows[:, 4])) < 1.5, r.prn
        assert abs(np.mean(r.cn0[k]) - np.mean(rows[:, 2])) < 2.0, r.prn
        d_code = _code_difference(to_code_start, rows)
        assert np.ptp(d_code) < 25e-9, (r.prn, np.ptp(d_code))
        offsets.append(np.mean(d_code))
    # The reference's log times on this capture stand 175 ns from ours, on its L1 C/A
    # channels (176 ns) as on E1. That constant apart, every satellite's code start is the
    # reference's within 1.1 ns in the mean; a channel on a side peak would be 489 ns off
    assert abs(np.median(offsets) + 175e-9) < 20e-9, np.median(offsets)
    assert np.max(np.abs(offsets - np.median(offsets))) < 5e-9, offsets
