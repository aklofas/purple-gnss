"""The pilot tracker on the L5 band recording of 2021-12-26, against Pocket SDR's C tracker.

The recording holds five GPS satellites with L5 and six Galileo satellites on E5a above
38 dB-Hz. The reference log comes from Pocket SDR with its GPS L5 code generator corrected
(its unmodified L5 codes are damaged in the tail of every period and read up to 8 dB low);
it tracks the data and the pilot component of each satellite as separate channels.
"""

import numpy as np
from conftest import capture_file
from pocketsdr_ctrk import load_ctrk_ch

from pyrx.acquisition import acquire_signal
from pyrx.observables import code_starts
from pyrx.samples import read_samples
from pyrx.signalspec import E5A, L5
from pyrx.tracking import track_channels

RECORDING = "L5_20211226_082401_24MHz_IQ.bin"
REFERENCE = "pocketsdr-trk/L5_20211226_082401_24MHz_IQ.l5fix.ctrk.log"
FS = 24e6
# the satellites whose data channel the reference holds for 20 s or more at 38 dB-Hz or more
MUST_PASS = {L5: [4, 8, 18, 26, 27], E5A: [2, 7, 8, 25, 30, 36]}
ABSENT = {L5: 1, E5A: 1}
REFERENCE_SIGNALS = {"L5": ("L5Q", "L5I"), "E5A": ("E5AQ", "E5AI")}  # pilot, data channel
COLUMNS = ("time", "lock", "cn0", "coff_ms", "dop")
FNAV_SYNC = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0])  # Galileo OS SIS ICD 4.2.2.1
FNAV_PAGE = 500  # symbols


def _tracks(path, spec):
    head = read_samples(path, FS, True, 0.0, 0.020)
    seeds = []
    for prn in [*MUST_PASS[spec], ABSENT[spec]]:
        acq = acquire_signal(head, FS, 0.0, spec, prn)
        assert acq.detected == (prn != ABSENT[spec]), (spec.name, prn, acq.cn0)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
    return track_channels(path, FS, True, 0.0, seeds, signal=spec.name)


def _at_log_times(r, rows):
    """The reference rows taken 4 s or more into both channels, the epoch holding each row's
    time, and the time from the row to our next code start."""
    starts = code_starts(r)
    rows = rows[(rows[:, 1] >= 4.0) & (rows[:, 0] >= starts[0] + 4.0) & (rows[:, 0] < starts[-1])]
    k = np.searchsorted(starts, rows[:, 0], side="right") - 1
    return rows, k, starts[k + 1] - rows[:, 0]


def _sync_pattern_offsets(symbols, inverted):
    """Offsets within a page at which every page of the stream starts with the sync pattern."""
    bits = (symbols < 0).astype(int) ^ int(inverted)
    out = []
    for offset in range(FNAV_PAGE):
        starts = np.arange(offset, len(bits) - len(FNAV_SYNC) + 1, FNAV_PAGE)
        if len(starts) >= 2 and all(np.array_equal(bits[s : s + 12], FNAV_SYNC) for s in starts):
            out.append(offset)
    return out


def test_pilots_match_the_reference(captures):
    path = capture_file(captures, RECORDING)
    log = capture_file(captures, REFERENCE)
    for spec in (L5, E5A):
        pilot_sig, data_sig = REFERENCE_SIGNALS[spec.name]
        pilot_ref = load_ctrk_ch(log, pilot_sig, COLUMNS)
        data_ref = load_ctrk_ch(log, data_sig, COLUMNS)
        for r in _tracks(path, spec):
            who = (spec.name, r.prn)
            # held from the sync to the end of the 30 s file
            assert r.locked and r.t[-1] > 30.0, who
            assert r.sec_sync is not None and r.t[r.sec_sync] <= 2.0, who

            # Doppler and C/N0 against the reference's pilot channel at each of its log lines'
            # own time (0.23 Hz and 0.9 dB largest, on the 38 to 40 dB-Hz satellites)
            rows, k, to_code_start = _at_log_times(r, pilot_ref[r.prn])
            assert len(rows) >= 10, who
            assert np.max(np.abs(r.doppler[k] - rows[:, 4])) < 1.0, who
            assert abs(np.mean(r.cn0[k]) - np.mean(rows[:, 2])) < 2.0, who
            # The code start. The reference's second-order code loop is still ringing at the
            # end of the file from the half sample (21 ns) its acquisition leaves: its own
            # pilot and data channels of G27 stand 30 ns apart. Its data channels start
            # earlier and have settled further: 12 ns from ours at most, its pilot channels 33.
            d_code = (to_code_start - rows[:, 3] * 1e-3 + 0.5e-3) % 1e-3 - 0.5e-3
            assert np.max(np.abs(d_code)) < 35e-9, (who, np.max(np.abs(d_code)))
            rows, k, to_code_start = _at_log_times(r, data_ref[r.prn])
            assert len(rows) >= 10, who
            d_code = (to_code_start - rows[:, 3] * 1e-3 + 0.5e-3) % 1e-3 - 0.5e-3
            assert np.max(np.abs(d_code)) < 15e-9, (who, np.max(np.abs(d_code)))

            # no false lock: every complete window after the sync is phase-locked (0.98 the
            # lowest), and the Doppler stays with the reference's data channel, whose 1 ms
            # Costas loop scatters by up to 2.4 Hz at 38 dB-Hz; a false lock stands 25 Hz off
            ends = np.arange(999, len(r.t), 1000)
            windows = r.phase_lock[ends[ends - 999 > r.sec_sync]]
            assert len(windows) >= 27 and windows.min() > 0.9, (who, windows.min())
            assert np.max(np.abs(r.doppler[k] - rows[:, 4])) < 3.0, who

            # the data component: as strong as the pilot, and on its own axis
            e = np.arange(r.symbol_epoch, len(r.t))
            ratio = np.mean(np.abs(r.data_prompt[e]) ** 2) / np.mean(np.abs(r.corr[e, 0]) ** 2)
            assert abs(10 * np.log10(ratio)) < 1.0, (who, ratio)
            dsec = spec.data_secondary(r.prn)
            data = r.data_prompt[e] * dsec[(e - r.sec_origin) % len(dsec)] * r.sec_polarity
            if r.cn0[-1] > 42.0:
                along = (data * np.conj(spec.data_axis)).real
                assert np.sum(along**2) >= 0.9 * np.sum(np.abs(data) ** 2), who
            # its secondary code and symbol edges are where the pilot's sync puts them: the
            # epochs of a symbol add up coherently (9.5 to 9.95 of 10 on L5, 19.8 of 20 on
            # E5a; any other alignment gives less than 1)
            count = len(r.symbols)
            sums = data[: count * spec.symbol_epochs].reshape(count, spec.symbol_epochs).sum(axis=1)
            gain = np.mean(np.abs(sums)) / np.mean(np.abs(data))
            assert gain >= 0.9 * spec.symbol_epochs, (who, gain)

            # Galileo: the F/NAV pages start with an uncoded sync pattern every 500 symbols.
            # Found as sent, not inverted, it confirms the data axis and the sample convention.
            if spec is E5A:
                assert len(_sync_pattern_offsets(r.symbols, inverted=False)) == 1, who
                assert _sync_pattern_offsets(r.symbols, inverted=True) == [], who
