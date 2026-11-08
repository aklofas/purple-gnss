from types import SimpleNamespace

import numpy as np
import pytest
from support import assert_tracks_equal

from pyrx import channels
from pyrx.acquisition import AcqResult, acquire_l1ca
from pyrx.channels import track_prn, track_prns
from pyrx.navdecode import decode_lnav, ephemerides_by_prn
from pyrx.observables import LLI_HALF_CYCLE, LLI_SLIP, make_observables
from pyrx.samples import SampleFile, read_samples
from pyrx.tracking import track_l1ca

FS = 2.5e6


def test_track_prn_returns_two_segments_around_a_gap(gap_file):
    segs = track_prn(gap_file, FS, True, 0.0, 4, interval=5.0)
    assert len(segs) == 2
    a, b = segs
    assert a.t[0] < 0.01 and 15.0 <= a.lost_at <= 18.0 and not a.locked
    assert 25.0 <= b.t[0] < 25.0 + 0.001 + 0.02 and b.locked  # grid 0,5,..: 20 is noise, 25 detects
    assert b.t[0] > a.t[-1]


def test_track_prns_is_independent_of_the_worker_count(gap_file):
    one = track_prns(gap_file, FS, True, 0.0, [4, 9], workers=1, interval=5.0)
    two = track_prns(gap_file, FS, True, 0.0, [4, 9], workers=2, interval=5.0)
    assert [s.prn for s in one] == [4, 4]  # PRN 9 is not in the file
    assert len(two) == len(one)
    for s1, s2 in zip(one, two):
        assert_tracks_equal(s1, s2)


def test_absent_prn_is_probed_on_every_grid_instant(gap_file, monkeypatch):
    calls = []
    real = channels.acquire_l1ca

    def counting(samples, fs, f_if, prn, **kw):
        calls.append(len(samples))
        return real(samples, fs, f_if, prn, **kw)

    monkeypatch.setattr(channels, "acquire_l1ca", counting)
    assert track_prn(gap_file, FS, True, 0.0, 9) == []
    assert len(calls) == 4 and set(calls) == {round(FS * 0.020)}  # 0, 10, 20, 30 s fit; 40 does not


def test_false_alarm_seed_is_dropped_and_the_grid_kept(gap_file, monkeypatch):
    # a "detection" of the absent PRN 9 at every probe: the trial never bit-syncs
    tracked = []
    real = channels.track_l1ca

    def recording(*a, **kw):
        tracked.append(kw["t_start"])
        return real(*a, **kw)

    monkeypatch.setattr(channels, "acquire_l1ca", lambda *a, **k: _always_detected(9))
    monkeypatch.setattr(channels, "track_l1ca", recording)
    assert track_prn(gap_file, FS, True, 0.0, 9, duration=22.0) == []
    assert tracked == [0.0, 10.0, 20.0]  # one trial per grid instant, no full track


def test_track_prn_short_duration_makes_no_probe(gap_file, monkeypatch):
    monkeypatch.setattr(channels, "acquire_l1ca", lambda *a, **k: pytest.fail("probed"))
    assert track_prn(gap_file, FS, True, 0.0, 4, duration=0.015) == []


def test_probe_grid_anchors_at_t_start(gap_file, monkeypatch):
    starts = []
    real = channels.read_samples

    def recording(path, fs, iq, t, d):
        starts.append(t)
        return real(path, fs, iq, t, d)

    monkeypatch.setattr(channels, "read_samples", recording)
    track_prn(gap_file, FS, True, 0.0, 9, t_start=3.0, interval=10.0)
    assert starts == [3.0, 13.0, 23.0, 33.0]
    # after a lost segment the walk resumes on the same grid: PRN 4 found at 3 s is lost near
    # 16 s, 23 s falls in the gap, 33 s finds it again
    starts.clear()
    segs = track_prn(gap_file, FS, True, 0.0, 4, t_start=3.0, interval=10.0)
    assert starts == [3.0, 23.0, 33.0]
    assert [round(s.t[0]) for s in segs] == [3, 33] and 15.0 <= segs[0].lost_at <= 18.0


def test_a_loss_exactly_on_a_grid_instant_is_probed_there(gap_file, monkeypatch):
    # (32.002 - 2.002) / 10 is 3.0000000000000004 in floating point: a ceil of that quotient
    # skipped the probe at 32.002 s and went to 42.002 s
    starts = []
    real = channels.read_samples

    def recording(path, fs, iq, t, d):
        starts.append(t)
        return real(path, fs, iq, t, d)

    lost = iter([32.002, 32.002, None, None])  # trial and segment per accepted probe
    monkeypatch.setattr(channels, "read_samples", recording)
    monkeypatch.setattr(channels, "acquire_l1ca", lambda *a, **k: _always_detected(9))
    monkeypatch.setattr(channels, "track_l1ca", lambda *a, **k: SimpleNamespace(lost_at=next(lost)))
    monkeypatch.setattr(channels, "_bit_syncs", lambda trial: True)
    track_prn(gap_file, FS, True, 0.0, 9, t_start=2.002, interval=10.0)
    assert starts == [2.002, 32.002]


def _always_detected(prn):
    return AcqResult(prn, 0.0005, 0.0, 36.0, True, np.zeros((1, 1), np.float32), np.zeros(1))


def test_ephemeris_assembles_across_the_gap(gap_file):
    segs = track_prn(gap_file, FS, True, 0.0, 4, interval=5.0)
    lnavs = [decode_lnav(s) for s in segs]
    assert [[sf.sf_id for sf in ln.subframes] for ln in lnavs] == [[1, 2], [3]]
    assert all(ln.ephemeris is None for ln in lnavs)
    assert ephemerides_by_prn(lnavs)[4].iode == 149


def test_observables_cover_both_segments_with_a_fresh_carrier_on_the_second(gap_file):
    segs = track_prn(gap_file, FS, True, 0.0, 4, interval=5.0)
    lnavs = [decode_lnav(s) for s in segs]
    epochs = make_observables(segs, lnavs)
    per_epoch = [(e.tow, [o for o in e.obs if o.prn == 4]) for e in epochs]
    assert all(len(obs) == 1 for _, obs in per_epoch)
    t0 = per_epoch[0][0]
    first = [o for tow, (o,) in per_epoch if tow - t0 < 17.0]
    second = [o for tow, (o,) in per_epoch if tow - t0 >= 17.0]
    assert first and second
    gap_start = per_epoch[len(first)][0] - per_epoch[len(first) - 1][0]
    assert 8.0 <= gap_start <= 14.0  # loss near 16 s, bit sync of the new segment near 28.5 s
    assert second[0].lli & LLI_SLIP and second[0].lli & LLI_HALF_CYCLE and second[-1].lli == 0
    assert not (second[1].lli & LLI_SLIP)
    assert abs(second[0].pseudorange - first[-1].pseudorange) < 3e3  # 800 Hz for about 12 s


def test_short_tail_segment_yields_no_observations(gap_file):
    # a segment too short to decode a subframe adds nothing and breaks nothing
    segs = track_prn(gap_file, FS, True, 0.0, 4, interval=5.0)[:1]
    acq = acquire_l1ca(read_samples(gap_file, FS, True, 37.0, 0.02), FS, 0.0, 4)
    tail = track_l1ca(
        SampleFile(gap_file, FS, True), 0.0, 4, acq.code_offset, acq.doppler, acq.cn0, t_start=37.0
    )
    assert tail.locked and decode_lnav(tail).subframes == []
    lnavs = [decode_lnav(s) for s in segs]
    with_tail = make_observables(segs + [tail], lnavs + [decode_lnav(tail)])
    assert with_tail == make_observables(segs, lnavs)
