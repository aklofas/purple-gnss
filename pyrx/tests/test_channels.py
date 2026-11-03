import numpy as np
import pytest

from pyrx import channels
from pyrx.acquisition import AcqResult
from pyrx.channels import track_prn, track_prns
from pyrx.navdecode import decode_lnav, ephemerides_by_prn

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
    assert [(s.prn, s.t[0]) for s in one] == [(s.prn, s.t[0]) for s in two]
    for s1, s2 in zip(one, two):
        np.testing.assert_array_equal(s1.doppler, s2.doppler)
    assert [s.prn for s in one] == [4, 4]  # PRN 9 is not in the file


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
    def always(samples, fs, f_if, prn, **kw):
        return AcqResult(prn, 0.0005, 0.0, 36.0, True, np.zeros((1, 1), np.float32), np.zeros(1))

    tracked = []
    real = channels.track_l1ca

    def recording(*a, **kw):
        tracked.append(kw["t_start"])
        return real(*a, **kw)

    monkeypatch.setattr(channels, "acquire_l1ca", always)
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


def test_ephemeris_assembles_across_the_gap(gap_file):
    segs = track_prn(gap_file, FS, True, 0.0, 4, interval=5.0)
    lnavs = [decode_lnav(s) for s in segs]
    assert [[sf.sf_id for sf in ln.subframes] for ln in lnavs] == [[1, 2], [3]]
    assert all(ln.ephemeris is None for ln in lnavs)
    assert ephemerides_by_prn(lnavs)[4].iode == 149
