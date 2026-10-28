"""Observables of the two 30 s recordings against Pocket SDR's C tracker and an RTKLIB fix."""

import numpy as np
import pytest
from conftest import capture_file
from pocketsdr_ctrk import load_ctrk_ch
from rtklib import horizontal, parse_pos, rms, rnx2rtkp_single

from pyrx.acquisition import acquire_l1ca
from pyrx.navdecode import decode_lnav
from pyrx.observables import code_starts, make_observables, transmit_times
from pyrx.rinex import write_rinex_nav, write_rinex_obs
from pyrx.samples import read_samples
from pyrx.tracking import track_channels

# name, fs, f_if, prns, broadcast nav, Pocket SDR C log, week, Pocket SDR-based fix, Pocket SDR
# observations (None when Pocket SDR formed none). Both recordings were made at the same site,
# so the 2021 fix is the position reference for the 2022 recording too.
CASES = [
    (
        "L1_20211226_082401_24MHz_I.bin",
        24e6,
        6.0e6,
        [4, 16, 18, 26, 27, 29, 31],
        "brdc/BRDC00WRD_R_20213600000_01D_GN.rnx",
        "pocketsdr-trk/L1_20211226_082401_24MHz_I.ctrk.log",
        2190,
        "pocketsdr-trk/L1_20211226_082401_24MHz_I.psdr.rnx2rtkp.pos",
        "pocketsdr-trk/L1_20211226_082401_24MHz_I.psdr.obs",
    ),
    (
        "L1_20220114_004440_8MHz_I.bin",
        8e6,
        2.0e6,
        [10, 12, 15, 23, 24, 32],
        "brdc/BRDC00WRD_R_20220140000_01D_GN.rnx",
        "pocketsdr-trk/L1_20220114_004440_8MHz_I.ctrk.log",
        2192,
        "pocketsdr-trk/L1_20211226_082401_24MHz_I.psdr.rnx2rtkp.pos",
        None,
    ),
]
Q_SINGLE = 5  # RTKLIB solution quality flag of a single-point fix
IDS = [c[0] for c in CASES]
_CHAINS: dict = {}


def _chain(captures, case):
    """Acquire, track and decode one recording once per session."""
    name, fs, f_if, prns = case[:4]
    if name not in _CHAINS:
        path = capture_file(captures, name)
        head = read_samples(path, fs, iq=False, duration=0.010)
        seeds = []
        for prn in prns:
            a = acquire_l1ca(head, fs, f_if, prn)
            seeds.append((prn, a.code_offset, a.doppler, a.cn0))
        tracks = track_channels(path, fs, False, f_if, seeds)
        _CHAINS[name] = (tracks, [decode_lnav(r) for r in tracks])
    return _CHAINS[name]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_transmit_time_matches_pocketsdr(captures, case):
    tracks, lnavs = _chain(captures, case)
    ch = load_ctrk_ch(capture_file(captures, case[5]))
    for track, lnav in zip(tracks, lnavs):
        rows = ch[track.prn]
        rows = rows[rows[:, 4] > 0]  # tow valid
        tc, tt = code_starts(track), transmit_times(track, lnav)
        t = rows[:, 0]
        inside = (t >= tc[lnav.stream.epoch[0]]) & (t < tc[-1])
        assert inside.sum() >= 5, track.prn  # the C tracker logs one row per second
        t = t[inside]
        k = np.searchsorted(tc, t, side="right") - 1
        ours = tt[k] + (t - tc[k]) / (tc[k + 1] - tc[k]) * 1e-3
        # Pocket SDR: the code start `coff` after `time` was transmitted at `tow`
        theirs = (rows[inside, 5] - rows[inside, 1]) * 1e-3
        assert np.abs(ours - theirs).max() < 100e-9, (track.prn, np.abs(ours - theirs).max())


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_rtklib_fix(captures, tmp_path, case):
    nav_name, week, fix_name, psdr_obs_name = case[4], case[6], case[7], case[8]
    tracks, lnavs = _chain(captures, case)
    brdc = capture_file(captures, nav_name)
    epochs = make_observables(tracks, lnavs, week=week)
    obs = tmp_path / "pyrx.obs"
    write_rinex_obs(obs, epochs)
    sol, resid = rnx2rtkp_single(obs, brdc, tmp_path / "brdc")
    assert len(sol) >= 25 and all(s["q"] == Q_SINGLE and s["ns"] >= 5 for s in sol)
    ref = {s["time"]: s for s in parse_pos(capture_file(captures, fix_name)) if s["ns"] == 5}
    assert len(ref) == 7
    if psdr_obs_name is None:
        # another day at the same site: compare the mean positions
        keys = ("lat", "lon", "height")
        mean = {k: float(np.mean([s[k] for s in sol])) for k in keys}
        ref_mean = {k: float(np.mean([r[k] for r in ref.values()])) for k in keys}
        assert horizontal(mean, ref_mean) < 5.0 and abs(mean["height"] - ref_mean["height"]) < 10.0
        return
    common = [(s, ref[s["time"]]) for s in sol if s["time"] in ref]
    assert len(common) == 7
    for s, r in common:
        assert horizontal(s, r) < 5.0 and abs(s["height"] - r["height"]) < 10.0, s["time"]
    # pseudorange residuals over those epochs, ours against Pocket SDR's observations
    _, ref_resid = rnx2rtkp_single(capture_file(captures, psdr_obs_name), brdc, tmp_path / "psdr")
    ours = rms([v for prn in ref_resid for v in resid[prn][-7:]])
    theirs = rms([v for prn in ref_resid for v in ref_resid[prn][-7:]])
    assert ours < 2 * theirs, (ours, theirs)
    # our own navigation file gives a fix within the spec thresholds of the broadcast-nav one
    # (RTKLIB uses the day file's 08:00 records, nearest in toe; ours holds the 10:00 ones)
    ephs = [ln.ephemeris for ln in lnavs if ln.ephemeris is not None]
    tow1 = {ln.prn: sf.tow_s for ln in lnavs for sf in ln.subframes if sf.sf_id == 1}
    nav = tmp_path / "pyrx.nav"
    write_rinex_nav(nav, ephs, tow1, week=week)
    sol_own, _ = rnx2rtkp_single(obs, nav, tmp_path / "own")
    assert len(sol_own) == len(sol)
    for a, b in zip(sol, sol_own):
        assert horizontal(a, b) < 5.0 and abs(a["height"] - b["height"]) < 10.0, a["time"]
