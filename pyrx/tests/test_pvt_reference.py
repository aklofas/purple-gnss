"""Our standalone fix against RTKLIB's, on our recordings and on a day of the IGS station USUD."""

import numpy as np
import pymap3d as pm
import pytest
from conftest import capture_file
from rtklib import rnx2rtkp_models
from test_observables_reference import CASES, _chain

from pyrx.models import geodetic
from pyrx.observables import make_observables
from pyrx.pvt import solve
from pyrx.rinex import read_rinex_nav, read_rinex_obs, write_rinex_obs

USUD_OBS = "igs/USUD00JPN_R_20213600000_01D_30S_MO.crx.gz"
USUD_NAV = "igs/USUD00JPN_R_20213600000_01D_GN.rnx.gz"
USUD_XYZ = np.array([-3855263.0361, 3427432.5571, 3741020.2929])  # ITRF, IGS site log
# navigation file with ionosphere coefficients per recording (the worldwide GPS-only files have none)
NAVS = {2190: USUD_NAV, 2192: "brdc/BRDC00IGS_R_20220140000_01D_MN.rnx"}


def _key(sol):
    return (sol.week, round(sol.tow, 1))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_fix_matches_rtklib_on_our_recordings(captures, tmp_path, case):
    week = case[6]
    tracks, lnavs = _chain(captures, case)
    epochs = make_observables(tracks, lnavs, week=week)
    obs = tmp_path / "pyrx.obs"
    write_rinex_obs(obs, epochs)
    nav = capture_file(captures, NAVS[week])
    ephs, iono = read_rinex_nav(nav)
    assert iono is not None
    ours = solve(epochs, ephs, iono)
    ref, vel = rnx2rtkp_models(obs, nav, tmp_path / "rtk")
    assert len(ours) == len(epochs) and len(ref) == len(epochs)
    for s in ours:
        r = ref[_key(s)]
        assert r["q"] == 5 and r["ns"] == len(s.prns) >= 5, s.tow
        assert np.linalg.norm(s.position - r["xyz"]) < 1.0, (s.tow, s.position - r["xyz"])
        if week == 2190:  # our Doppler is in the file; USUD's is not
            lat, lon, _ = geodetic(s.position)
            enu = np.array(pm.ecef2enuv(*s.velocity, lat, lon, deg=False))
            assert np.linalg.norm(enu - vel[_key(s)]) < 0.1, s.tow


def test_fix_matches_rtklib_and_the_marker_on_usud(captures, tmp_path):
    obs = capture_file(captures, USUD_OBS)
    nav = capture_file(captures, USUD_NAV)
    epochs = read_rinex_obs(obs)
    assert 2870 <= len(epochs) <= 2880
    ephs, iono = read_rinex_nav(nav)
    ours = solve(epochs, ephs, iono)
    ref, _ = rnx2rtkp_models(obs, nav, tmp_path / "rtk")
    assert len(ours) >= 2870 and len(ref) >= 2870
    common = [(s, ref[_key(s)]) for s in ours if _key(s) in ref]
    assert len(common) >= 2860
    # a satellite crossing the mask within the elevation resolution of a 1 m position
    # difference may be counted by one solver and not the other: allow two such epochs
    mismatched = [s.tow for s, r in common if r["ns"] != len(s.prns)]
    assert len(mismatched) <= 2, mismatched
    for s, r in common:
        bound = 1.0 if r["ns"] == len(s.prns) else 5.0
        assert np.linalg.norm(s.position - r["xyz"]) < bound, (s.tow, s.position - r["xyz"])
    # same models number for number: the two solvers differ by 0.08 mm at the 99th percentile
    # (a model applied differently shifts every epoch by more) and by 0.9 mm in the mean,
    # which a handful of epochs up to 0.5 m apart set
    d = np.array(
        [np.linalg.norm(s.position - r["xyz"]) for s, r in common if r["ns"] == len(s.prns)]
    )
    assert d.mean() < 2e-3 and np.percentile(d, 99) < 1e-3, (d.mean(), np.percentile(d, 99))
    assert all(s.velocity is None for s in ours)
    mean = np.mean([s.position for s in ours], axis=0)
    assert np.linalg.norm(mean - USUD_XYZ) < 5.0, mean - USUD_XYZ
