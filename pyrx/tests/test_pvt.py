from dataclasses import replace

import numpy as np
import pytest
from test_orbits import _eph

from pyrx.lnav import Ionosphere
from pyrx.models import azimuth_elevation, geodetic, geometric_range, klobuchar, saastamoinen
from pyrx.observables import C_LIGHT, Epoch, Observation
from pyrx.orbits import OMEGA_E, satellite_state
from pyrx.pvt import LAMBDA_L1, solve, solve_epoch

USUD = np.array([-3855263.0361, 3427432.5571, 3741020.2929])
TOW = 28800.0
IONO = Ionosphere(
    (1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07),
    (1.1469e05, -1.6384e05, -1.9661e05, 9.1750e05),
)


def constellation():
    # six satellites on the test orbit, spread so that all are above 15 deg at USUD at toe
    out = []
    # offsets found by search: elevations 76, 24, 28, 27, 70 and 46 deg, GDOP 3.5
    offsets = [
        (-2.31, 0.12),
        (1.83, -3.11),
        (2.54, 2.97),
        (-1.37, -1.37),
        (2.26, 1.88),
        (-1.48, -0.89),
    ]
    for k, (dm, do) in enumerate(offsets):
        out.append(_eph(prn=k + 1, m0=-1.2345678 + dm, omega0=2.1 + do, toe=TOW, toc=TOW))
    return out


def truth_epoch(ephs, xyz, dtr, vel=None, ddtr=None, week=2190, tow=TOW, iono=None, tropo=True):
    lat, lon, h = geodetic(xyz)
    obs = []
    for eph in ephs:
        t_tx = tow - dtr - 0.07  # about 70 ms of light time; refined below
        for _ in range(4):
            s = satellite_state(eph, t_tx)
            t_tx = tow - dtr - geometric_range(xyz, s.position)[0] / C_LIGHT
        s = satellite_state(eph, t_tx)
        rho, e = geometric_range(xyz, s.position)
        az, el = azimuth_elevation(xyz, s.position)
        corr = (klobuchar(iono, lat, lon, az, el, tow) if iono else 0.0) + (
            saastamoinen(lat, h, el) if tropo else 0.0
        )
        p = rho + C_LIGHT * (dtr - s.clock) + corr + C_LIGHT * eph.tgd
        d = float("nan")
        if vel is not None:
            rate = e @ (s.velocity - vel) + OMEGA_E / C_LIGHT * (
                s.velocity[1] * xyz[0]
                + s.position[1] * vel[0]
                - s.velocity[0] * xyz[1]
                - s.position[0] * vel[1]
            )
            d = -(rate + C_LIGHT * (ddtr - s.clock_drift)) / LAMBDA_L1
        obs.append(Observation(eph.prn, p, 0.0, d, 45.0))  # carrier phase unused here
    return Epoch(week, tow, obs)


def test_constellation_is_visible_from_usud():
    for eph in constellation():
        assert np.degrees(azimuth_elevation(USUD, satellite_state(eph, TOW).position)[1]) > 15


def test_solve_epoch_recovers_position_clock_and_velocity():
    ephs = constellation()
    vel, ddtr = np.array([3.0, -4.0, 1.0]), 2e-9
    ep = truth_epoch(ephs, USUD, 1.5e-4, vel, ddtr, iono=IONO)
    sol = solve_epoch(ep, ephs, IONO)
    assert sol is not None and sol.prns == (1, 2, 3, 4, 5, 6)
    np.testing.assert_allclose(sol.position, USUD, atol=1e-3)
    assert sol.clock_bias == pytest.approx(1.5e-4, abs=1e-12)
    np.testing.assert_allclose(sol.velocity, vel, atol=1e-3)
    assert sol.clock_drift == pytest.approx(ddtr, abs=1e-12)
    assert np.abs(sol.residuals).max() < 1e-3 and 0 < sol.gdop < 30
    assert (sol.week, sol.tow) == (2190, TOW)


def test_solve_epoch_without_doppler():
    ephs = constellation()
    sol = solve_epoch(truth_epoch(ephs, USUD, 0.0), ephs)
    np.testing.assert_allclose(sol.position, USUD, atol=1e-3)
    assert sol.velocity is None and sol.clock_drift is None


def test_solve_epoch_without_ionosphere():
    ephs = constellation()
    ep = truth_epoch(ephs, USUD, 0.0, iono=IONO)
    far = solve_epoch(ep, ephs, None).position  # no correction applied: a few metres off, mostly up
    assert 1.0 < np.linalg.norm(far - USUD) < 30.0
    np.testing.assert_allclose(solve_epoch(ep, ephs, IONO).position, USUD, atol=1e-3)


def test_solve_epoch_applies_the_elevation_mask():
    ephs = constellation()
    ep = truth_epoch(ephs, USUD, 0.0)
    lowest = min(ephs, key=lambda e: azimuth_elevation(USUD, satellite_state(e, TOW).position)[1])
    el = np.degrees(azimuth_elevation(USUD, satellite_state(lowest, TOW).position)[1])
    sol = solve_epoch(ep, ephs, mask_deg=el + 0.5)
    assert lowest.prn not in sol.prns and len(sol.prns) == 5


def test_solve_epoch_excludes_unhealthy_stale_and_unknown():
    ephs = constellation()
    ep = truth_epoch(ephs, USUD, 0.0)
    ephs[0] = replace(ephs[0], health=1)
    ephs[1] = replace(ephs[1], toe=TOW - 7300.0)
    ep = Epoch(ep.week, ep.tow, ep.obs + [Observation(7, 2.2e7, float("nan"), float("nan"), 40.0)])
    sol = solve_epoch(ep, ephs)
    assert sol.prns == (3, 4, 5, 6)
    np.testing.assert_allclose(sol.position, USUD, atol=1e-3)


def test_solve_epoch_needs_four_satellites():
    ephs = constellation()[:3]
    assert solve_epoch(truth_epoch(ephs, USUD, 0.0), ephs) is None


def test_solve_epoch_rejects_blunder():
    ephs = constellation()
    ep = truth_epoch(ephs, USUD, 0.0)
    o = ep.obs[2]
    bad = ep.obs[:2] + [replace(o, pseudorange=o.pseudorange + 1000.0)] + ep.obs[3:]
    assert (
        solve_epoch(Epoch(ep.week, ep.tow, bad), ephs) is None
    )  # chi-square, 2 degrees of freedom


def test_solve_converges_from_the_earth_centre_and_seeds_the_next_epoch():
    ephs = constellation()
    ep0 = truth_epoch(ephs, USUD, 1e-3)
    ep1 = truth_epoch(ephs, USUD + np.array([1.0, 2.0, 3.0]), 1e-3, tow=TOW + 1.0)
    sols = solve([ep0, ep1], ephs)
    assert len(sols) == 2
    np.testing.assert_allclose(sols[1].position - sols[0].position, [1.0, 2.0, 3.0], atol=1e-3)


def test_solve_skips_unsolvable_epochs():
    ephs = constellation()
    good = truth_epoch(ephs, USUD, 0.0)
    few = Epoch(good.week, good.tow + 1.0, good.obs[:3])
    sols = solve([good, few, good], ephs)
    assert [s.tow for s in sols] == [TOW, TOW]
