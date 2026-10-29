from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2
from support import IONO

from pyrx.ephemeris import Ionosphere
from pyrx.models import azimuth_elevation
from pyrx.observations import Epoch, Observation
from pyrx.orbits import satellite_state
from pyrx.pvt import IONO_DEFAULT, _variance, chi2_threshold, solve, solve_epoch
from pyrx.synth import TOW, USUD, constellation, truth_epoch


def test_constellation_is_visible_from_usud():
    for eph in constellation():
        assert np.degrees(azimuth_elevation(USUD, satellite_state(eph, TOW).position)[1]) > 15


def test_solve_epoch_recovers_position_clock_and_velocity():
    ephs = constellation()
    vel, ddtr = np.array([3.0, -4.0, 1.0]), 2e-9
    ep = truth_epoch(ephs, USUD, 1.5e-4, vel, ddtr, iono=IONO)
    sol = solve_epoch(ep, ephs, IONO)
    assert sol is not None and sol.prns == (1, 2, 3, 4, 5, 6)
    np.testing.assert_allclose(sol.position, USUD, atol=1e-3, rtol=0)
    assert sol.clock_bias == pytest.approx(1.5e-4, abs=1e-12)
    np.testing.assert_allclose(sol.velocity, vel, atol=1e-3, rtol=0)
    assert sol.clock_drift == pytest.approx(ddtr, abs=1e-12)
    assert np.abs(sol.residuals).max() < 1e-3 and 0 < sol.gdop < 30
    assert (sol.week, sol.tow) == (2190, TOW)


def test_solve_epoch_without_doppler():
    ephs = constellation()
    sol = solve_epoch(truth_epoch(ephs, USUD, 0.0), ephs)
    np.testing.assert_allclose(sol.position, USUD, atol=1e-3, rtol=0)
    assert sol.velocity is None and sol.clock_drift is None


def test_zero_doppler_is_a_missing_observation():
    # RINEX writes a missing value as 0.000 and some readers keep it: a zero Doppler is
    # left out of the velocity solution, as RTKLIB leaves it out
    ephs = constellation()
    vel, ddtr = np.array([3.0, -4.0, 1.0]), 2e-9
    ep = truth_epoch(ephs, USUD, 1.5e-4, vel, ddtr)
    one = Epoch(ep.week, ep.tow, [replace(ep.obs[0], doppler=0.0)] + ep.obs[1:])
    sol = solve_epoch(one, ephs)
    np.testing.assert_allclose(sol.velocity, vel, atol=1e-3, rtol=0)
    none = Epoch(ep.week, ep.tow, [replace(o, doppler=0.0) for o in ep.obs])
    assert solve_epoch(none, ephs).velocity is None


def test_solve_epoch_without_coefficients_uses_rtklib_defaults():
    # a navigation file without GPSA/GPSB gets RTKLIB's default Klobuchar coefficients, so
    # the two solvers still agree on such files; the broadcast set, when present, is used
    ephs = constellation()
    ep = truth_epoch(ephs, USUD, 0.0, iono=IONO_DEFAULT)
    np.testing.assert_allclose(solve_epoch(ep, ephs, None).position, USUD, atol=1e-3, rtol=0)
    zeros = Ionosphere((0.0,) * 4, (0.0,) * 4)  # a file carrying all-zero lines, as RTKLIB reads it
    assert np.array_equal(solve_epoch(ep, ephs, zeros).position, solve_epoch(ep, ephs).position)
    strong = Ionosphere(tuple(3 * a for a in IONO.alpha), IONO.beta)
    ep = truth_epoch(ephs, USUD, 0.0, iono=strong)
    off = solve_epoch(ep, ephs, None).position  # the defaults leave meters, mostly up
    assert 1.0 < np.linalg.norm(off - USUD) < 30.0
    np.testing.assert_allclose(solve_epoch(ep, ephs, strong).position, USUD, atol=1e-3, rtol=0)


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
    np.testing.assert_allclose(sol.position, USUD, atol=1e-3, rtol=0)


def test_degenerate_records_and_geometry_lose_the_satellite_or_the_epoch_not_the_run():
    ephs = constellation()
    ep = truth_epoch(ephs, USUD, 0.0)
    ephs[0] = replace(ephs[0], sqrt_a=0.0)  # a blank orbit field in a navigation file
    ephs[1] = replace(ephs[1], e=float("nan"))
    sol = solve_epoch(ep, ephs)
    assert sol.prns == (3, 4, 5, 6)
    np.testing.assert_allclose(sol.position, USUD, atol=1e-3, rtol=0)
    # five copies of one observation: a singular normal matrix, no solution, no exception
    same = Epoch(ep.week, ep.tow, [ep.obs[0]] * 5)
    assert solve_epoch(same, constellation()) is None
    assert [s.tow for s in solve([same, ep], constellation())] == [TOW]


def test_solve_epoch_needs_four_satellites():
    ephs = constellation()[:3]
    assert solve_epoch(truth_epoch(ephs, USUD, 0.0), ephs) is None


def test_chi_square_threshold_is_rtklib_table():
    # RTKLIB's rounded table (alpha 0.001), not the exact quantiles, so the two solvers
    # accept and reject the same epochs at the boundaries
    assert [chi2_threshold(n) for n in (1, 2, 3, 10, 60, 100)] == [10.8, 13.8, 16.3, 29.6, 100, 149]
    assert chi2_threshold(1) < chi2.isf(0.001, 1) < 10.9


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
    np.testing.assert_allclose(
        sols[1].position - sols[0].position, [1.0, 2.0, 3.0], atol=1e-3, rtol=0
    )


def test_solve_skips_unsolvable_epochs():
    ephs = constellation()
    good = truth_epoch(ephs, USUD, 0.0)
    few = Epoch(good.week, good.tow + 1.0, good.obs[:3])
    sols = solve([good, few, good], ephs)
    assert [s.tow for s in sols] == [TOW, TOW]


def test_variance_floors_only_the_weight_term_at_five_degrees():
    # RTKLIB floors the elevation of its a priori error at 5 degrees but takes the
    # troposphere term at the raw elevation; below the mask both stay finite and positive
    def expected(el):
        s = np.sin(max(el, np.radians(5.0)))
        return 0.3**2 * (1 + 1 / s) + 2.4**2 + 0.3**2 + (0.3 / (np.sin(el) + 0.1)) ** 2

    for el in (np.radians(2.0), 0.0, -0.001, np.radians(15.0)):
        assert _variance(0, el, 0.0) == pytest.approx(expected(el), rel=1e-12)
    assert _variance(0, np.radians(2.0), 0.0) > _variance(0, np.radians(5.0), 0.0)
    assert _variance(0, np.radians(15.0), 0.0) < _variance(0, np.radians(5.0), 0.0)
