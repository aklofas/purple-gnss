from dataclasses import replace

import numpy as np
from test_pvt import TOW, USUD, constellation

from pyrx.models import (
    Station,
    azimuth_elevation,
    geodetic,
    geometric_range,
    marker_from_arp,
    nmf_dry,
    zenith_hydrostatic,
)
from pyrx.observables import C_LIGHT
from pyrx.orbits import sat_id, satellite_state
from pyrx.rinex import gps_datetime
from pyrx.rtk import Rtk, RtkOptions, pair_epochs, solve_rtk
from pyrx.signals import Observations, wavelength

ROVER = USUD + np.array([5.0, -3.0, 2.0])
SIGNALS = ("G1C", "G2W")
ALL_SIGNALS = ("G1C", "G2W", "G5Q", "E1C", "E5Q")


def _galileo(ephs):
    return [replace(e, prn=e.prn + 10, system="E") for e in ephs]


def _ambiguities(rng, ephs, signals):
    return {
        (sat_id(e), s): int(rng.integers(-1000, 1000))
        for e in ephs
        for s in signals
        if s[0] == e.system
    }


_RNG = np.random.default_rng(0)
AMB_R = _ambiguities(_RNG, constellation() + _galileo(constellation()), ALL_SIGNALS)
AMB_B = _ambiguities(_RNG, constellation() + _galileo(constellation()), ALL_SIGNALS)


def _observations(ephs, xyz, dtr, tow, signals, amb, rng, phase_noise, code_noise, week):
    lat, _, h = geodetic(xyz)
    doy = gps_datetime(week, tow).timetuple().tm_yday
    ephs = sorted(ephs, key=sat_id)
    n, m = len(ephs), len(signals)
    P, L = np.full((n, m), np.nan), np.full((n, m), np.nan)
    for i, eph in enumerate(ephs):
        t_tx = tow - dtr - 0.07
        for _ in range(4):
            s = satellite_state(eph, t_tx)
            t_tx = tow - dtr - geometric_range(xyz, s.position)[0] / C_LIGHT
        s = satellite_state(eph, t_tx)
        rho, _ = geometric_range(xyz, s.position)
        _, el = azimuth_elevation(xyz, s.position)
        model = (
            rho + C_LIGHT * (dtr - s.clock) + zenith_hydrostatic(lat, h) * nmf_dry(lat, h, el, doy)
        )
        for j, sig in enumerate(signals):
            if sig[0] != eph.system:
                continue
            lam = wavelength(sig)
            L[i, j] = model / lam + amb[(sat_id(eph), sig)] + rng.normal(0.0, phase_noise) / lam
            P[i, j] = model + rng.normal(0.0, code_noise)
    return Observations(
        week,
        tow,
        tuple(sat_id(e) for e in ephs),
        tuple(signals),
        P,
        L,
        np.full((n, m), np.nan),
        np.where(np.isfinite(P), 45.0, np.nan),
        np.zeros((n, m), dtype=int),
    )


def synthetic_pair(
    ephs,
    base_xyz,
    rover_xyz,
    tow,
    signals,
    amb_rover,
    amb_base,
    rng,
    phase_noise=0.001,
    code_noise=0.3,
    week=2190,
):
    """Observations of a rover and a base at their antenna reference points from the package's models.

    The default phase noise is that of a 15 Hz PLL at 45 dB-Hz (about 0.7 mm); with it a
    fixed single-epoch position lands within 1 cm of the truth, a wrong integer 19 cm off.
    """
    args = (tow, signals, rng, phase_noise, code_noise, week)
    rover = _observations(ephs, rover_xyz, 1e-4, args[0], args[1], amb_rover, *args[2:])
    base = _observations(ephs, base_xyz, 2e-4, args[0], args[1], amb_base, *args[2:])
    return rover, base


def _pair(rng, ephs, tow, signals=SIGNALS, rover=ROVER, lli=None, drop=(), amb_r=None):
    r, b = synthetic_pair(ephs, USUD, rover, tow, signals, amb_r or AMB_R, AMB_B, rng)
    if lli:
        flags = r.lli.copy()
        for (sat, sig), flag in lli.items():
            flags[r.sats.index(sat), r.signals.index(sig)] = flag
        r = replace(r, lli=flags)
    if drop:
        P, L = b.pseudorange.copy(), b.carrier_phase.copy()
        for sat, sig in drop:
            P[b.sats.index(sat), b.signals.index(sig)] = np.nan
            L[b.sats.index(sat), b.signals.index(sig)] = np.nan
        b = replace(b, pseudorange=P, carrier_phase=L)
    return r, b


def _rtk(ephs, **options):
    return Rtk(ephs, Station(USUD, np.zeros(3)), np.zeros(3), RtkOptions(**options))


def test_pair_epochs_takes_the_latest_base_within_age():
    def mk(tow):
        z = np.zeros((0, 0))
        return Observations(2190, tow, (), (), z, z, z, z, np.zeros((0, 0), int))

    rover = [mk(0.0), mk(30.0), mk(60.0), mk(120.0)]
    base = [mk(0.0), mk(30.01), mk(59.0)]
    pairs = pair_epochs(rover, base, 30.0)
    assert [None if b is None else b.tow for _, b in pairs] == [0.0, 30.01, 59.0, None]


def test_rtk_fixes_a_synthetic_dual_frequency_pair():
    rng = np.random.default_rng(10)
    ephs = constellation()
    rtk = _rtk(ephs)
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 30.0 * k)) for k in range(4)]
    assert all(f is not None and f.ns == 6 and f.nb == 10 for f in fixes)
    first = next(i for i, f in enumerate(fixes) if f.status == "fix")
    assert first <= 2 and all(f.status == "fix" and f.ratio >= 3.0 for f in fixes[first:])
    for f in fixes[first:]:
        np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)
    assert (fixes[3].week, fixes[3].tow, fixes[3].age) == (2190, TOW + 90.0, 0.0)


def test_rtk_fixes_single_frequency_over_epochs():
    rng = np.random.default_rng(11)
    ephs = constellation()
    rtk = _rtk(ephs)
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, signals=("G1C",))) for k in range(10)]
    assert all(f is not None for f in fixes) and fixes[-1].status == "fix"
    np.testing.assert_allclose(fixes[-1].position, ROVER, atol=0.01, rtol=0)


def test_rtk_static_mode_carries_the_position():
    rng = np.random.default_rng(12)
    ephs = constellation()
    rtk = _rtk(ephs, mode="static")
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, signals=("G1C",))) for k in range(6)]
    assert fixes[-1].status == "fix"
    np.testing.assert_allclose(fixes[-1].position, ROVER, atol=0.01, rtol=0)
    # the float position improves while it is carried; in kinematic mode each epoch starts afresh
    errors = [np.linalg.norm(f.position - ROVER) for f in fixes]
    assert errors[-1] < 0.01 and errors[-1] <= errors[0]


def test_rtk_reinitialises_after_a_slip():
    rng = np.random.default_rng(13)
    ephs = constellation()
    rtk = _rtk(ephs)
    amb = dict(AMB_R)
    for k in range(3):
        fix = rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, amb_r=amb))
    assert fix.status == "fix"
    amb[("G03", "G1C")] += 7  # a cycle slip on the rover, flagged by the receiver
    for k, lli in ((3, {("G03", "G1C"): 1}), (4, None), (5, None)):
        fix = rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, lli=lli, amb_r=amb))
        assert fix is not None and fix.nb == 10
        if fix.status == "fix":  # never the stale integer: a fix is right or the epoch floats
            np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)
    assert fix.status == "fix"
    amb[("G03", "G1C")] += 7  # an unflagged slip on L1 only: the geometry-free jump must catch it
    fix = rtk.step(*_pair(rng, ephs, TOW + 180.0, amb_r=amb))
    assert fix is not None and (
        fix.status == "float" or np.linalg.norm(fix.position - ROVER) < 0.01
    )


def test_rtk_uses_only_signals_present_at_both_receivers():
    rng = np.random.default_rng(14)
    ephs = constellation()
    rtk = _rtk(ephs)
    for k in range(3):
        fix = rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, drop={("G02", "G2W"), ("G05", "G2W")}))
    assert fix.status == "fix" and fix.nb == 8 and fix.ns == 6
    np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)


def test_rtk_skips_epochs_without_a_base():
    rng = np.random.default_rng(15)
    ephs = constellation()
    rtk = _rtk(ephs)
    amb = dict(AMB_R)
    for k in range(3):
        rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, amb_r=amb))
    for k in range(3, 10):  # seven epochs without a base: every ambiguity state ages out
        assert rtk.step(_pair(rng, ephs, TOW + 30.0 * k, amb_r=amb)[0], None) is None
    amb.update({k: v + 3 for k, v in amb.items()})  # the receiver re-acquired everything meanwhile
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 300.0 + 30.0 * k, amb_r=amb)) for k in range(3)]
    assert all(f is not None and f.nb == 10 for f in fixes) and fixes[-1].status == "fix"
    for f in fixes:
        if f.status == "fix":
            np.testing.assert_allclose(
                f.position, ROVER, atol=0.01, rtol=0
            )  # recreated states, not stale


def test_rtk_drops_satellites_without_ephemeris():
    rng = np.random.default_rng(16)
    ephs = constellation()
    rtk = _rtk(ephs[1:])  # no record for G01
    for k in range(3):
        fix = rtk.step(*_pair(rng, ephs, TOW + 30.0 * k))
    assert fix.status == "fix" and fix.ns == 5 and fix.nb == 8


def test_rtk_requires_four_satellites():
    rng = np.random.default_rng(17)
    ephs = constellation()[:3]
    rtk = _rtk(ephs)
    assert rtk.step(*_pair(rng, ephs, TOW)) is None


def test_rtk_with_galileo_uses_a_reference_per_system():
    rng = np.random.default_rng(18)
    ephs = constellation() + _galileo(constellation())
    rtk = _rtk(ephs)
    signals = ("G1C", "G5Q", "E1C", "E5Q")
    for k in range(3):
        fix = rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, signals=signals))
    assert fix.status == "fix" and fix.ns == 12 and fix.nb == 20  # (6 - 1) * 2 per system
    np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)


def test_solve_rtk_reports_the_marker_and_skips_unpaired_epochs():
    rng = np.random.default_rng(19)
    ephs = constellation()
    delta = np.array([0.0, 0.0, 1.5])
    pairs = [
        synthetic_pair(ephs, USUD, ROVER, TOW + 30.0 * k, SIGNALS, AMB_R, AMB_B, rng)
        for k in range(4)
    ]
    rover = [p[0] for p in pairs]
    base = [p[1] for p in pairs[:2]] + [pairs[3][1]]
    fixes = solve_rtk(rover, base, ephs, Station(USUD, np.zeros(3)), delta)
    assert [f.tow for f in fixes] == [TOW, TOW + 30.0, TOW + 60.0, TOW + 90.0]
    assert fixes[2].age == 30.0 and fixes[-1].status == "fix"
    np.testing.assert_allclose(fixes[-1].position, marker_from_arp(ROVER, delta), atol=0.01, rtol=0)


def test_rtk_keeps_ambiguities_when_the_first_signal_is_missing_for_an_epoch():
    rng = np.random.default_rng(21)
    ephs = constellation()
    rtk = _rtk(ephs)
    signals = ("G1C", "G2W", "G5Q")
    for k in range(5):
        drop = {("G03", "G1C")} if k == 3 else ()
        fix = rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, signals=signals, drop=drop))
        assert fix is not None
    # the geometry-free test is skipped, not fooled, when L1 is absent: L2 and L5 keep their locks
    locks = {sig: rtk.biases[("G03", sig)].lock for sig in signals}
    assert locks == {"G1C": 4, "G2W": 5, "G5Q": 5}
    assert fix.status == "fix" and fix.nb == 15
    np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)


def test_rtk_resets_on_a_half_cycle_transition():
    rng = np.random.default_rng(22)
    ephs = constellation()
    rtk = _rtk(ephs)
    amb = dict(AMB_R)
    for k in range(3):
        rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, amb_r=amb, signals=("G1C",)))
    # the receiver tracks G03 with an unresolved half cycle for one epoch, then resolves it
    amb[("G03", "G1C")] += 0.5
    fix = rtk.step(
        *_pair(rng, ephs, TOW + 90.0, amb_r=amb, lli={("G03", "G1C"): 2}, signals=("G1C",))
    )
    assert fix is not None and rtk.biases[("G03", "G1C")].half
    amb[("G03", "G1C")] += 0.5
    fixes = [
        rtk.step(*_pair(rng, ephs, TOW + 120.0 + 30.0 * k, amb_r=amb, signals=("G1C",)))
        for k in range(3)
    ]
    assert (
        rtk.biases[("G03", "G1C")].lock == 3
    )  # reset at the transition, not kept half a cycle off
    assert fixes[-1].status == "fix"
    for f in fixes:
        if f.status == "fix":
            np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)
