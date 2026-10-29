from dataclasses import replace

import numpy as np
import pytest
from support import AMB_B, AMB_R, ROVER, SIGNALS

from pyrx.models import Station, marker_from_arp
from pyrx.observations import Observations
from pyrx.rtk import Rtk, RtkOptions, pair_epochs, solve_rtk
from pyrx.synth import TOW, USUD, as_galileo, constellation, synthetic_pair


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
    # seven epochs without a base: unpaired epochs leave the ambiguity states as they are
    for k in range(3, 10):
        assert rtk.step(_pair(rng, ephs, TOW + 30.0 * k, amb_r=amb)[0], None) is None
    amb.update({k: v + 3 for k, v in amb.items()})  # the receiver re-acquired everything meanwhile
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 300.0 + 30.0 * k, amb_r=amb)) for k in range(3)]
    assert all(f is not None and f.nb == 10 for f in fixes) and fixes[-1].status == "fix"
    for f in fixes:
        if f.status == "fix":
            np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)


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
    ephs = constellation() + as_galileo(constellation())
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


def test_rtk_seeds_from_l1_ca_and_needs_a_gps_signal(monkeypatch):
    # the single-point seed is an L1 solver (L1 group delay, L1 ionosphere): it takes L1 C/A
    # whatever the signal order, and a signal list without GPS is an error, not a crash
    import pyrx.rtk as rtk_module

    seen = []
    original = rtk_module.to_epoch

    def spy(obs, signal="G1C"):
        seen.append(signal)
        return original(obs, signal)

    monkeypatch.setattr(rtk_module, "to_epoch", spy)
    rng = np.random.default_rng(23)
    ephs = constellation()
    fix = _rtk(ephs).step(*_pair(rng, ephs, TOW, signals=("G5Q", "G1C")))
    assert fix is not None and seen == ["G1C"]
    galileo = as_galileo(ephs)
    with pytest.raises(ValueError, match="GPS"):
        _rtk(galileo).step(*_pair(rng, galileo, TOW, signals=("E1C", "E5Q")))


def test_a_reused_base_epoch_applies_its_slip_flag_once():
    # a 1 Hz rover against a 30 s base reuses one base epoch for 30 rover epochs; a slip the
    # base flagged resets the ambiguity at the first rover epoch that sees it, not at all 30
    rng = np.random.default_rng(24)
    ephs = constellation()
    rtk = _rtk(ephs)
    key = ("G03", "G1C")
    rover0, base = _pair(rng, ephs, TOW, signals=("G1C",))
    flags = base.lli.copy()
    flags[base.sats.index("G03"), 0] = 1
    base = replace(base, lli=flags)
    rtk.step(rover0, base)
    assert rtk.biases[key].lock == 1  # flagged at the base: a fresh state
    for k in range(1, 4):
        rover, _ = _pair(rng, ephs, TOW + 1.0 * k, signals=("G1C",))
        rtk.step(rover, base)
    assert rtk.biases[key].lock == 4


def _one_hz_run(rng, ephs, rtk, base_every, seconds, amb_r=AMB_R, amb_b=AMB_B, edit=None):
    """Fixes by rover time of a 1 Hz L1-only rover against a base every `base_every` s;
    `edit` may replace the (rover, base) pair of a rover time before it is stepped."""
    rover, base = [], []
    for k in range(seconds):
        r, b = synthetic_pair(ephs, USUD, ROVER, TOW + 1.0 * k, ("G1C",), amb_r, amb_b, rng)
        if edit is not None:
            r, b = edit(k, r, b)
        rover.append(r)
        if k % base_every == 0:
            base.append(b)
    fixes = {}
    for r, b in pair_epochs(rover, base, rtk.opt.max_age):
        f = rtk.step(r, b)
        if f is not None:
            fixes[r.tow - TOW] = f
    return fixes


def _blank_phase(obs, sat, sig):
    L = obs.carrier_phase.copy()
    L[obs.sats.index(sat), obs.signals.index(sig)] = np.nan
    return replace(obs, carrier_phase=L)


def _flag(obs, sat, sig, flag):
    lli = obs.lli.copy()
    lli[obs.sats.index(sat), obs.signals.index(sig)] = flag
    return replace(obs, lli=lli)


def test_a_base_slip_is_applied_while_the_rover_phase_is_missing():
    # 1 Hz rover, 5 s base, L1 only (no second signal for the geometry-free test to
    # catch it): the base epoch at 60 s flags a slip on G03 and the rover's G03 phase
    # is missing at that one epoch (code kept); the slip must reset the ambiguity at
    # the base epoch's first use, as RTKLIB's per-receiver test does, not be lost
    # because the rover had no phase to pair with
    rng = np.random.default_rng(30)
    ephs = constellation()
    rtk = _rtk(ephs)
    key = ("G03", "G1C")
    amb_b = dict(AMB_B)

    def edit(k, r, b):
        if k == 60:
            amb_b[key] += 7
            r, b = synthetic_pair(ephs, USUD, ROVER, TOW + 1.0 * k, ("G1C",), AMB_R, amb_b, rng)
            r, b = _blank_phase(r, *key), _flag(b, *key, 1)
        return r, b

    locks = {}
    original = rtk.step

    def step(r, b):
        f = original(r, b)
        locks[r.tow - TOW] = rtk.biases[key].lock if key in rtk.biases else None
        return f

    rtk.step = step
    fixes = _one_hz_run(rng, ephs, rtk, 5, 75, amb_b=amb_b, edit=edit)
    assert fixes[59.0].status == "fix" and locks[59.0] == 60
    assert locks[61.0] <= 2  # a fresh state once the rover phase returns
    assert np.linalg.norm(fixes[61.0].position - ROVER) < 1.0
    assert fixes[74.0].status == "fix"
    np.testing.assert_allclose(fixes[74.0].position, ROVER, atol=0.01, rtol=0)


def test_a_rover_slip_is_applied_while_the_base_phase_is_missing():
    # the mirror: both receivers at 1 Hz, the rover flags a slip on G03 at 60 s while the
    # base has no G03 phase at that epoch; the next epoch carries no flag, so the slip
    # is lost unless the rover's own indicator was applied when it came
    rng = np.random.default_rng(31)
    ephs = constellation()
    rtk = _rtk(ephs)
    key = ("G03", "G1C")
    amb_r = dict(AMB_R)

    def edit(k, r, b):
        if k == 60:
            amb_r[key] += 7
            r, b = synthetic_pair(ephs, USUD, ROVER, TOW + 1.0 * k, ("G1C",), amb_r, AMB_B, rng)
            r, b = _flag(r, *key, 1), _blank_phase(b, *key)
        return r, b

    locks = {}
    original = rtk.step

    def step(r, b):
        f = original(r, b)
        locks[r.tow - TOW] = rtk.biases[key].lock if key in rtk.biases else None
        return f

    rtk.step = step
    fixes = _one_hz_run(rng, ephs, rtk, 1, 75, amb_r=amb_r, edit=edit)
    assert fixes[59.0].status == "fix" and locks[59.0] == 60
    assert locks[61.0] <= 2
    assert np.linalg.norm(fixes[61.0].position - ROVER) < 1.0
    assert fixes[74.0].status == "fix"
    np.testing.assert_allclose(fixes[74.0].position, ROVER, atol=0.01, rtol=0)


def test_ambiguities_survive_a_base_gap_longer_than_the_outage_limit():
    # 1 Hz rover, 30 s base with one epoch missing: the rover epochs that get no base (or
    # no seed) never reach the relative solution, so they must not count as outages; the
    # ambiguities are kept and the fix resumes at once when the base returns
    rng = np.random.default_rng(25)
    ephs = constellation()
    station = Station(USUD, np.zeros(3))
    rover, base = [], []
    for k in range(90):
        r, b = synthetic_pair(ephs, USUD, ROVER, TOW + 1.0 * k, SIGNALS, AMB_R, AMB_B, rng)
        rover.append(r)
        if k in (0, 60):  # the base epoch at 30 s is missing
            base.append(b)
    rtk = Rtk(ephs, station, np.zeros(3))
    fixes = {}
    for r, b in pair_epochs(rover, base, rtk.opt.max_age):
        f = rtk.step(r, b)
        if f is not None:
            fixes[r.tow - TOW] = f
    assert sorted(fixes) == [*range(31), *range(60, 90)]
    assert all(rtk.biases[k].lock == 61 for k in rtk.biases)  # one per relative epoch
    assert fixes[60.0].status == "fix"
    np.testing.assert_allclose(fixes[60.0].position, ROVER, atol=0.01, rtol=0)


def test_a_base_epoch_sharing_no_satellite_is_not_an_outage():
    rng = np.random.default_rng(29)
    ephs = constellation()
    rtk = _rtk(ephs)
    rtk.step(*_pair(rng, ephs, TOW, signals=("G1C",)))
    rover, base = _pair(rng, ephs, TOW + 30.0, signals=("G1C",))
    nothing = replace(base, pseudorange=np.full_like(base.pseudorange, np.nan))
    assert rtk.step(rover, nothing) is None
    assert all(b.outc == 0 for b in rtk.biases.values())


def test_an_epoch_with_fewer_than_four_satellites_still_updates_the_filter():
    # RTKLIB commits the float update and the lock counts, then reports no solution
    rng = np.random.default_rng(26)
    ephs = constellation()
    rtk = _rtk(ephs)
    for k in range(2):
        rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, signals=("G1C",)))
    drop = {(f"G{p:02d}", "G1C") for p in (4, 5, 6)}
    assert rtk.step(*_pair(rng, ephs, TOW + 60.0, signals=("G1C",), drop=drop)) is None
    assert {k: b.lock for k, b in rtk.biases.items()} == {
        ("G01", "G1C"): 3, ("G02", "G1C"): 3, ("G03", "G1C"): 3,
        ("G04", "G1C"): 2, ("G05", "G1C"): 2, ("G06", "G1C"): 2,
    }  # fmt: skip


def test_ratio_is_reported_as_zero_when_the_best_candidate_has_no_residual(monkeypatch):
    # RTKLIB accepts the fix when the best squared norm is zero and reports ratio 0.0
    import pyrx.rtk as rtk_module

    original = rtk_module.lambda_fix
    rng = np.random.default_rng(27)
    ephs = constellation()
    best = []

    def zero_norm(y, Q, m):
        cands = original(y, Q, m)[0]
        best.append(cands[0])
        return cands, [0.0, 1.0]

    monkeypatch.setattr(rtk_module, "lambda_fix", zero_norm)
    fix = _rtk(ephs).step(*_pair(rng, ephs, TOW))
    assert fix.status == "fix" and fix.ratio == 0.0
    assert [a.integer for a in fix.ambiguities] == [round(v) for v in best[0]]


def test_zero_innovation_limit_means_no_limit():
    rng = np.random.default_rng(28)
    ephs = constellation()
    fix = _rtk(ephs, max_innovation=0.0).step(*_pair(rng, ephs, TOW))
    assert fix is not None and fix.nb == 10


def _truth(a):
    """The double-differenced integer ambiguity of a record: reference minus satellite."""
    return (AMB_R[(a.ref, a.signal)] - AMB_B[(a.ref, a.signal)]) - (
        AMB_R[(a.sat, a.signal)] - AMB_B[(a.sat, a.signal)]
    )


def test_fix_reports_the_float_state_and_the_ambiguities():
    rng = np.random.default_rng(1)
    ephs = constellation()
    rtk = _rtk(ephs)
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 30.0 * k)) for k in range(10)]
    fix = fixes[-1]
    assert fix.status == "fix"
    assert fix.float_position.shape == (3,) and fix.position_std.shape == (3,)
    assert np.all(fix.position_std > 0)
    assert np.linalg.norm(fix.float_position - fix.position) < 1.0
    assert len(fix.ambiguities) == fix.nb >= 2
    # the float estimates close in on the integers as the filter accumulates epochs
    for a in fix.ambiguities:
        assert a.ref != a.sat and a.signal in SIGNALS and a.std > 0
        assert a.integer is not None and abs(a.value - a.integer) < min(0.5, 3 * a.std)
        assert a.integer == _truth(a)  # the labels, the sign and the integer itself
    assert max(abs(a.value - a.integer) for a in fix.ambiguities) < max(
        abs(a.value - a.integer) for a in fixes[0].ambiguities
    )


def test_float_epoch_has_no_integers():
    rng = np.random.default_rng(1)
    ephs = constellation()
    rtk = _rtk(ephs, ratio_threshold=999.0)
    fix = rtk.step(*_pair(rng, ephs, TOW))
    assert fix.status == "float"
    assert len(fix.ambiguities) == fix.nb >= 2
    assert all(a.integer is None for a in fix.ambiguities)
    np.testing.assert_array_equal(fix.position, fix.float_position)
    np.testing.assert_array_equal(fix.position_std, np.sqrt(np.diag(rtk.P[:3, :3])))
    kept = fix.position.copy(), fix.float_position.copy(), fix.position_std.copy()
    rtk.step(*_pair(rng, ephs, TOW + 30.0))  # the next epoch resets the state in place
    for a, b in zip(kept, (fix.position, fix.float_position, fix.position_std)):
        np.testing.assert_array_equal(a, b)


def test_one_eligible_pair_is_reported_as_a_float_ambiguity():
    # half-cycle flags keep four of the six satellites out of the search, not out of the
    # float update: one pair is left, too few for LAMBDA
    rng = np.random.default_rng(1)
    ephs = constellation()
    halves = {(sat, "G1C"): 2 for sat in ("G03", "G04", "G05", "G06")}
    fix = _rtk(ephs).step(*_pair(rng, ephs, TOW, signals=("G1C",), lli=halves))
    assert fix is not None and fix.status == "float" and fix.ns == 6
    assert fix.nb == len(fix.ambiguities) == 1
    (a,) = fix.ambiguities
    assert (a.ref, a.sat, a.signal, a.integer) == ("G01", "G02", "G1C", None)
    assert a.std > 0 and abs(a.value - _truth(a)) < 3 * a.std
