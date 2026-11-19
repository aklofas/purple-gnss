from dataclasses import replace

import numpy as np
import pytest
from support import AMB_B, AMB_R, ROVER, SIGNALS

from pyrx.models import Station, marker_from_arp
from pyrx.observations import Observations
from pyrx.rtk import Rtk, RtkOptions, _moved, pair_epochs, solve_rtk
from pyrx.signals import wavelength
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


def test_rover_epochs_before_the_first_base_epoch_are_paired_with_it_within_age():
    # RTKLIB's forward pairing starts on the first base epoch: a base file that begins
    # late still serves the rover epochs up to max_age before it, at a negative age
    def mk(tow):
        z = np.zeros((0, 0))
        return Observations(2190, tow, (), (), z, z, z, z, np.zeros((0, 0), int))

    pairs = pair_epochs([mk(t) for t in (0.0, 5.0, 20.0, 35.0, 40.0)], [mk(35.0), mk(40.0)], 30.0)
    assert [None if b is None else b.tow for _, b in pairs] == [None, 35.0, 35.0, 35.0, 40.0]
    assert pair_epochs([mk(0.0)], [], 30.0)[0][1] is None
    # and the engine solves them: the base's first three epochs are missing
    ephs = constellation()
    pairs = _l1_run(np.random.default_rng(31), ephs, 12)
    rover, base = [r for r, _ in pairs], [b for _, b in pairs[3:]]
    fixes = solve_rtk(rover, base, ephs, Station(USUD, np.zeros(3)), np.zeros(3))
    assert [f.age for f in fixes] == [-3.0, -2.0, -1.0] + [0.0] * 9
    assert fixes[-1].status == "fix"
    for f in fixes:
        if f.status == "fix":
            np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)


@pytest.mark.parametrize("tows", [(30.0, 0.0, 60.0), (0.0, 30.0, 30.0)])
@pytest.mark.parametrize("which", ["rover", "base"])
def test_pair_epochs_refuses_epochs_out_of_order_or_repeated(tows, which):
    # the pairing walks both lists once, forward: out of order it would pair with the wrong
    # base epoch and lose slips without a word
    def mk(tow):
        z = np.zeros((0, 0))
        return Observations(2190, tow, (), (), z, z, z, z, np.zeros((0, 0), int))

    good, bad = [mk(0.0), mk(30.0), mk(60.0)], [mk(t) for t in tows]
    rover, base = (bad, good) if which == "rover" else (good, bad)
    with pytest.raises(ValueError, match=f"{which} epochs"):
        pair_epochs(rover, base, 30.0)


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


def test_a_slip_flagged_in_a_skipped_base_epoch_is_carried_to_the_next_paired_one():
    # the inverse rates: a 5 s rover against a 1 Hz base, L1 only. The base flags a slip
    # on G03 at 61 s, an epoch no rover epoch pairs with, and its ambiguity is 7 cycles
    # off from then on; the base epoch at 65 s carries no flag of its own
    rng = np.random.default_rng(32)
    ephs = constellation()
    rtk = _rtk(ephs)
    key = ("G03", "G1C")
    amb_b = dict(AMB_B)
    rover, base = [], []
    for k in range(100):
        if k == 61:
            amb_b[key] += 7
        r, b = synthetic_pair(ephs, USUD, ROVER, TOW + 1.0 * k, ("G1C",), AMB_R, amb_b, rng)
        base.append(_flag(b, *key, 1) if k == 61 else b)
        if k % 5 == 0:
            rover.append(r)
    fixes, locks = {}, {}
    for r, b in pair_epochs(rover, base, rtk.opt.max_age):
        fixes[r.tow - TOW] = rtk.step(r, b)
        locks[r.tow - TOW] = rtk.biases[key].lock
    assert fixes[60.0].status == "fix" and locks[60.0] == 13
    assert locks[65.0] == 1  # a fresh state at the first paired base epoch after the slip
    assert fixes[95.0].status == "fix"
    np.testing.assert_allclose(fixes[95.0].position, ROVER, atol=0.01, rtol=0)


def test_pair_epochs_carries_skipped_slip_events_once_and_only_onto_a_phase():
    # base epochs every second, rover epochs at 0, 5 and 10 s: the flags at 1 s and the
    # half-cycle bit set at 2 s and cleared at 3 s are events of skipped epochs. G03 has
    # no phase at 5 s, so its event waits for the paired epoch at 10 s
    rng = np.random.default_rng(33)
    ephs = constellation()
    rover, base = [], []
    for k in range(11):
        r, b = _pair(rng, ephs, TOW + 1.0 * k, signals=("G1C",))
        rover.append(r)
        base.append(b)
    i3, i5, i6 = (base[0].sats.index(s) for s in ("G03", "G05", "G06"))
    base[1] = _flag(_flag(base[1], "G03", "G1C", 1), "G05", "G1C", 1)
    base[2] = _flag(base[2], "G06", "G1C", 2)
    base[5] = _blank_phase(base[5], "G03", "G1C")
    pairs = pair_epochs([rover[0], rover[5], rover[10]], base, 30.0)
    assert pairs[0][1] is base[0]
    carried = pairs[1][1].lli[:, 0]
    assert carried[i5] == 1 and carried[i6] == 1 and carried.sum() == 2
    assert not base[5].lli.any()  # the input epochs are left as they were
    assert pairs[2][1].lli[i3, 0] == 1 and pairs[2][1].lli.sum() == 1


@pytest.mark.parametrize(
    ("half_cycle_epochs", "at_0", "at_5"),
    [
        (range(2, 11), 0, 3),  # the bit comes on in a skipped epoch and stays: its own 2, carried 1
        (range(3), 2, 1),  # on from the start, it goes off in a skipped epoch: carried 1
        (range(11), 2, 2),  # on throughout: no change, nothing carried
    ],
)
def test_each_edge_of_the_half_cycle_bit_in_a_skipped_epoch_is_carried(half_cycle_epochs, at_0, at_5):
    rng = np.random.default_rng(35)
    ephs = constellation()
    rover, base = [], []
    for k in range(11):
        r, b = _pair(rng, ephs, TOW + 1.0 * k, signals=("G1C",))
        rover.append(r)
        base.append(_flag(b, "G06", "G1C", 2) if k in half_cycle_epochs else b)
    i6 = base[0].sats.index("G06")
    pairs = pair_epochs([rover[0], rover[5], rover[10]], base, 30.0)
    assert [int(b.lli[i6, 0]) for _, b in pairs[:2]] == [at_0, at_5]
    assert all(int(b.lli.sum()) == b.lli[i6, 0] for _, b in pairs)  # no other satellite touched


def _run_with_a_slip_before_an_epoch_without_a_solution(spoil):
    """A 1 Hz base against a rover every 5 s: the base slips G03 by 7 cycles at 61 s, flagged
    there only, and `spoil` edits the pair at 65 s, the epoch that slip is carried to. Returns
    the fixes by rover time and the filter."""
    rng = np.random.default_rng(32)
    ephs = constellation()
    rtk = _rtk(ephs)
    amb_b = dict(AMB_B)
    rover, base = [], []
    for k in range(100):
        if k == 61:
            amb_b["G03", "G1C"] += 7
        r, b = synthetic_pair(ephs, USUD, ROVER, TOW + 1.0 * k, ("G1C",), AMB_R, amb_b, rng)
        if k == 65:
            r, b = spoil(r, b)
        base.append(_flag(b, "G03", "G1C", 1) if k == 61 else b)
        if k % 5 == 0:
            rover.append(r)
    fixes = {}
    for r, b in pair_epochs(rover, base, rtk.opt.max_age):
        fixes[r.tow - TOW] = rtk.step(r, b)
    return fixes, rtk


def _blunder(r, b):  # 500 m on one pseudorange: the single-point seed is rejected
    P = r.pseudorange.copy()
    P[r.sats.index("G05"), 0] += 500.0
    return replace(r, pseudorange=P), b


def _no_common_satellite(r, b):
    return r, replace(b, pseudorange=np.full_like(b.pseudorange, np.nan))


@pytest.mark.parametrize("spoil", [_blunder, _no_common_satellite])
def test_a_slip_at_an_epoch_without_a_solution_is_applied_at_the_next(spoil):
    # the epoch at 65 s ends before the ambiguities are looked at, and the base epochs after
    # it carry no flag: the slip has to wait for the epoch at 70 s, or G03's ambiguity stays
    # 7 cycles off and pulls the float solution tens of meters away
    fixes, rtk = _run_with_a_slip_before_an_epoch_without_a_solution(spoil)
    assert fixes[60.0].status == "fix" and fixes[65.0] is None
    assert rtk.biases["G03", "G1C"].lock == 6  # started again at 70 s, kept to 95 s
    assert all(rtk.biases[k].lock == 19 for k in rtk.biases if k[0] != "G03")
    for t in (70.0, 75.0, 95.0):
        np.testing.assert_allclose(fixes[t].position, ROVER, atol=0.05, rtol=0)
    assert fixes[95.0].status == "fix"


def test_a_change_of_the_half_cycle_bit_at_an_epoch_without_a_solution_is_held():
    # the half-cycle bit comes on at an epoch that ends early and stays: the next epoch sees
    # no change of the bit, so the slip it stands for is the held one or none
    ephs = constellation()
    key = ("G03", "G1C")
    pairs = _l1_run(np.random.default_rng(34), ephs, 12, lli_at=lambda k: {key: 2} if k >= 6 else None)
    pairs[6] = _blunder(*pairs[6])
    rtk = _rtk(ephs)
    for k, pair in enumerate(pairs):
        assert (rtk.step(*pair) is None) == (k == 6)
    assert (rtk.biases[key].lock, rtk.biases[key].half) == (5, True)
    assert rtk.biases[("G02", "G1C")].lock == 11


def test_a_held_slip_ends_the_ambiguity_of_a_satellite_absent_at_the_next_epoch():
    # flagged at an epoch that ends early, without a base observation at the next, back
    # unflagged after that: the ambiguity goes at the epoch that applies the held slips,
    # with or without the satellite, and does not live through the gap
    ephs = constellation()
    key = ("G03", "G1C")
    pairs = _l1_run(
        np.random.default_rng(35), ephs, 12, amb_at=_offset(key, 7.0, 6),
        lli_at=lambda k: {key: 1} if k == 6 else None, drop_at=lambda k: {key} if k == 7 else (),
    )  # fmt: skip
    pairs[6] = _blunder(*pairs[6])
    rtk = _six(ephs)
    for k, pair in enumerate(pairs):
        fix = rtk.step(*pair)
        assert (fix is None) == (k == 6)
        assert (key in rtk.biases) == (k != 7)
        if fix is not None and fix.status == "fix":
            np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)
    assert rtk.biases[key].lock == 4 and fix.status == "fix"


def test_a_rover_slip_at_an_epoch_without_a_base_is_applied_at_the_next():
    rng = np.random.default_rng(34)
    ephs = constellation()
    rtk = _rtk(ephs)
    amb_r = dict(AMB_R)
    for k in range(12):
        if k == 10:
            amb_r["G03", "G1C"] += 7
        r, b = _pair(rng, ephs, TOW + 30.0 * k, signals=("G1C",), amb_r=amb_r)
        if k == 10:  # the rover flags the slip in an epoch that has no base
            assert rtk.step(_flag(r, "G03", "G1C", 1), None) is None
        else:
            fix = rtk.step(r, b)
    assert rtk.biases["G03", "G1C"].lock == 1
    np.testing.assert_allclose(fix.position, ROVER, atol=0.05, rtol=0)


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
    # a phase 200 cycles (38 m) off: beyond the default limit its rows are dropped and the
    # ambiguity is not used; without a limit the rows are taken, whatever they do
    key = ("G03", "G1C")
    pairs = _l1_run(np.random.default_rng(28), ephs, 8, amb_at=_offset(key, 200.0, 6))
    for limit, lock, ns in ((30.0, 6, 5), (0.0, 8, 6)):
        rtk = _rtk(ephs, max_innovation=limit)
        fixes = [rtk.step(*pair) for pair in pairs]
        assert (rtk.biases[key].lock, fixes[-1].ns) == (lock, ns), limit


@pytest.mark.parametrize("sat", ["G01", "G03"])
def test_a_dropped_phase_row_is_no_use_of_its_ambiguity(sat):
    # RTKLIB's vsat: the lock, outage and satellite counts follow the phase rows that pass
    # the innovation limit. One L1 phase jumps 200 cycles (38 m) without a flag, its code
    # stays: the satellite's phase rows are dropped, its ambiguity ages out after
    # max_outage epochs and begins again on the phase as it is. On the highest satellite,
    # the reference of every double difference, no phase row is left and the epochs have
    # no solution; the code rows alone once kept every count up and the epochs fixed,
    # decimeters to meters off
    ephs = constellation()
    key = (sat, "G1C")
    pairs = _l1_run(np.random.default_rng(5), ephs, 14, amb_at=_offset(key, 200.0, 6))
    rtk = _rtk(ephs)
    fixes, locks = [], []
    for pair in pairs:
        fixes.append(rtk.step(*pair))
        locks.append(rtk.biases[key].lock)
    reference = sat == "G01"
    assert [f is None for f in fixes] == [False] * 6 + [reference] * 5 + [False] * 3
    assert locks == [1, 2, 3, 4, 5, 6] + [6] * 5 + [1, 2, 3]
    for f in fixes[6:11]:
        assert reference or (f.ns == 5 and f.nb == 4 and not _searched(f, sat))
    for f in fixes:
        if f is not None and f.status == "fix":
            np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)
    assert (fixes[-1].status, fixes[-1].ns, fixes[-1].nb) == ("fix", 6, 5)


def test_a_flag_is_read_while_the_satellite_is_under_the_mask_at_the_rover_alone():
    # RTKLIB selects the satellites above the mask at the base and reads the indicators of
    # all of them; the rover's own mask only keeps rows out of the update. For three epochs
    # the rover is 40 km off, where the lowest satellite stands under the mask, and the
    # rover flags a slip of seven cycles there. Back above the mask the phase carries no
    # flag: the ambiguity must have restarted in between, or it is seven cycles off
    from pyrx.models import azimuth_elevation, geodetic
    from pyrx.orbits import satellite_state

    ephs = constellation()
    key = ("G02", "G1C")
    lat, lon, _ = llh = geodetic(USUD)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])

    def elevations(xyz, tow):
        at = [satellite_state(e, tow - 0.07).position for e in ephs]
        return np.degrees([azimuth_elevation(xyz, p, geodetic(xyz))[1] for p in at])

    az = azimuth_elevation(USUD, satellite_state(ephs[1], TOW - 0.07).position, llh)[0]
    far = USUD - 40e3 * (np.sin(az) * east + np.cos(az) * north)
    mask = 24.0
    for k in range(12):
        near_els, far_els = elevations(ROVER, TOW + k), elevations(far, TOW + k)
        assert near_els.min() > mask + 0.1 and elevations(USUD, TOW + k).min() > mask + 0.1
        assert far_els[1] < mask - 0.1 and np.delete(far_els, 1).min() > mask + 0.1
    rng = np.random.default_rng(29)
    rtk = _six(ephs, mask_deg=mask)
    fixes = []
    for k in range(12):
        away = 6 <= k < 9
        r, b = _pair(
            rng, ephs, TOW + 1.0 * k, signals=("G1C",), rover=far if away else ROVER,
            amb_r=_offset(key, 7.0, 7)(k), lli={key: 1} if k == 7 else None,
        )  # fmt: skip
        fixes.append(rtk.step(r, b))
        assert fixes[-1].ns == (5 if away else 6)
        if fixes[-1].status == "fix":
            where = far if away else ROVER
            np.testing.assert_allclose(fixes[-1].position, where, atol=0.01, rtol=0)
    assert rtk.biases[key].lock == 3 and fixes[-1].status == "fix"


def test_a_phase_without_its_code_keeps_the_row_of_its_ambiguity():
    # RTKLIB asks a phase row for the two phases alone. From the seventh epoch one rover
    # code is missing: the ambiguity it has stays in the update and in the search, where it
    # once went unused and was dropped after max_outage epochs for good. Only a new
    # ambiguity needs the code, to begin on phase minus code
    ephs = constellation()
    key = ("G03", "G2W")

    def run(first):
        pairs = _l1_run(np.random.default_rng(30), ephs, 14, signals=SIGNALS)
        for k in range(first, 14):
            r, b = pairs[k]
            code = r.pseudorange.copy()
            code[r.sats.index(key[0]), r.signals.index(key[1])] = np.nan
            pairs[k] = (replace(r, pseudorange=code), b)
        rtk = _rtk(ephs)
        fixes = [rtk.step(*pair) for pair in pairs]
        for f in fixes:
            if f.status == "fix":
                np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)
        assert fixes[-1].status == "fix"
        return rtk, [f.nb for f in fixes]

    rtk, nb = run(6)
    assert nb == [10] * 14 and rtk.biases[key].lock == 14
    rtk, nb = run(0)
    assert nb == [9] * 14 and key not in rtk.biases


@pytest.mark.parametrize("slips", ["restart", "check"])
def test_a_phase_without_its_code_goes_with_the_ambiguity_a_step_takes_away(slips):
    # the first signal's code is missing at the epoch at which the second signal steps
    # five cycles without a flag: the geometry-free step ends both ambiguities, and the
    # first signal's phase, which rested on the one it had, has no code to begin another on
    ephs = constellation()
    first, second = ("G03", "G1C"), ("G03", "G2W")
    pairs = _l1_run(np.random.default_rng(30), ephs, 10, amb_at=_offset(second, 5.0, 6), signals=SIGNALS)
    r, b = pairs[6]
    code = r.pseudorange.copy()
    code[r.sats.index("G03"), r.signals.index("G1C")] = np.nan
    pairs[6] = (replace(r, pseudorange=code), b)
    rtk = _six(ephs, slips=slips)
    for k, pair in enumerate(pairs):
        fix = rtk.step(*pair)
        assert (first in rtk.biases) == (k != 6)
        if fix.status == "fix":
            np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)
    assert (rtk.biases[first].lock, rtk.biases[second].lock) == (3, 4) and fix.status == "fix"


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


def _l1_run(rng, ephs, seconds, amb_at=None, lli_at=None, signals=("G1C",), drop_at=None):
    """A 1 Hz run (L1 alone unless `signals` says otherwise) as (rover, base) pairs;
    `amb_at(k)` gives the rover ambiguities of second k, `lli_at(k)` its indicators and
    `drop_at(k)` the base observations missing."""
    return [
        _pair(
            rng, ephs, TOW + 1.0 * k, signals=signals,
            amb_r=amb_at(k) if amb_at else None, lli=lli_at(k) if lli_at else None,
            drop=drop_at(k) if drop_at else (),
        )
        for k in range(seconds)
    ]  # fmt: skip


def test_the_backward_pass_runs_last_to_first_and_reports_in_time_order():
    rng = np.random.default_rng(40)
    ephs = constellation()
    pairs = _l1_run(rng, ephs, 20)
    rover, base = [p[0] for p in pairs], [p[1] for p in pairs]
    station = Station(USUD, np.zeros(3))
    fwd = solve_rtk(rover, base, ephs, station, np.zeros(3))
    bwd = solve_rtk(rover, base, ephs, station, np.zeros(3), RtkOptions(backward=True))
    assert [f.tow for f in bwd] == [f.tow for f in fwd] == [r.tow for r in rover]
    # each filter ends converged where the other begins: the ambiguities are known best at
    # the last epoch forward and at the first backward
    spread = lambda f: max(a.std for a in f.ambiguities)
    assert spread(fwd[-1]) < 0.5 * spread(fwd[0]) and spread(bwd[0]) < 0.5 * spread(bwd[-1])
    assert fwd[-1].status == "fix" and bwd[0].status == "fix"
    for f in (*fwd, *bwd):
        if f.status == "fix":
            np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)


def test_backward_a_flagged_slip_is_met_at_the_phase_before_it():
    # the flag sits on the first phase after the slip; a filter coming from the other side
    # passes that phase with its ambiguity intact and must restart at the one before it
    rng = np.random.default_rng(41)
    ephs = constellation()
    key = ("G03", "G1C")
    after = dict(AMB_R)
    after[key] += 7
    pairs = _l1_run(
        rng, ephs, 8,
        amb_at=lambda k: after if k >= 5 else AMB_R,
        lli_at=lambda k: {key: 1} if k == 5 else None,
    )  # fmt: skip
    rtk = _rtk(ephs, backward=True)
    locks = {}
    for k in range(7, -1, -1):
        fix = rtk.step(*pairs[k])
        locks[k] = rtk.biases[key].lock
        if fix.status == "fix":
            np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)
    assert locks == {7: 1, 6: 2, 5: 3, 4: 1, 3: 2, 2: 3, 1: 4, 0: 5}
    assert fix.status == "fix"


def test_backward_the_half_cycle_bit_is_read_from_the_phase_processed_before():
    # RTKLIB's rule: backward, the bit that keeps an ambiguity out of the search is the one
    # of the indicator read before, so it follows a change of the file's bit one epoch late
    rng = np.random.default_rng(42)
    ephs = constellation()
    key = ("G03", "G1C")
    open_ = dict(AMB_R)
    open_[key] += 0.5
    pairs = _l1_run(
        rng, ephs, 8,
        amb_at=lambda k: open_ if k <= 3 else AMB_R,
        lli_at=lambda k: {key: 2} if k <= 3 else None,
    )  # fmt: skip
    rtk = _rtk(ephs, backward=True)
    seen = {}
    for k in range(7, -1, -1):
        rtk.step(*pairs[k])
        seen[k] = (rtk.biases[key].lock, rtk.biases[key].half)
    assert seen[4] == (4, False)
    assert seen[3] == (1, False)  # restarted at the change; the bit is still the one of 4
    assert seen[2] == (2, True) and seen[0] == (4, True)
    forward = _rtk(ephs)
    for k in range(8):
        forward.step(*pairs[k])
        if k == 3:
            assert forward.biases[key].half
    assert not forward.biases[key].half and forward.biases[key].lock == 4
    # the base's indicator counts the same way
    pairs = _l1_run(np.random.default_rng(42), ephs, 8)
    pairs = [(r, _flag(b, *key, 2)) if k <= 3 else (r, b) for k, (r, b) in enumerate(pairs)]
    rtk = _rtk(ephs, backward=True)
    seen = {}
    for k in range(7, -1, -1):
        rtk.step(*pairs[k])
        seen[k] = (rtk.biases[key].lock, rtk.biases[key].half)
    assert seen[4] == (4, False) and seen[3] == (1, False) and seen[2] == (2, True)


def _searched(fix, sat):
    return any(sat in (a.ref, a.sat) for a in fix.ambiguities)


def test_a_restarted_ambiguity_waits_min_lock_updates_before_the_search():
    # RTKLIB's pos2-arlockcnt: an ambiguity begun again counts up from -min_lock and is
    # searched once it is above zero; so does one begun on a phase that slipped, or after
    # the first max_outage solved epochs. One seen for the first time within those epochs,
    # with no slip before it, starts at zero
    rng = np.random.default_rng(43)
    ephs = constellation()
    key = ("G03", "G1C")
    after = dict(AMB_R)
    after[key] += 7
    pairs = _l1_run(
        rng, ephs, 12,
        amb_at=lambda k: after if k >= 6 else AMB_R,
        lli_at=lambda k: {key: 1} if k == 6 else None,
    )  # fmt: skip
    rtk = _rtk(ephs, min_lock=3)
    fixes = [rtk.step(*p) for p in pairs]
    assert [f.nb for f in fixes] == [5] * 6 + [4] * 3 + [5] * 3
    assert [_searched(f, "G03") for f in fixes] == [True] * 6 + [False] * 3 + [True] * 3
    assert rtk.biases[key].lock == 3 and rtk.biases[("G02", "G1C")].lock == 12
    for f in fixes:
        if f.status == "fix":
            np.testing.assert_allclose(f.position, ROVER, atol=0.01, rtol=0)
    stock = _rtk(ephs)
    assert [stock.step(*p).nb for p in pairs] == [5] * 12


def test_an_ambiguity_begun_again_after_an_outage_waits_too():
    rng = np.random.default_rng(44)
    ephs = constellation()
    gone = {("G03", "G1C")}
    rtk = _rtk(ephs, min_lock=2)
    fixes = [
        rtk.step(*_pair(rng, ephs, TOW + 1.0 * k, signals=("G1C",), drop=gone if 3 <= k < 8 else ()))
        for k in range(13)
    ]
    # back at the sixth epoch after its last update, past max_outage: the state was dropped
    assert [_searched(f, "G03") for f in fixes] == [True] * 3 + [False] * 5 + [False] * 2 + [True] * 3
    kept = _rtk(ephs, min_lock=2)
    fixes = [
        kept.step(*_pair(rng, ephs, TOW + 1.0 * k, signals=("G1C",), drop=gone if 3 <= k < 7 else ()))
        for k in range(9)
    ]
    # back at the fifth: the ambiguity is the old one and is searched at once
    assert [_searched(f, "G03") for f in fixes] == [True] * 3 + [False] * 4 + [True] * 2


def test_no_integer_search_below_the_satellite_floor():
    # a search over few ambiguities passes the ratio test without pinning the position; with
    # min_fix_sats the epoch stays float unless the eligible ambiguities span that many
    # satellites, the reference among them
    ephs = constellation()
    five = {("G06", "G1C")}  # no base observation: five satellites in the solution

    def run(floor, drop):
        rng = np.random.default_rng(45)
        rtk = _rtk(ephs, min_fix_sats=floor)
        return [
            rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, signals=("G1C",), drop=drop)) for k in range(12)
        ]

    below = run(6, five)
    assert all(f.ns == 5 and f.nb == 4 for f in below)
    for f in below:
        assert f.status == "float" and f.ratio == 0.0
        assert len(f.ambiguities) == 4 and all(a.integer is None for a in f.ambiguities)
        np.testing.assert_array_equal(f.position, f.float_position)
    at = run(5, five)
    assert at[-1].status == "fix" and at[-1].ratio >= 3.0
    np.testing.assert_allclose(at[-1].position, ROVER, atol=0.01, rtol=0)
    assert [f.status for f in at] == [f.status for f in run(0, five)]
    assert run(6, ())[-1].status == "fix"


def test_the_satellite_floor_counts_one_reference_per_system():
    # every system has its own reference satellite, so three GPS and two Galileo satellites
    # are three double differences, as few as four satellites of one system: the floor counts
    # the satellites beside the references, and five of one system are four
    gps = constellation()
    ephs = gps + as_galileo(gps)
    sats = [f"G{e.prn:02d}" for e in gps] + [f"E{e.prn + 10:02d}" for e in gps]

    def run(floor, kept):
        rng = np.random.default_rng(45)
        rtk = _rtk(ephs, min_fix_sats=floor)
        drop = {(sat, sig) for sat in sats if sat not in kept for sig in ("G1C", "E1C") if sig[0] == sat[0]}
        return [
            rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, signals=("G1C", "E1C"), drop=drop))
            for k in range(12)
        ]

    three_and_two = sats[:3] + sats[6:8]
    below = run(5, three_and_two)
    assert all(f.ns == 5 and f.nb == 3 for f in below)
    assert all(f.status == "float" and f.ratio == 0.0 for f in below)
    assert all(a.integer is None for f in below for a in f.ambiguities)
    # one more satellite of either system is the fourth double difference
    for kept in (sats[:4] + sats[6:8], sats[:3] + sats[6:9]):
        at = run(5, kept)
        assert all(f.ns == 6 and f.nb == 4 for f in at)
        assert all(f.ratio > 0.0 for f in at[1:])  # searched, as without a floor
        assert [f.ratio for f in at] == [f.ratio for f in run(0, kept)]
        assert all(f.status == "float" and f.ratio == 0.0 for f in run(6, kept))
    # and without a floor the five satellites are searched
    assert any(f.ratio > 0.0 for f in run(0, three_and_two))


# --- slips="check": a phase the rover flags is looked at before its ambiguity restarts ----

G03 = ("G03", "G1C")


def _offset(key, cycles, start, stop=None):
    """Rover ambiguities by second with `key` moved by `cycles` from `start` (to `stop`)."""
    moved = dict(AMB_R)
    moved[key] += cycles
    return lambda k: moved if k >= start and (stop is None or k < stop) else AMB_R


def _six(ephs, **options):
    """An engine that fixes only with all six satellites in the search: a fix is then right
    or an ambiguity is wrong, where five satellites at 1 Hz leave either to the noise."""
    return _rtk(ephs, min_fix_sats=6, **options)


def _stepped(rtk, pairs):
    """Step the pairs; every fix must be the true position."""
    fixes = {}
    for k in range(len(pairs)):
        fixes[k] = rtk.step(*pairs[k])
        if fixes[k].status == "fix":
            np.testing.assert_allclose(fixes[k].position, ROVER, atol=0.01, rtol=0)
    return fixes


def test_check_mode_without_a_flag_is_the_restart_mode():
    ephs = constellation()
    pairs = _l1_run(np.random.default_rng(50), ephs, 10, signals=SIGNALS)
    restart, check = _rtk(ephs), _rtk(ephs, slips="check")
    for pair in pairs:
        a, b = restart.step(*pair), check.step(*pair)
        assert (a.status, a.ratio, a.nb, a.ns) == (b.status, b.ratio, b.nb, b.ns)
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.float_position, b.float_position)
    with pytest.raises(ValueError, match="slips"):
        _rtk(ephs, slips="maybe")


def test_a_flagged_phase_that_fits_the_others_keeps_its_ambiguity():
    # a fade flag says a slip was possible; the phase is continuous, and the solution of
    # the unflagged satellites says so to a fraction of a cycle
    ephs = constellation()
    flags = lambda k: {G03: 1} if k in (6, 7, 9) else None
    pairs = _l1_run(np.random.default_rng(51), ephs, 12, lli_at=flags)
    check, restart = _six(ephs, slips="check", min_lock=3), _six(ephs, min_lock=3)
    fixes = _stepped(check, pairs)
    _stepped(restart, pairs)
    assert check.biases[G03].lock == 12 and restart.biases[G03].lock == 3 - 3
    assert all(_searched(f, "G03") for f in fixes.values())
    assert fixes[11].status == "fix"


@pytest.mark.parametrize("cycles", [0.5, -0.5, 1.0, 7.0])
def test_a_flagged_phase_that_slipped_restarts(cycles):
    ephs = constellation()
    pairs = _l1_run(
        np.random.default_rng(52), ephs, 12, amb_at=_offset(G03, cycles, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None,
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    fixes = _stepped(rtk, pairs)
    assert rtk.biases[G03].lock == 6 and rtk.biases[("G02", "G1C")].lock == 12
    # half a cycle off is no integer: nothing marks the phase here (the flag is the
    # lost-lock bit alone), so its ambiguity is searched and the search does not pass
    assert fixes[11].status == ("fix" if cycles % 1 == 0 else "float")


@pytest.mark.parametrize("cycles,kept", [(0.2, True), (-0.2, True), (0.3, False), (-0.3, False)])
def test_the_phase_must_fit_within_a_quarter_cycle(cycles, kept):
    # half the distance to the nearest slip of a Costas loop
    ephs = constellation()
    pairs = _l1_run(
        np.random.default_rng(53), ephs, 8, amb_at=_offset(G03, cycles, 6, 7),
        lli_at=lambda k: {G03: 1} if k == 6 else None,
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    for pair in pairs[:7]:
        rtk.step(*pair)
    assert rtk.biases[G03].lock == (7 if kept else 1)


def test_a_flag_from_the_base_restarts_without_a_look():
    ephs = constellation()
    for both in (False, True):
        pairs = _l1_run(
            np.random.default_rng(54), ephs, 12,
            lli_at=lambda k, both=both: {G03: 1} if both and k == 6 else None,
        )
        pairs[6] = (pairs[6][0], _flag(pairs[6][1], *G03, 1))
        rtk = _six(ephs, slips="check")
        _stepped(rtk, pairs)
        assert rtk.biases[G03].lock == 6


def test_a_flag_after_a_missed_epoch_restarts_without_a_look():
    # only an ambiguity updated at the previous epoch is looked at: across a gap the phase
    # may have gone anywhere, and a new track's fractional phase could pass by chance
    ephs = constellation()
    flags = lambda k: {G03: 1} if k == 6 else None
    gap = lambda k: {G03} if k == 5 else ()
    pairs = _l1_run(np.random.default_rng(55), ephs, 12, lli_at=flags, drop_at=gap)
    rtk = _six(ephs, slips="check")
    _stepped(rtk, pairs)
    assert rtk.biases[G03].lock == 6


@pytest.mark.parametrize("flagged,kept", [(1, True), (2, False), (3, False), (6, False)])
def test_too_few_unflagged_satellites_cannot_vouch_for_a_phase(flagged, kept):
    # five satellites not in question place the flagged one within 0.11 cycle; four have no
    # redundancy left (0.8 to 1 cycle here), three leave the position to the code, none
    # leave nothing to difference against
    ephs = constellation()
    sats = [f"G{n:02d}" for n in (3, 4, 5, 2, 6, 1)][:flagged]
    flags = lambda k: {(sat, "G1C"): 1 for sat in sats} if k == 6 else None
    pairs = _l1_run(np.random.default_rng(56), ephs, 12, lli_at=flags)
    rtk = _six(ephs, slips="check")
    _stepped(rtk, pairs)
    locks = {sat: rtk.biases[(sat, "G1C")].lock for sat in sats}
    assert locks == dict.fromkeys(sats, 12 if kept else 6)


def test_phases_all_in_question_do_not_align_the_state_of_an_absent_satellite():
    # every observed phase of the signal is flagged, one of them 5000 cycles off, and a sixth
    # satellite has no base observation at that epoch. With no phase to trust, the existing
    # states are not aligned to the epoch's clock at all (they restart): aligned to the mean
    # of the flagged phases, the absent satellite's state came back a thousand cycles off
    # and the run never fixed again
    ephs = constellation()
    absent = ("G06", "G1C")
    flagged = {(f"G{n:02d}", "G1C"): 1 for n in range(1, 6)}
    pairs = _l1_run(
        np.random.default_rng(67), ephs, 20, amb_at=_offset(G03, 5000.0, 8),
        lli_at=lambda k: flagged if k == 8 else None, drop_at=lambda k: {absent} if k == 8 else (),
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    fixes = _stepped(rtk, pairs[:8])
    before = rtk.x[rtk._index(absent)]
    fixes[8] = rtk.step(*pairs[8])
    assert rtk.x[rtk._index(absent)] == before
    assert [rtk.biases[key].lock for key in flagged] == [1] * 5
    fixes = _stepped(rtk, pairs[9:])
    assert fixes[len(fixes) - 1].status == "fix"


def test_a_phase_in_question_is_not_the_reference_of_another():
    # two flagged satellites, the highest of all among them and a cycle off; six Galileo
    # satellites keep the position sharp. The other flagged phase must be differenced
    # against the highest satellite that is not in question, or it inherits that cycle
    ephs = constellation() + as_galileo(constellation())
    top = ("G01", "G1C")
    flags = lambda k: {top: 1, G03: 1} if k == 6 else None
    pairs = _l1_run(
        np.random.default_rng(66), ephs, 10, amb_at=_offset(top, 1.0, 6), lli_at=flags,
        signals=("G1C", "E1C"),
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check")
    for pair in pairs:
        rtk.step(*pair)
    assert (rtk.biases[top].lock, rtk.biases[G03].lock) == (4, 10)


@pytest.mark.parametrize("limit,kept", [(0.10, False), (0.11, True), (0.125, True)])
def test_the_fit_must_be_predicted_sharply(monkeypatch, limit, kept):
    # the phase of the one flagged satellite of the run above is predicted within 0.107
    # cycle against the highest of the others; against a lower one the noise is larger
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", limit)
    ephs = constellation()
    pairs = _l1_run(np.random.default_rng(56), ephs, 8, lli_at=lambda k: {G03: 1} if k == 6 else None)
    rtk = _rtk(ephs, slips="check")
    for pair in pairs:
        rtk.step(*pair)
    assert rtk.biases[G03].lock == (8 if kept else 2)


def test_the_other_signal_vouches_for_a_flagged_phase(monkeypatch):
    # the geometry-free difference of a satellite's two signals did not step: the flagged
    # one is continuous, whatever the other satellites say
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)  # the test against the filter keeps nothing
    ephs = constellation()
    flags = lambda k: {G03: 1} if k in (6, 7) else None
    dual = _l1_run(np.random.default_rng(57), ephs, 12, lli_at=flags, signals=SIGNALS)
    rtk = _six(ephs, slips="check")
    fixes = _stepped(rtk, dual)
    assert rtk.biases[G03].lock == 12 and fixes[11].status == "fix"
    second = ("G03", "G2W")
    other = _l1_run(
        np.random.default_rng(57), ephs, 12, lli_at=lambda k: {second: 1} if k == 6 else None,
        signals=SIGNALS,
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    _stepped(rtk, other)
    assert rtk.biases[second].lock == 12
    single = _l1_run(np.random.default_rng(57), ephs, 12, lli_at=flags)
    rtk = _six(ephs, slips="check")
    _stepped(rtk, single)
    assert rtk.biases[G03].lock == 12 - 7
    # two flagged signals do not vouch for each other
    both = _l1_run(
        np.random.default_rng(57), ephs, 12, lli_at=lambda k: {G03: 1, second: 1} if k == 6 else None,
        signals=SIGNALS,
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    _stepped(rtk, both)
    assert (rtk.biases[G03].lock, rtk.biases[second].lock) == (6, 6)


def test_the_reference_moves_on_when_a_signal_of_the_pair_restarts():
    # the base flags the first signal while the rover flags the second: the first restarts,
    # and the geometry-free reference is the new value, not the one from before the flags.
    # 4 cm at 6 s and 4 cm more at 7 s are then two small steps, not one of 8 cm
    ephs = constellation()
    second = ("G03", "G2W")
    moved = {k: dict(AMB_R) for k in range(12)}
    for k in range(6, 12):
        moved[k][G03] += 0.21 * min(k - 5, 2)
    pairs = _l1_run(
        np.random.default_rng(65), ephs, 12, amb_at=lambda k: moved[k],
        lli_at=lambda k: {second: 1} if k == 6 else None, signals=SIGNALS,
    )  # fmt: skip
    pairs[6] = (pairs[6][0], _flag(pairs[6][1], *G03, 1))
    rtk = _rtk(ephs, slips="check")
    for pair in pairs[:8]:
        rtk.step(*pair)
    assert (rtk.biases[G03].lock, rtk.biases[second].lock) == (2, 8)


def test_a_geometry_free_step_restarts_both_signals_whichever_is_flagged():
    # a fade hits both signals; the flag on one does not say the other held
    ephs = constellation()
    second = ("G03", "G2W")
    for slipped, flagged in ((G03, G03), (G03, second), (second, G03)):
        pairs = _l1_run(
            np.random.default_rng(58), ephs, 12, amb_at=_offset(slipped, 1.0, 6),
            lli_at=lambda k, flagged=flagged: {flagged: 1} if k == 6 else None, signals=SIGNALS,
        )  # fmt: skip
        rtk = _six(ephs, slips="check")
        _stepped(rtk, pairs)
        assert (rtk.biases[G03].lock, rtk.biases[second].lock) == (6, 6)


def test_a_slip_spread_over_flagged_epochs_is_measured_whole():
    # 2.3 cm a second for three flagged seconds: each step is below the 5 cm threshold, the
    # sum is not; the reference stays where the phase was last unflagged
    ephs = constellation()
    second = ("G03", "G2W")
    creep = {k: dict(AMB_R) for k in range(12)}
    for k in range(12):
        creep[k][G03] += 0.12 * min(max(k - 5, 0), 3)
    pairs = _l1_run(
        np.random.default_rng(59), ephs, 12, amb_at=lambda k: creep[k],
        lli_at=lambda k: {G03: 1} if k in (6, 7, 8) else None, signals=SIGNALS,
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    locks = {}
    for k, pair in enumerate(pairs):
        rtk.step(*pair)
        locks[k] = (rtk.biases[G03].lock, rtk.biases[second].lock)
    assert locks[7] == (8, 8) and locks[8] == (1, 1) and locks[11] == (4, 4)


def test_a_restart_by_the_filter_test_does_not_leave_a_stale_reference():
    # the second signal is missing when the first slips, so the slip is found against the
    # filter; the old geometry-free reference must not restart both when the second returns
    ephs = constellation()
    second = ("G03", "G2W")
    pairs = _l1_run(
        np.random.default_rng(60), ephs, 12, amb_at=_offset(G03, 1.0, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=SIGNALS,
        drop_at=lambda k: {second} if k == 6 else (),
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    fixes = _stepped(rtk, pairs)
    assert (rtk.biases[G03].lock, rtk.biases[second].lock) == (6, 11)
    assert fixes[11].status == "fix"


def test_a_continuing_phase_keeps_an_established_half_cycle():
    # fades leave the half cycle open in the file (bit 1) from 6 to 8 s; the phase did not
    # move, so an ambiguity that was established stays in the search
    ephs = constellation()
    flags = lambda k: {G03: 2} if 6 <= k < 9 else None
    pairs = _l1_run(np.random.default_rng(61), ephs, 12, lli_at=flags)
    check, restart = _six(ephs, slips="check"), _six(ephs)
    fixes, stock = _stepped(check, pairs), _stepped(restart, pairs)
    assert [_searched(fixes[k], "G03") for k in range(12)] == [True] * 12
    assert [_searched(stock[k], "G03") for k in range(12)] == [True] * 6 + [False] * 3 + [True] * 3
    assert (check.biases[G03].lock, restart.biases[G03].lock) == (12, 3)


def test_a_restarted_phase_takes_the_half_cycle_bit_of_the_file():
    # the loop slips half a cycle at 6 s (flagged, bit 1 open); at 9 s a subframe shows the
    # new polarity and the receiver puts the half cycle back (flagged, bit closed)
    ephs = constellation()
    flags = lambda k: {G03: 3} if k == 6 else {G03: 2} if k in (7, 8) else {G03: 1} if k == 9 else None
    pairs = _l1_run(np.random.default_rng(62), ephs, 12, amb_at=_offset(G03, 0.5, 6, 9), lli_at=flags)
    rtk = _six(ephs, slips="check")
    fixes = _stepped(rtk, pairs)
    assert [_searched(fixes[k], "G03") for k in range(12)] == [True] * 6 + [False] * 3 + [True] * 3
    assert rtk.biases[G03].lock == 3 and not rtk.biases[G03].half


def test_a_closing_half_cycle_bit_on_a_continuing_phase_establishes_it():
    # before the first subframe the bit is open; the subframe confirms the polarity the
    # receiver assumed and the phase goes on as it was
    ephs = constellation()
    pairs = _l1_run(np.random.default_rng(63), ephs, 12, lli_at=lambda k: {G03: 2} if k < 6 else None)
    check, restart = _six(ephs, slips="check"), _six(ephs)
    fixes = _stepped(check, pairs)
    _stepped(restart, pairs)
    assert [_searched(fixes[k], "G03") for k in range(12)] == [False] * 6 + [True] * 6
    assert (check.biases[G03].lock, restart.biases[G03].lock) == (12, 6)


def test_backward_a_flagged_phase_is_looked_at_from_the_other_side():
    # a slip of one cycle between 5 and 6 s, flagged at 6: backward the filter meets it at
    # 5, restarts there, and keeps the phases flagged for nothing at 8 and 3
    ephs = constellation()
    flags = lambda k: {G03: 1} if k in (3, 6, 8) else None
    pairs = _l1_run(np.random.default_rng(64), ephs, 12, amb_at=_offset(G03, 1.0, 6), lli_at=flags)
    rtk = _six(ephs, slips="check", backward=True)
    locks = {}
    for k in range(11, -1, -1):
        rtk.step(*pairs[k])
        locks[k] = rtk.biases[G03].lock
    assert locks[6] == 6 and locks[5] == 1 and locks[0] == 6


@pytest.mark.parametrize("cycles", [900.0, -5000.0])
def test_a_large_flagged_slip_restarts_as_cleanly_as_without_the_check(cycles):
    # the states of a signal are moved together onto the current phase-minus-code level; a
    # slipped phase in question must not take part in that mean, or the restarted state
    # starts a fraction of the slip away from the others and its phase is thrown out
    ephs = constellation()
    pairs = _l1_run(
        np.random.default_rng(67), ephs, 30, amb_at=_offset(G03, cycles, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None,
    )  # fmt: skip
    check, restart = _six(ephs, slips="check"), _six(ephs)
    a, b = _stepped(check, pairs), _stepped(restart, pairs)
    assert [a[k].status for k in range(30)] == [b[k].status for k in range(30)]
    assert a[29].status == "fix" and check.biases[G03].lock == 24


def test_backward_a_restarted_phase_takes_the_half_cycle_bit_of_its_own_epoch():
    # the run of test_a_restarted_phase_takes_the_half_cycle_bit_of_the_file from the other
    # side: the ambiguity begun at 8 s is half a cycle off until 6 s and must stay out of the
    # search there, although the indicator read before it (9 s) has the bit closed
    ephs = constellation()
    flags = lambda k: {G03: 3} if k == 6 else {G03: 2} if k in (7, 8) else {G03: 1} if k == 9 else None
    pairs = _l1_run(np.random.default_rng(62), ephs, 12, amb_at=_offset(G03, 0.5, 6, 9), lli_at=flags)
    rtk = _six(ephs, slips="check", backward=True)
    searched = {}
    for k in range(11, -1, -1):
        fix = rtk.step(*pairs[k])
        searched[k] = _searched(fix, "G03")
        if fix.status == "fix":
            np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)
    assert [searched[k] for k in range(12)] == [True] * 6 + [False] * 3 + [True] * 3
    assert rtk.biases[G03].lock == 6 and not rtk.biases[G03].half


def test_only_an_early_unflagged_first_ambiguity_skips_the_lock_count():
    # RTKLIB's counter is at -minlock for every ambiguity begun on a flagged phase, and for
    # one that begins after max_outage epochs without it: a first ambiguity enters the
    # search at once only when it comes unflagged within the first five solved epochs
    ephs = constellation()

    def searched(first, flagged=False):
        pairs = _l1_run(
            np.random.default_rng(68), ephs, 12,
            lli_at=lambda k: {G03: 1} if flagged and k == first else None,
            drop_at=lambda k: {G03} if k < first else (),
        )  # fmt: skip
        rtk = _rtk(ephs, min_lock=3)
        return [_searched(rtk.step(*pair), "G03") for pair in pairs]

    assert searched(0) == [True] * 12
    assert searched(4) == [False] * 4 + [True] * 8
    assert searched(5) == [False] * 5 + [False] * 3 + [True] * 4
    assert searched(0, flagged=True) == [False] * 3 + [True] * 9
    assert searched(2, flagged=True) == [False] * 2 + [False] * 3 + [True] * 7
    # and one begun again waits whatever ended the first: here an unflagged step of the
    # geometry-free difference at the third epoch
    pairs = _l1_run(np.random.default_rng(68), ephs, 8, amb_at=_offset(G03, 1.0, 2), signals=SIGNALS)
    rtk = _rtk(ephs, min_lock=3)
    assert [_searched(rtk.step(*pair), "G03") for pair in pairs] == [True] * 2 + [False] * 3 + [True] * 3


def test_an_ambiguity_no_update_has_used_is_not_looked_at():
    # at the first epoch one satellite alone has the second signal at the base: its
    # ambiguity begins and no double difference uses it. Flagged at the next epoch, it was
    # not "updated at the previous epoch": it restarts and waits the lock count, where the
    # first signal once vouched for it and it kept counting from its first epoch
    ephs = constellation()
    second = ("G03", "G2W")
    alone = {(f"G{n:02d}", "G2W") for n in (1, 2, 4, 5, 6)}

    def locks(**options):
        pairs = _l1_run(
            np.random.default_rng(81), ephs, 4, signals=SIGNALS,
            lli_at=lambda k: {second: 1} if k == 1 else None, drop_at=lambda k: alone if k == 0 else (),
        )  # fmt: skip
        rtk = _rtk(ephs, min_lock=3, **options)
        out = []
        for pair in pairs:
            rtk.step(*pair)
            out.append(rtk.biases[second].lock)
        return out

    assert locks(slips="check") == locks() == [0, -2, -1, 0]


@pytest.mark.parametrize("slips", ["restart", "check"])
def test_a_slip_before_the_first_ambiguity_makes_it_wait_the_lock_count(slips):
    # RTKLIB puts its counter at -minlock at a slip whether or not the phase has a state.
    # A geometry-free step: the second signal has no base code at the first epoch, so no
    # ambiguity, and steps a cycle without a flag at the next, where its first one begins
    ephs = constellation()
    second = ("G03", "G2W")
    pairs = _l1_run(np.random.default_rng(82), ephs, 4, amb_at=_offset(second, 1.0, 1), signals=SIGNALS)
    r, b = pairs[0]
    code = b.pseudorange.copy()
    code[b.sats.index("G03"), 1] = np.nan
    pairs[0] = (r, replace(b, pseudorange=code))
    rtk = _rtk(ephs, min_lock=3, slips=slips)
    locks = []
    for pair in pairs:
        rtk.step(*pair)
        locks.append([rtk.biases[key].lock if key in rtk.biases else None for key in (G03, second)])
    assert locks == [[1, None], [-2, -2], [-1, -1], [0, 0]]
    # a flag at an epoch that ends early, on a satellite the base did not have until then
    pairs = _l1_run(
        np.random.default_rng(82), ephs, 5, lli_at=lambda k: {G03: 1} if k == 2 else None,
        drop_at=lambda k: {G03} if k < 2 else (),
    )  # fmt: skip
    pairs[2] = _blunder(*pairs[2])
    rtk = _rtk(ephs, min_lock=3, slips=slips)
    locks = []
    for pair in pairs:
        rtk.step(*pair)
        locks.append(rtk.biases[G03].lock if G03 in rtk.biases else None)
    assert locks == [None, None, None, -2, -1]


def test_a_slip_held_from_an_unsolved_epoch_leaves_no_reference_to_vouch_with():
    # the second signal slips 7 cycles, flagged at an epoch that gets no solution; the first
    # slips 9 at the next. The two steps cancel in the geometry-free difference to 3 mm, and
    # the second signal's phase carries no flag at that epoch: it must not vouch
    ephs = constellation()
    second = ("G03", "G2W")
    after = {5: dict(AMB_R), 6: dict(AMB_R)}
    after[5][second] += 7
    after[6][second] += 7
    after[6][G03] += 9
    pairs = _l1_run(
        np.random.default_rng(69), ephs, 12,
        amb_at=lambda k: after[min(k, 6)] if k >= 5 else AMB_R,
        lli_at=lambda k: {second: 1} if k == 5 else {G03: 1} if k == 6 else None, signals=SIGNALS,
    )  # fmt: skip
    pairs[5] = _blunder(*pairs[5])  # no single-point seed at 5 s: the epoch ends early
    rtk = _six(ephs, slips="check")
    for k, pair in enumerate(pairs):
        fix = rtk.step(*pair)
        assert (fix is None) == (k == 5)
        if fix is not None and fix.status == "fix":
            np.testing.assert_allclose(fix.position, ROVER, atol=0.01, rtol=0)
    assert (rtk.biases[G03].lock, rtk.biases[second].lock) == (6, 6)


def test_check_mode_keeps_the_rules_that_need_no_flag():
    ephs = constellation()
    second, third = ("G03", "G2W"), ("G03", "G5Q")
    # an unflagged geometry-free step restarts both signals
    pairs = _l1_run(np.random.default_rng(70), ephs, 10, amb_at=_offset(G03, 1.0, 6), signals=SIGNALS)
    rtk = _six(ephs, slips="check")
    _stepped(rtk, pairs)
    assert (rtk.biases[G03].lock, rtk.biases[second].lock) == (4, 4)
    # a flagged phase without a code at the rover cannot be tested: its ambiguity goes
    pairs = _l1_run(
        np.random.default_rng(70), ephs, 10, lli_at=lambda k: {G03: 1} if k == 6 else None, signals=SIGNALS
    )
    r, b = pairs[6]
    code = r.pseudorange.copy()
    code[r.sats.index("G03"), 0] = np.nan
    pairs[6] = (replace(r, pseudorange=code), b)
    rtk = _six(ephs, slips="check")
    for pair in pairs:
        rtk.step(*pair)
    assert (rtk.biases[G03].lock, rtk.biases[second].lock) == (3, 10)
    # a restart by the test against the filter waits the lock count like any other
    pairs = _l1_run(
        np.random.default_rng(70), ephs, 12, amb_at=_offset(G03, 1.0, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None,
    )  # fmt: skip
    rtk = _six(ephs, slips="check", min_lock=3)
    fixes = _stepped(rtk, pairs)
    assert [_searched(fixes[k], "G03") for k in range(12)] == [True] * 6 + [False] * 3 + [True] * 3
    assert third not in rtk.biases


def test_a_vouched_first_signal_vouches_for_a_third(monkeypatch):
    # three signals, the first and the third flagged, the second not: the second settles the
    # first, and the first, now known to be continuous, settles the third
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)  # the test against the filter keeps nothing
    ephs = constellation()
    signals = ("G1C", "G2W", "G5Q")
    third = ("G03", "G5Q")
    pairs = _l1_run(
        np.random.default_rng(71), ephs, 12, lli_at=lambda k: {G03: 1, third: 1} if k == 6 else None,
        signals=signals,
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check")
    for pair in pairs:
        rtk.step(*pair)
    assert [rtk.biases[("G03", sig)].lock for sig in signals] == [12, 12, 12]
    # and in the other order: the first two flagged, the third settles the first after the
    # second was looked at, and the second is settled with it
    pairs = _l1_run(
        np.random.default_rng(71), ephs, 12,
        lli_at=lambda k: {G03: 1, ("G03", "G2W"): 1} if k == 6 else None, signals=signals,
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check")
    for pair in pairs:
        rtk.step(*pair)
    assert [rtk.biases[("G03", sig)].lock for sig in signals] == [12, 12, 12]


@pytest.mark.parametrize("signals", [("G1C", "G2W", "G5Q"), ("G1C", "G5Q", "G2W")])
@pytest.mark.parametrize("flagged,cycles", [("G5Q", (4, 3)), ("G2W", (9, 7))])
def test_a_first_signal_that_stepped_vouches_for_nothing(signals, flagged, cycles):
    # three signals: the first slips without a flag, which shows against one of the others
    # and restarts both; the remaining one slips with a flag. Against the first the two
    # slips cancel to 3 mm, but the first is not continuous: its word for the flagged
    # phase does not hold, whether it was given before the step showed or would come after
    ephs = constellation()
    key = ("G03", flagged)
    after = dict(AMB_R)
    after[G03] += cycles[0]
    after[key] += cycles[1]
    pairs = _l1_run(
        np.random.default_rng(73), ephs, 12, amb_at=lambda k: after if k >= 6 else AMB_R,
        lli_at=lambda k: {key: 1} if k == 6 else None, signals=signals,
    )  # fmt: skip
    rtk = _six(ephs, slips="check")
    fixes = _stepped(rtk, pairs)
    assert [rtk.biases[("G03", sig)].lock for sig in signals] == [6, 6, 6]
    assert fixes[11].status == "fix"


def test_a_held_base_flag_of_a_system_the_rover_does_not_use_is_no_trouble():
    # the base file carries Galileo, the rover is asked for GPS only: a Galileo flag of an
    # epoch that ended early has no first signal on the rover's side to look up
    gps = constellation()
    ephs = gps + as_galileo(gps)
    rng = np.random.default_rng(74)
    rtk = _rtk(ephs, slips="check")
    for k in range(8):
        r, _ = _pair(rng, gps, TOW + 1.0 * k, signals=("G1C",))
        _, b = _pair(rng, ephs, TOW + 1.0 * k, signals=("G1C", "E1C"))
        if k == 4:
            r, b = _blunder(r, _flag(b, "E13", "E1C", 1))
        fix = rtk.step(r, b)
        assert (fix is None) == (k == 4)
    assert fix.status == "fix" and rtk.biases[G03].lock == 7


def test_the_floor_counts_satellites_of_every_signal_not_double_differences():
    ephs = constellation()
    five = {("G06", "G1C"), ("G06", "G2W")}  # five satellites, eight double differences
    rng = np.random.default_rng(72)
    rtk = _rtk(ephs, min_fix_sats=6)
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, drop=five)) for k in range(4)]
    assert all(f.nb == 8 and f.status == "float" and f.ratio == 0.0 for f in fixes)
    # a satellite whose first signal is out of the search counts through its second
    rng = np.random.default_rng(72)
    rtk = _rtk(ephs, min_fix_sats=6)
    fixes = [rtk.step(*_pair(rng, ephs, TOW + 30.0 * k, lli={("G06", "G1C"): 2})) for k in range(4)]
    assert fixes[-1].nb == 9 and fixes[-1].status == "fix"


def _changes(rng, sats, bands=2, noise=0.002):
    """Changes (m) of `sats` satellites on `bands` frequencies that one change of position and
    one clock per frequency explain: (change, lines of sight, frequency index, satellite
    index, clean)."""
    az, el = rng.uniform(0.0, 2.0 * np.pi, sats), rng.uniform(np.radians(15.0), np.pi / 2, sats)
    up = np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=1)
    los, band = np.repeat(up, bands, axis=0), np.tile(np.arange(bands), sats)
    change = -los @ rng.normal(0.0, 1.0, 3) + rng.normal(0.0, 0.3, bands)[band]
    change += rng.normal(0.0, noise, len(band))
    return change, los, band, np.repeat(np.arange(sats), bands), np.ones(len(band), dtype=bool)


def _which(verdict):
    return list(np.flatnonzero(verdict))


def test_changes_that_one_solution_explains_are_not_moved():
    for seed in range(20):
        assert not _moved(*_changes(np.random.default_rng(seed), 8), 0.03).any()


@pytest.mark.parametrize("move", [0.05, -0.05, 0.10, -0.10, 0.25])
@pytest.mark.parametrize("flagged", [False, True])
def test_a_phase_that_moved_alone_stands_out(move, flagged):
    # whether or not a receiver flags it: a flagged phase takes no part in the solution of
    # the others and is judged by it all the same
    for seed in range(20):
        change, los, band, sat, clean = _changes(np.random.default_rng(seed), 9)
        change[5] += move
        clean[5] = not flagged
        assert _which(_moved(change, los, band, sat, clean, 0.03)) == [5]
    change, los, band, sat, clean = _changes(np.random.default_rng(0), 9)
    change[5] += 0.02
    assert not _moved(change, los, band, sat, clean, 0.03).any()


def test_a_pair_that_moved_together_stands_out():
    # both signals of a satellite by the same length: their difference shows nothing; against
    # the other satellites they stand out together
    found = 0
    for seed in range(40):
        change, los, band, sat, clean = _changes(np.random.default_rng(seed), 12)
        change[[6, 7]] += 0.06
        verdict = _which(_moved(change, los, band, sat, clean, 0.03))
        # where the satellite stands alone in its part of the sky the position takes the
        # move up and nothing shows; no other phase is blamed for it
        assert verdict in ([6, 7], [])
        found += verdict == [6, 7]
    assert found >= 38


def test_flagged_phases_do_not_steer_the_solution():
    change, los, band, sat, clean = _changes(np.random.default_rng(3), 10)
    change[:6] += 0.5
    clean[:6] = False
    assert _which(_moved(change, los, band, sat, clean, 0.03)) == [0, 1, 2, 3, 4, 5]


def test_no_verdict_below_the_least_number_of_unflagged_phases():
    change, los, band, sat, clean = _changes(np.random.default_rng(4), 10)
    clean[:9] = False
    assert _moved(change, los, band, sat, clean, 0.03) is None  # 11 unflagged
    clean[8] = True
    assert not _moved(change, los, band, sat, clean, 0.03).any()


@pytest.mark.parametrize("flagged,moved,verdict", [(0, 4, True), (0, 8, False), (1, 7, False)])
def test_no_verdict_when_too_many_phases_moved(flagged, moved, verdict):
    # twenty phases, some of them moved, each by its own 20 cm to 1 m. A solution of five
    # unknowns explains five phases for nothing, so twelve explained and eight left out
    # (7 against 8) is no majority, sixteen and four (11 against 4) is one; and with one of
    # the twenty flagged, twelve and seven (7 against 7) is none either. Six more phases
    # that are flagged and did not move add nothing to the explained side
    for seed in range(20):
        rng = np.random.default_rng(seed)
        change, los, band, sat, clean = _changes(rng, 13)
        clean[20:] = False
        clean[:flagged] = False
        which = np.sort(flagged + rng.permutation(20 - flagged)[:moved])
        change[which] += rng.uniform(0.2, 1.0, moved) * rng.choice([-1, 1], moved)
        result = _moved(change, los, band, sat, clean, 0.03)
        if verdict:
            assert _which(result) == list(which)
        else:
            assert result is None


def test_one_frequency_has_one_clock():
    change, los, band, sat, clean = _changes(np.random.default_rng(6), 16, bands=1)
    change[4] += 0.05
    assert _which(_moved(change, los, band, sat, clean, 0.03)) == [4]


@pytest.mark.parametrize("unflagged,judged", [(1, False), (2, False), (3, False), (4, True)])
def test_a_frequency_is_judged_on_three_explained_phases_or_more(unflagged, judged):
    # a clock fitted to one phase explains it whatever it did, and to two not much less: on
    # the second frequency all but a few phases are flagged, and the first unflagged one
    # moved by 6 cm. With fewer than three others explained beside it nothing on that
    # frequency is judged, the flagged phases that did not move least of all
    for seed in range(20):
        change, los, band, sat, clean = _changes(np.random.default_rng(seed), 16)
        second = np.flatnonzero(band == 1)
        clean[second[unflagged:]] = False
        change[second[0]] += 0.06
        verdict = _moved(change, los, band, sat, clean, 0.03)
        assert _which(verdict) == ([second[0]] if judged else [])


@pytest.mark.parametrize("signals", [(0, 0, 1, 2), (0, 1, 2)])
def test_no_solution_from_phases_that_cannot_tell_the_position_from_the_clocks(signals):
    # unflagged satellites at one elevation: their lines of sight have one height, so a
    # change of height is a change of every clock to them. A flagged satellite at the
    # zenith changed by what 16 cm of height and clocks of -8 cm do, exactly: any solution
    # of the others that leaves the height where it was calls it moved by 8 cm. Three
    # satellites with two signals on one of three frequencies, or four on three: twelve
    # unflagged phases either way, and no verdict
    ring = 3 if len(signals) == 4 else 4

    def verdict(others):
        az = 2.0 * np.pi * np.arange(ring) / ring
        up = [[np.sqrt(0.75) * np.sin(a), np.sqrt(0.75) * np.cos(a), 0.5] for a in az]
        up = np.array([*up, *others, [0.0, 0.0, 1.0]])
        los, band = np.repeat(up, len(signals), axis=0), np.tile(signals, len(up))
        sat = np.repeat(np.arange(len(up)), len(signals))
        change = los @ [0.0, 0.0, 0.16] - 0.08
        clean = sat < len(up) - 1
        assert clean.sum() == (ring + len(others)) * len(signals)
        return _moved(change, los, band, sat, clean, 0.03)

    assert verdict([]) is None  # twelve unflagged phases
    # two more unflagged satellites at other elevations: a solution, and it is the right one
    assert not verdict([[0.6, 0.0, 0.8], [0.0, -0.8, 0.6]]).any()


@pytest.mark.parametrize("satellites,judged", [(2, False), (3, True)])
def test_a_frequency_is_judged_on_three_satellites_not_three_phases(satellites, judged):
    # two signals on the second frequency (as L1 C/A and L1 P on the first would be): two
    # satellites are four explained phases there, and a clock fitted to two satellites
    # still explains them whatever they did
    change, los, band, sat, clean = _changes(np.random.default_rng(3), 12, bands=1)
    twice = np.repeat(np.arange(satellites), 2)
    rows = np.concatenate([np.arange(12), twice, [11]])
    change, los, sat = change[rows], los[rows], sat[rows]
    band = np.concatenate([np.zeros(12, dtype=int), np.ones(len(twice) + 1, dtype=int)])
    clean = np.ones(len(rows), dtype=bool)
    clean[-1] = False
    change[-1] += 0.5  # a flagged phase of the second frequency, half a meter off
    verdict = _moved(change, los, band, sat, clean, 0.03)
    assert _which(verdict) == ([len(rows) - 1] if judged else [])


@pytest.mark.parametrize(
    "bands,sats,flagged,moved,least", [(2, 10, 1, 6, 13), (2, 10, 0, 7, 13), (3, 8, 0, 9, None)]
)
def test_the_majority_counts_one_phase_off_for_every_unknown(bands, sats, flagged, moved, least):
    # two frequencies, five unknowns: of 19 unflagged phases six moved (thirteen explained
    # less five outnumber six) and of 20 seven moved (eight against seven): a verdict,
    # where the solution finds them (15 and 17 of the 20 epochs). Three frequencies, six
    # unknowns: of 24 nine moved, fifteen less six do not outnumber nine: none
    exact = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        change, los, band, sat, clean = _changes(rng, sats, bands=bands)
        clean[:flagged] = False
        which = np.sort(flagged + rng.permutation(len(change) - flagged)[:moved])
        change[which] += rng.uniform(0.2, 1.0, moved) * rng.choice([-1, 1], moved)
        verdict = _moved(change, los, band, sat, clean, 0.03)
        assert verdict is None or _which(verdict) == list(which)
        exact += verdict is not None
    assert exact == 0 if least is None else exact >= least


def test_no_verdict_when_no_unflagged_line_of_sight_has_a_part_in_one_direction():
    # the unflagged satellites stand in one plane through the antenna: a step of the rover
    # across it shows in none of their phases, and the flagged one off the plane would be
    # called moved for that step
    rng = np.random.default_rng(5)
    _, los, band, sat, clean = _changes(rng, 10)
    los[sat != 0, 0] = 0.0
    los[sat == 0] = [0.6, 0.0, 0.8]
    clean[sat == 0] = False
    change = -los @ np.array([0.5, 0.2, -0.1]) + np.array([0.3, -0.2])[band]
    change += rng.normal(0.0, 0.002, len(band))
    assert _moved(change, los, band, sat, clean, 0.03) is None


def test_a_frequency_without_an_unflagged_phase_is_not_judged():
    # its clock is not known, so the changes of its phases say nothing
    change, los, band, sat, clean = _changes(np.random.default_rng(7), 16)
    clean[band == 1] = False
    change[band == 1] += 7.0
    change[4] += 0.05  # an unflagged phase of the other frequency
    assert _which(_moved(change, los, band, sat, clean, 0.03)) == [4]


@pytest.mark.parametrize("noise,found_least,blamed_most", [(0.002, 93, 3), (0.005, 93, 3)])
def test_the_verdict_holds_from_strong_to_weak_phases(noise, found_least, blamed_most):
    # twelve satellites, eight of their 24 phases flagged, among them one signal of the
    # satellite whose two phases move together by 6 cm; the changes carry the noise of a
    # strong pair of phases (2 mm) and of a weak one (5 mm: 4 mm is what the phases of a
    # parked car at 30 to 34 dB-Hz show). Of 100 epochs: how often both are found, how often
    # another phase is blamed, and that nothing is called moved when nothing moved. Measured
    # on these seeds 100 and 0 at both; on the next hundred 95 and 1, 96 and 2
    found = blamed = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        change, los, band, sat, clean = _changes(rng, 12, noise=noise)
        clean[[0, *rng.permutation(np.arange(2, 24))[:7]]] = False
        assert not _moved(change, los, band, sat, clean, 0.03)[clean].any()
        change[:2] += 0.06
        verdict = _moved(change, los, band, sat, clean, 0.03)
        found += verdict[:2].all()
        blamed += verdict[2:].any()
    assert found >= found_least and blamed <= blamed_most, (found, blamed)


def test_the_solutions_without_one_satellite_find_a_small_move_among_weak_phases():
    # twelve unflagged phases of twelve satellites with 5 mm of noise, a flagged phase and
    # its unflagged partner moved by 4 cm: the partner pulls on the solution of all the
    # unflagged phases, which then finds the two in 119 of 200 epochs; with the solutions
    # that leave one satellite out, 149
    found = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        change, los, band, sat, clean = _changes(rng, 12, noise=0.005)
        clean[[0, *rng.permutation(np.arange(2, 24))[:11]]] = False
        change[:2] += 0.04
        verdict = _moved(change, los, band, sat, clean, 0.03)
        found += verdict is not None and verdict[:2].all()
    assert found >= 140


# two systems on two frequencies each, three in all (L1 and E1 are one): 24 phases, enough
# for a verdict on an epoch's changes
BOTH = ("G1C", "G2W", "E1C", "E5Q")
G03_2 = ("G03", "G2W")


def _two_systems():
    return constellation() + as_galileo(constellation())


def _moved_by(meters, start, stop=None, keys=(G03, G03_2)):
    """Rover ambiguities by second with the phases `keys` moved by one length from `start`
    (to `stop`): what the geometry-free difference of two signals cannot show."""
    moved = dict(AMB_R)
    for key in keys:
        moved[key] += meters / wavelength(key[1])
    return lambda k: moved if k >= start and (stop is None or k < stop) else AMB_R


def _locks(pairs, ephs, keys=(G03, G03_2), **options):
    rtk = _rtk(ephs, slips="check", **options)
    for pair in pairs:
        rtk.step(*pair)
    return [rtk.biases[key].lock for key in keys]


@pytest.mark.parametrize("flagged,locks", [(G03, [6, 12]), (G03_2, [12, 6])])
def test_a_flagged_phase_whose_pair_moved_is_not_vouched_for(flagged, locks):
    # both signals of a satellite move by 8 cm at 6 s and one of them carries a flag. Their
    # difference did not step, so the unflagged one vouched for the flagged one; against the
    # other satellites both moved, and the flagged one goes to the test against the rest of
    # the epoch, which it fails by a third of a cycle or more
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(80), ephs, 12, amb_at=_moved_by(0.08, 6),
        lli_at=lambda k: {flagged: 1} if k == 6 else None, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs) == [12, 12]
    assert _locks(pairs, ephs, change_limit=0.03) == locks


@pytest.mark.parametrize("flagged,locks", [(G03, [6, 12]), (G03_2, [12, 6])])
def test_a_flagged_phase_that_moved_under_the_step_threshold_is_not_vouched_for(
    monkeypatch, flagged, locks
):
    # 4 cm on the flagged signal alone: under the 5 cm of the geometry-free test, over the
    # limit against the other satellites
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)  # the test against the filter keeps nothing
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(81), ephs, 12, amb_at=_moved_by(0.04, 6, keys=(flagged,)),
        lli_at=lambda k: {flagged: 1} if k == 6 else None, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs) == [12, 12]
    assert _locks(pairs, ephs, change_limit=0.03) == locks


@pytest.mark.parametrize("meters", [0.0, 0.02])
def test_the_other_signal_still_vouches_when_neither_moved(monkeypatch, meters):
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(82), ephs, 12, amb_at=_moved_by(meters, 6),
        lli_at=lambda k: {G03: 1} if k in (6, 7) else None, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs, change_limit=0.03) == [12, 12]


def test_without_a_verdict_the_other_signal_vouches_as_before():
    # one system: 12 phases, and 11 without the flagged one, too few to judge their changes
    ephs = constellation()
    pairs = _l1_run(
        np.random.default_rng(83), ephs, 12, amb_at=_moved_by(0.08, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=SIGNALS,
    )  # fmt: skip
    assert _locks(pairs, ephs, change_limit=0.03) == [12, 12]


@pytest.mark.parametrize("options", [{}, {"slips": "restart"}])
def test_the_limit_changes_nothing_without_a_flag_or_in_restart_mode(options):
    ephs = _two_systems()
    flags = (lambda k: {G03: 1} if k == 6 else None) if options else None
    pairs = _l1_run(
        np.random.default_rng(84), ephs, 12, amb_at=_moved_by(0.08, 6), lli_at=flags, signals=BOTH
    )
    options = {"slips": "check", **options}
    plain, limited = _rtk(ephs, **options), _rtk(ephs, change_limit=0.03, **options)
    moved = set()
    for pair in pairs:
        a, b = plain.step(*pair), limited.step(*pair)
        assert (a.status, a.ratio, a.nb, a.ns) == (b.status, b.ratio, b.nb, b.ns)
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.float_position, b.float_position)
        moved |= limited.moved
    # judged in check mode only (and there the pair did stand out)
    assert moved == (set() if options["slips"] == "restart" else {G03, G03_2})
    assert bool(limited.residuals) == (options["slips"] == "check")


def test_a_second_signal_that_moved_is_not_settled_through_a_third(monkeypatch):
    # three signals, the first two flagged, the third settles the first after the second
    # was looked at. The second moved by 4 cm on its own: it stays in question
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)
    ephs = _two_systems()
    signals = ("G1C", "G2W", "G5Q", "E1C", "E5Q")
    keys = [("G03", sig) for sig in signals[:3]]
    pairs = _l1_run(
        np.random.default_rng(85), ephs, 12, amb_at=_moved_by(0.04, 6, keys=(G03_2,)),
        lli_at=lambda k: {G03: 1, G03_2: 1} if k == 6 else None, signals=signals,
    )  # fmt: skip
    assert _locks(pairs, ephs, keys) == [12, 12, 12]
    assert _locks(pairs, ephs, keys, change_limit=0.03) == [12, 6, 12]


def test_the_change_is_taken_over_an_epoch_without_a_solution():
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(86), ephs, 12, amb_at=_moved_by(0.08, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=BOTH,
    )  # fmt: skip
    # no base at 5 s: the epoch ends before the ambiguities, and 6 s is compared with 4 s
    for limit, lock in ((0.0, 11), (0.03, 6)):
        rtk = _rtk(ephs, slips="check", change_limit=limit)
        for k, (rover, base) in enumerate(pairs):
            rtk.step(rover, None if k == 5 else base)
        assert (rtk.biases[G03].lock, rtk.biases[G03_2].lock) == (lock, 11)


def test_a_satellite_without_a_phase_the_epoch_before_has_no_change():
    # G04 is missing at the base at 5 s: nothing to compare its phases at 6 s with, and the
    # others are judged without it
    ephs = _two_systems()
    gone = {("G04", "G1C"), ("G04", "G2W")}
    pairs = _l1_run(
        np.random.default_rng(87), ephs, 12, amb_at=_moved_by(0.08, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=BOTH,
        drop_at=lambda k: gone if k == 5 else (),
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check", change_limit=0.03)
    for k, pair in enumerate(pairs):
        rtk.step(*pair)
        if k == 6:
            assert rtk.moved == {G03, G03_2}
    assert rtk.biases[G03].lock == 6


def test_backward_a_pair_that_moved_is_not_vouched_for():
    # the move between 5 and 6 s, flagged at 6: backward the filter meets it at 5
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(88), ephs, 12, amb_at=_moved_by(0.08, 6),
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=BOTH,
    )  # fmt: skip
    for limit, lock in ((0.0, 12), (0.03, 6)):
        rtk = _rtk(ephs, slips="check", backward=True, change_limit=limit)
        for k in range(11, -1, -1):
            rtk.step(*pairs[k])
        assert (rtk.biases[G03].lock, rtk.biases[G03_2].lock) == (lock, 12)


def test_a_negative_change_limit_is_refused():
    with pytest.raises(ValueError, match="change_limit"):
        _rtk(constellation(), slips="check", change_limit=-0.01)


def _shifted(obs, keys, meters):
    """The observations with the phases `keys` moved by one length."""
    phase = obs.carrier_phase.copy()
    for sat, sig in keys:
        phase[obs.sats.index(sat), obs.signals.index(sig)] += meters / wavelength(sig)
    return replace(obs, carrier_phase=phase)


def test_a_move_at_the_base_counts_as_one_at_the_rover():
    # the ambiguity is of the difference of the two receivers' phases: both of the base's
    # phases of G03 move by 8 cm at 6 s, unflagged, while the rover flags its first signal
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(89), ephs, 12, lli_at=lambda k: {G03: 1} if k == 6 else None,
        signals=BOTH,
    )  # fmt: skip
    pairs = [(r, _shifted(b, (G03, G03_2), 0.08) if k >= 6 else b) for k, (r, b) in enumerate(pairs)]
    assert _locks(pairs, ephs) == [12, 12]
    assert _locks(pairs, ephs, change_limit=0.03) == [6, 12]


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("receiver", ["rover", "base"])
def test_phases_either_receiver_flags_take_no_part_in_the_solution(receiver, backward):
    # every GPS phase is flagged at 6 s by one receiver and stands somewhere else, each by
    # its own length. The twelve Galileo phases alone give the verdict: the GPS phases on
    # their frequency moved; G2W, with no unflagged phase on its frequency, is not judged.
    # Were the flagged phases part of the solution there would be no majority and no verdict
    ephs = _two_systems()
    gps = [(f"G{n:02d}", sig) for n in range(1, 7) for sig in ("G1C", "G2W")]
    after = dict(AMB_R)
    for i, key in enumerate(gps):
        after[key] += 2 + i
    flags = {key: 1 for key in gps}
    pairs = _l1_run(
        np.random.default_rng(90), ephs, 12, amb_at=lambda k: after if k >= 6 else AMB_R,
        lli_at=(lambda k: flags if k == 6 else None) if receiver == "rover" else None, signals=BOTH,
    )  # fmt: skip
    if receiver == "base":
        r, b = pairs[6]
        lli = b.lli.copy()
        lli[[b.sats.index(sat) for sat, _ in gps], [b.signals.index(sig) for _, sig in gps]] = 1
        pairs[6] = (r, replace(b, lli=lli))
    rtk = _rtk(ephs, slips="check", change_limit=0.03, backward=backward)
    seen = {}
    for k in range(11, -1, -1) if backward else range(12):
        rtk.step(*pairs[k])
        seen[k] = set(rtk.moved)
    # the move lies between 5 and 6 s: forward it is met at 6, backward at 5
    at = 5 if backward else 6
    assert seen[at] == {key for key in gps if key[1] == "G1C"}
    assert all(not moved for k, moved in seen.items() if k != at)


def test_a_phase_with_the_half_cycle_bit_is_as_good_as_any_for_the_solution():
    # every phase carries the half-cycle bit throughout, as L1 C/A phases do before the
    # polarity is known: that is no lost lock, and the verdict is given
    ephs = _two_systems()
    keys = [key for key in AMB_R if key[1] in BOTH]
    pairs = _l1_run(
        np.random.default_rng(91), ephs, 12, amb_at=_moved_by(0.08, 6),
        lli_at=lambda k: {key: 3 if key == G03 and k == 6 else 2 for key in keys}, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs) == [12, 12]
    assert _locks(pairs, ephs, change_limit=0.03) == [6, 12]


def test_a_change_is_taken_from_the_epoch_before_and_no_earlier(monkeypatch):
    # the pair moves at 6 s without a flag and stays there; a flag at 8 s finds nothing
    # that changed since 7 s, and the other signal vouches
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(92), ephs, 12, amb_at=_moved_by(0.08, 6),
        lli_at=lambda k: {G03: 1} if k == 8 else None, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs, change_limit=0.03) == [12, 12]


def test_a_turn_of_the_antenna_moves_no_phase(monkeypatch):
    # every phase of the rover by half a cycle at 6 s: 9.5 cm on one frequency, 12.2 and
    # 12.7 cm on the others, a clock term apiece. One clock for all would leave them 1.6 cm
    # out, which a limit of 1 cm shows
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)
    ephs = _two_systems()
    turned = {key: n + 0.5 for key, n in AMB_R.items()}
    pairs = _l1_run(
        np.random.default_rng(93), ephs, 12, amb_at=lambda k: turned if k >= 6 else AMB_R,
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs, change_limit=0.01) == [12, 12]


def test_the_engine_hands_the_verdict_each_phase_with_its_frequency_and_satellite(monkeypatch):
    import pyrx.rtk as rtk_module

    calls = []
    judge = rtk_module._moved
    monkeypatch.setattr(rtk_module, "_moved", lambda *a: calls.append(a) or judge(*a))
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(94), ephs, 3, lli_at=lambda k: {G03: 1} if k == 2 else None,
        signals=BOTH,
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check", change_limit=0.03)
    kept = []
    for pair in pairs:
        rtk.step(*pair)
        kept.append(rtk.residuals)
    assert len(calls) == 2  # nothing to compare the first epoch with
    change, los, band, sat, clean, limit = calls[-1]
    keys = list(rtk.residuals)
    assert len(change) == len(keys) == 24 and limit == 0.03
    for i, (name, sig) in enumerate(keys):
        same_sat = [j for j, key in enumerate(keys) if key[0] == name]
        assert [j for j in range(24) if sat[j] == sat[i]] == same_sat
        np.testing.assert_array_equal(los[i], los[same_sat[0]])
        same_band = [j for j, key in enumerate(keys) if wavelength(key[1]) == wavelength(sig)]
        assert [j for j in range(24) if band[j] == band[i]] == same_band
        assert clean[i] == ((name, sig) != G03)
    assert len(set(band)) == 3  # L1 and E1 are one frequency
    np.testing.assert_allclose(np.linalg.norm(los, axis=1), 1.0)
    assert np.abs(change).max() < 5.0 and np.abs(change - change.mean()).max() > 1e-4
    # each phase with its own change since the epoch before, and with the line of sight of
    # this epoch: the residuals the engine keeps are those of the last step
    for i, key in enumerate(keys):
        assert change[i] == kept[2][key][0] - kept[1][key][0]
        np.testing.assert_array_equal(los[i], kept[2][key][1])
        assert (los[i] != kept[1][key][1]).any()


def test_a_pair_whose_difference_stepped_restarts_both_as_before():
    # the first signal alone by 8 cm, flagged: the geometry-free test comes first and
    # restarts both, with the limit as without
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(95), ephs, 12, amb_at=_moved_by(0.08, 6, keys=(G03,)),
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs) == [6, 6]
    assert _locks(pairs, ephs, change_limit=0.03) == [6, 6]


def test_a_move_that_is_back_at_the_next_epoch_costs_the_flagged_phase_its_ambiguity():
    # out by 8 cm at 6 s and back at 7 s, the common case in a fade: the flagged phase is
    # not vouched for at 6 s, fails the test and begins again on the displaced phase
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(96), ephs, 12, amb_at=_moved_by(0.08, 6, 7),
        lli_at=lambda k: {G03: 1} if k == 6 else None, signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs) == [12, 12]
    assert _locks(pairs, ephs, change_limit=0.03) == [6, 12]


def test_no_verdict_across_a_long_step():
    # 30 s between epochs: the lines of sight turn, and with them what an error of the
    # epoch's seed position puts into the residuals, by centimeters; no verdict is given
    ephs = _two_systems()
    pairs = [
        _pair(
            np.random.default_rng(97 + k), ephs, TOW + 30.0 * k, signals=BOTH,
            amb_r=_moved_by(0.08, 6)(k), lli={G03: 1} if k == 6 else None,
        )
        for k in range(12)
    ]  # fmt: skip
    assert _locks(pairs, ephs, change_limit=0.03) == [12, 12]


@pytest.mark.parametrize("at,locks", [(5, [12, 12]), (7, [5, 12])])
def test_no_verdict_where_the_base_epoch_changed_by_another_step_than_the_rover(at, locks):
    # a base every 5 s under a 1 Hz rover: between base epochs the base's part of the
    # difference stands still and the verdict is the rover's; where the base epoch changes,
    # 5 s of the base's own residuals are in the change and none is given
    ephs = _two_systems()
    pairs = _l1_run(
        np.random.default_rng(98), ephs, 12, amb_at=_moved_by(0.08, at),
        lli_at=lambda k: {G03: 1} if k == at else None, signals=BOTH,
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check", change_limit=0.03)
    rover, base = [r for r, _ in pairs], [b for k, (_, b) in enumerate(pairs) if k % 5 == 0]
    for r, b in pair_epochs(rover, base, rtk.opt.max_age):
        rtk.step(r, b)
    assert [rtk.biases[key].lock for key in (G03, G03_2)] == locks


def test_a_flag_of_an_epoch_that_ended_early_keeps_its_phase_out_of_the_solution(monkeypatch):
    # G05's first signal slips and is flagged at 5 s, an epoch without a base: at 6 s its
    # indicator is clear again, but its change since 4 s holds the slip
    import pyrx.rtk as rtk_module

    calls = []
    judge = rtk_module._moved
    monkeypatch.setattr(rtk_module, "_moved", lambda *a: calls.append(a) or judge(*a))
    ephs = _two_systems()
    slipped = ("G05", "G1C")
    pairs = _l1_run(
        np.random.default_rng(99), ephs, 7, amb_at=_offset(slipped, 7.0, 5),
        lli_at=lambda k: {slipped: 1} if k == 5 else None, signals=BOTH,
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check", change_limit=0.03)
    for k, (rover, base) in enumerate(pairs):
        rtk.step(rover, None if k == 5 else base)
    clean = dict(zip(rtk.residuals, calls[-1][4], strict=True))
    assert not clean[slipped] and sum(clean.values()) == 23


def test_an_epoch_without_a_verdict_is_still_the_one_the_next_is_compared_with(monkeypatch):
    # at 5 s the pair moves and fourteen other phases carry a flag: too few unflagged for a
    # verdict. A flag on the pair at 6 s is judged against 5 s, where nothing changed since
    import pyrx.rtk as rtk_module

    monkeypatch.setattr(rtk_module, "CHECK_SIGMA", 0.0)
    ephs = _two_systems()
    flagged = ("G01", "G02", "G05", "G06", "E11", "E12", "E16")
    many = [(sat, sig) for sat in flagged for sig in BOTH if sig[0] == sat[0]]
    pairs = _l1_run(
        np.random.default_rng(100), ephs, 12, amb_at=_moved_by(0.08, 5),
        lli_at=lambda k: dict.fromkeys(many, 1) if k == 5 else {G03: 1} if k == 6 else None,
        signals=BOTH,
    )  # fmt: skip
    assert _locks(pairs, ephs, change_limit=0.03) == [12, 12]


def test_a_change_of_the_half_cycle_bit_keeps_a_phase_out_of_the_solution(monkeypatch):
    # the bit comes up on one phase at 3 s and goes at 6 s: the engine reads either change as
    # a slip, and the phase's changes into 3 s and into 6 s are not clean; with the bit
    # standing, at 4 and 5 s, they are
    import pyrx.rtk as rtk_module

    calls = []
    judge = rtk_module._moved
    monkeypatch.setattr(rtk_module, "_moved", lambda *a: calls.append(a) or judge(*a))
    ephs = _two_systems()
    key = ("G05", "G1C")
    pairs = _l1_run(
        np.random.default_rng(101), ephs, 8, lli_at=lambda k: {key: 2} if 3 <= k < 6 else None,
        signals=BOTH,
    )  # fmt: skip
    rtk = _rtk(ephs, slips="check", change_limit=0.03)
    clean = {}
    for k, pair in enumerate(pairs):
        rtk.step(*pair)
        if k:
            clean[k] = dict(zip(rtk.residuals, calls[-1][4], strict=True))
    assert [k for k in clean if not clean[k][key]] == [3, 6]
    assert all(sum(flags.values()) == (23 if k in (3, 6) else 24) for k, flags in clean.items())
    # and at the base alike: its bit up at 3 s only, a change into 3 s and one into 4 s
    other = ("E12", "E5Q")
    r, b = pairs[3]
    lli = b.lli.copy()
    lli[b.sats.index(other[0]), b.signals.index(other[1])] = 2
    pairs[3] = (r, replace(b, lli=lli))
    rtk = _rtk(ephs, slips="check", change_limit=0.03)
    seen = []
    for pair in pairs[:6]:
        rtk.step(*pair)
        seen.append(dict(zip(rtk.residuals, calls[-1][4], strict=True))[other])
    assert seen[2:] == [True, False, False, True]
