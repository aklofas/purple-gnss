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


def _searched(fix, sat):
    return any(sat in (a.ref, a.sat) for a in fix.ambiguities)


def test_a_restarted_ambiguity_waits_min_lock_updates_before_the_search():
    # RTKLIB's pos2-arlockcnt: an ambiguity begun again counts up from -min_lock and is
    # searched once it is above zero; one seen for the first time starts at zero
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
    # half a cycle off is no integer: the receiver marks such a phase and it is not searched
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


@pytest.mark.parametrize("limit,kept", [(0.10, False), (0.125, True)])
def test_the_fit_must_be_predicted_sharply(monkeypatch, limit, kept):
    # the phase of the one flagged satellite of the run above is predicted within 0.107 cycle
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
