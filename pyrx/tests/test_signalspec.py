import dataclasses

import numpy as np
import pytest

from pyrx.codes import (
    GAL_CS20,
    GAL_CS25,
    NH10,
    NH20,
    gal_cs100,
    gal_e1b,
    gal_e1c,
    gal_e5ai,
    gal_e5aq,
    gps_ca,
    gps_l5i,
    gps_l5q,
)
from pyrx.signals import FREQUENCY
from pyrx.signalspec import E1, E5A, L1CA, L5, SIGNALS


def test_the_signals():
    assert SIGNALS == {"L1CA": L1CA, "L5": L5, "E5A": E5A, "E1": E1}
    assert [s.rinex for s in (L1CA, L5, E5A)] == ["G1C", "G5Q", "E5Q"]
    assert [s.carrier for s in (L1CA, L5, E5A)] == [FREQUENCY["G1"], FREQUENCY["G5"], FREQUENCY["E5"]]
    assert [s.chip_rate for s in (L1CA, L5, E5A)] == [1.023e6, 10.23e6, 10.23e6]
    assert [s.code_length for s in (L1CA, L5, E5A)] == [1023, 10230, 10230]
    assert [s.pilot for s in (L1CA, L5, E5A)] == [False, True, True]
    assert [s.symbol_epochs for s in (L1CA, L5, E5A)] == [0, 10, 20]
    assert [s.prns for s in (L1CA, L5, E5A)] == [range(1, 33), range(1, 33), range(1, 37)]
    # RINEX 3.05 Table A23: L5Q is a quarter cycle from the band's reference signal L5I
    assert [s.phase_align for s in (L1CA, L5)] == [0.0, -0.25]


def test_codes_of_the_signals():
    assert np.array_equal(L1CA.code(7), gps_ca(7))
    assert L1CA.secondary is None and L1CA.data_code is None and L1CA.data_secondary is None
    assert np.array_equal(L5.code(6), gps_l5q(6)) and np.array_equal(L5.data_code(6), gps_l5i(6))
    assert np.array_equal(L5.secondary(6), NH20) and np.array_equal(L5.data_secondary(6), NH10)
    assert np.array_equal(E5A.code(2), gal_e5aq(2)) and np.array_equal(E5A.data_code(2), gal_e5ai(2))
    assert np.array_equal(E5A.secondary(2), gal_cs100(2))
    assert np.array_equal(E5A.data_secondary(2), GAL_CS20)


def test_a_symbol_spans_the_data_secondary_code():
    for spec in (L5, E5A):
        assert len(spec.data_secondary(1)) == spec.symbol_epochs
        assert len(spec.secondary(1)) % spec.symbol_epochs == 0
        assert abs(spec.data_axis) == pytest.approx(1.0) and spec.data_axis.real == 0.0
    assert L1CA.data_axis == 0j


@pytest.mark.parametrize("spec", [L5, E5A])
def test_secondary_codes_cannot_be_written_through(spec):
    # every PRN of a signal is handed the same array for a shared code: a write through one
    # would change the code of all
    prn = spec.prns[0]
    for code in (spec.secondary(prn), spec.data_secondary(prn)):
        before = code.copy()
        with pytest.raises(ValueError, match="read-only"):
            code[0] = -code[0]
        np.testing.assert_array_equal(code, before)


def test_the_galileo_e1_signal():
    # the pilot E1-C leads; its 4092-chip code lasts 4 ms, four of the tracker's epochs, and
    # every chip is one cycle of the BOC(1,1) subcarrier
    assert (E1.name, E1.rinex, E1.carrier) == ("E1", "E1C", FREQUENCY["E1"])
    assert (E1.chip_rate, E1.code_length, E1.epochs) == (1.023e6, 4092, 4)
    assert E1.prns == range(1, 37) and E1.pilot
    assert E1.subcarrier == (1, -1) and E1.replica_rate == 2.046e6
    assert np.array_equal(E1.code(11), gal_e1c(11)) and np.array_equal(E1.data_code(11), gal_e1b(11))
    replica = E1.replica(11)
    assert replica.shape == (8184,) and replica.dtype == np.int8
    assert np.array_equal(replica[0::2], gal_e1c(11)) and np.array_equal(replica[1::2], -gal_e1c(11))
    assert np.array_equal(E1.data_replica(11)[0::2], gal_e1b(11))
    # the 25-chip secondary code, each chip over the four epochs of a code period
    assert np.array_equal(E1.secondary(11), np.repeat(GAL_CS25, 4))
    # a symbol of E1-B is one code period, with no secondary code on it
    assert E1.symbol_epochs == 4 and E1.data_secondary(11).tolist() == [1]
    # ICD issue 2.2, Eq. 12: the two components on one axis with opposite signs
    assert E1.data_axis == -1
    # RINEX 3.05 Table A23: L1C half a cycle from the band's reference signal L1B
    assert E1.phase_align == 0.5
    # side peaks half a chip either side; the discriminator nine times as steep as a plain
    # code's at a quarter chip
    assert E1.side_taps == (0.5, -0.5) and E1.dll_slope == 9.0


def test_a_replica_is_of_the_code_it_is_asked_for():
    # a code function may build its array anew on every call, and a new array can take the
    # place in memory of one that is gone: the replica kept is found by the code's chips
    fresh = dataclasses.replace(E1, code=lambda prn: np.array(gal_e1c(prn)))
    for prn in range(1, 37):
        assert np.array_equal(fresh.replica(prn)[0::2], gal_e1c(prn)), prn
    assert E1.replica(11) is E1.replica(11)


def test_plain_signals_are_their_own_replica():
    for spec, prn in ((L1CA, 3), (L5, 6), (E5A, 2)):
        assert spec.epochs == 1 and spec.subcarrier == (1,) and spec.replica_rate == spec.chip_rate
        assert spec.replica(prn) is spec.code(prn)
        assert spec.side_taps == () and spec.dll_slope == 1.0
    assert L5.data_replica(6) is L5.data_code(6)
