import numpy as np
import pytest

from pyrx.codes import GAL_CS20, NH10, NH20, gal_cs100, gal_e5ai, gal_e5aq, gps_ca, gps_l5i, gps_l5q
from pyrx.signals import FREQUENCY
from pyrx.signalspec import E5A, L1CA, L5, SIGNALS


def test_the_three_signals():
    assert SIGNALS == {"L1CA": L1CA, "L5": L5, "E5A": E5A}
    assert [s.rinex for s in (L1CA, L5, E5A)] == ["G1C", "G5Q", "E5Q"]
    assert [s.carrier for s in (L1CA, L5, E5A)] == [FREQUENCY["G1"], FREQUENCY["G5"], FREQUENCY["E5"]]
    assert [s.chip_rate for s in (L1CA, L5, E5A)] == [1.023e6, 10.23e6, 10.23e6]
    assert [s.code_length for s in (L1CA, L5, E5A)] == [1023, 10230, 10230]
    assert [s.pilot for s in (L1CA, L5, E5A)] == [False, True, True]
    assert [s.symbol_epochs for s in (L1CA, L5, E5A)] == [0, 10, 20]
    assert [s.prns for s in (L1CA, L5, E5A)] == [range(1, 33), range(1, 33), range(1, 37)]


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
