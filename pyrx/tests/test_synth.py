import numpy as np
import pytest

from pyrx import synth
from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca, sample_code
from pyrx.correlator import correlate
from pyrx.signalspec import E5A, L5
from pyrx.synth import synth_l1ca, synth_pilot

FS = 4e6
N = round(FS * CA_PERIOD)


def _prompt_signs(x, code_offset, n_ms=60):
    code = sample_code(gps_ca(7), CA_CHIP_RATE, FS, N)
    out = []
    for m in range(n_ms):  # 60 ms = 3 bits
        start = code_offset + m * N
        out.append(np.sign(np.real(np.dot(x[start : start + N], code))))
    return np.array(out)


def test_synth_without_bits_is_constant_sign():
    x = synth_l1ca(7, FS, 0.0, 0.0, 1000, 80 * N, 60.0, True)
    assert np.all(_prompt_signs(x, 1000) == 1)


def test_synth_data_bits_flip_every_20ms():
    x = synth_l1ca(7, FS, 0.0, 0.0, 1000, 80 * N, 60.0, True, seed=3, data_bits=True)
    s = _prompt_signs(x, 1000)
    assert np.all(s[0:20] == s[0]) and np.all(s[20:40] == s[20]) and np.all(s[40:60] == s[40])
    assert len(set(s[[0, 20, 40]])) > 1  # seed 3 must produce at least one flip


def test_synth_explicit_bits():
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    x = synth_l1ca(7, FS, 0.0, 0.0, 1000, 100 * N, 60.0, True, bits=bits)
    s = _prompt_signs(x, 1000, n_ms=80)
    assert list(s[[0, 20, 40, 60]]) == [1, -1, -1, 1]
    with pytest.raises(ValueError):
        synth_l1ca(7, FS, 0.0, 0.0, 1000, 100 * N, 60.0, True, data_bits=True, bits=bits)


PILOT_FS = 12e6
PILOT_N = round(PILOT_FS * 1e-3)


def _pilot_prompts(x, spec, prn, code_offset, epochs, other_prn=None):
    """Prompt sums of the pilot and data codes (and of another PRN's pilot code, for the
    noise level) over `epochs` code periods from `code_offset`, zero Doppler."""
    out = np.empty((epochs, 3), dtype=np.complex128)
    codes = (spec.code(prn), spec.data_code(prn), spec.code(other_prn or prn))
    for e in range(epochs):
        block = x[code_offset + e * PILOT_N : code_offset + (e + 1) * PILOT_N]
        for i, code in enumerate(codes):
            out[e, i] = correlate(block, PILOT_FS, 0.0, 0.0, code, spec.chip_rate, 0.0, np.zeros(1))[0]
    return out


@pytest.mark.parametrize("spec,prn", [(L5, 6), (E5A, 2)], ids=["L5", "E5A"])
def test_synth_pilot_components(spec, prn):
    symbols = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    sec_index, epochs = 3, 45
    x = synth_pilot(spec, prn, PILOT_FS, 0.0, 0.0, 1000, 1000 + (epochs + 1) * PILOT_N, 70.0,
                    symbols=symbols, sec_index=sec_index)  # fmt: skip
    assert x.dtype == np.complex64
    p = _pilot_prompts(x, spec, prn, 1000, epochs)
    e = sec_index + np.arange(epochs)
    sec, dsec = spec.secondary(prn), spec.data_secondary(prn)
    # the pilot under its secondary code, on the axis that puts a symbol 0 on data_axis
    pilot = p[:, 0] * sec[e % len(sec)] * spec.data_axis
    assert np.all(pilot.real > 0.99 * np.abs(p[:, 0]))
    # the data component under its secondary code and the symbols, in quadrature to the pilot
    sign = 1 - 2 * symbols[e // spec.symbol_epochs]
    data = p[:, 1] * dsec[e % len(dsec)] * sign
    assert np.all(data.real > 0.99 * np.abs(p[:, 1]))
    ratio = np.mean(np.abs(p[:, 1]) ** 2) / np.mean(np.abs(p[:, 0]) ** 2)
    assert abs(10 * np.log10(ratio)) < 0.2


def test_synth_pilot_cn0_is_the_pilots():
    epochs = 1000
    x = synth_pilot(L5, 6, PILOT_FS, 0.0, 0.0, 500, 500 + (epochs + 1) * PILOT_N, 45.0, seed=3)
    p = _pilot_prompts(x, L5, 6, 500, epochs, other_prn=11)
    signal, noise = np.mean(np.abs(p[:, 0]) ** 2), np.mean(np.abs(p[:, 2]) ** 2)
    assert 10 * np.log10((signal - noise) / noise / 1e-3) == pytest.approx(45.0, abs=0.5)


def test_synth_pilot_default_symbols_follow_the_seed():
    n = 60 * PILOT_N
    a = synth_pilot(L5, 6, PILOT_FS, 0.0, 0.0, 0, n, 70.0, seed=1)
    b = synth_pilot(L5, 6, PILOT_FS, 0.0, 0.0, 0, n, 70.0, seed=1)
    c = synth_pilot(L5, 6, PILOT_FS, 0.0, 0.0, 0, n, 70.0, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    signs = np.sign(_pilot_prompts(a, L5, 6, 0, 59)[:, 1].real * L5.data_secondary(6)[np.arange(59) % 10])
    per_symbol = signs[:50].reshape(5, 10)
    assert np.all(per_symbol == per_symbol[:, :1])  # constant within a symbol
    assert len(set(per_symbol[:, 0])) == 2  # and the symbols differ


def test_synth_pilot_is_independent_of_the_block_size(monkeypatch):
    args = (E5A, 2, PILOT_FS, 0.0, 1234.0, 700, 30 * PILOT_N, 45.0)
    whole = synth_pilot(*args, seed=5, doppler_rate=40.0)
    monkeypatch.setattr(synth, "BLOCK", 50_000)
    assert np.array_equal(synth_pilot(*args, seed=5, doppler_rate=40.0), whole)


def test_synth_pilot_doppler_rate_moves_the_carrier():
    n = 200 * PILOT_N
    x = synth_pilot(L5, 6, PILOT_FS, 0.0, 100.0, 0, n, 70.0, doppler_rate=500.0, symbols=np.zeros(1))
    # 0.15 s in, the carrier is at 100 + 500 * 0.15 = 175 Hz: the pilot prompt of consecutive
    # epochs turns by 175 Hz * 1 ms, once the secondary code is wiped
    k0 = 150
    blocks = [x[(k0 + i) * PILOT_N : (k0 + i + 1) * PILOT_N] for i in range(2)]
    # the code has slipped by less than a hundredth of a chip in 0.15 s; correlate at phase 0
    p = [correlate(b, PILOT_FS, 0.0, 0.0, L5.code(6), L5.chip_rate, 0.0, np.zeros(1))[0] for b in blocks]
    sec = L5.secondary(6)
    turn = np.angle(p[1] * sec[(k0 + 1) % 20] * np.conj(p[0] * sec[k0 % 20])) / (2 * np.pi)
    assert turn == pytest.approx(0.175, abs=0.003)
