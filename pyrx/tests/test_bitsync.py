import numpy as np
from test_lnav import SF1, SF2, SF3

from pyrx.bitsync import BitStream, bit_sync, frame_sync
from pyrx.lnav import encode_subframe

rng = np.random.default_rng(1)


def _frame_bits(tow0=5044, n_sf=3, d29=0, d30=0):
    out = []
    for i, f in zip(range(1, n_sf + 1), (SF1, SF2, SF3, SF1, SF2)):
        sf = encode_subframe(i if i <= 3 else 1, (tow0 + i - 1) % 100800, f, d29, d30)
        d29, d30 = int(sf[28 + 270]), int(sf[29 + 270])
        out.append(sf)
    return np.concatenate(out)


def _prompts(bits, phase, cn0_scale=0.25, lead=0):
    """Per-ms prompt I: +/-1 per 20 ms with noise, bit edges at epoch phase (mod 20)."""
    sym = np.repeat(np.where(bits == 1, -1.0, 1.0), 20)  # 1 -> negative I, matching the test
    p = np.concatenate([np.zeros(phase), sym, np.zeros(lead)])
    return p + cn0_scale * rng.standard_normal(len(p))


def test_bit_sync_finds_phase_and_bits():
    bits = rng.integers(0, 2, 200).astype(np.uint8)
    stream = bit_sync(_prompts(bits, phase=7), start_epoch=1500)
    assert stream is not None and stream.phase == 7
    assert stream.epoch[0] == 1507
    np.testing.assert_array_equal(stream.bits[:200], bits)


def test_bit_sync_rejects_no_transitions():
    assert bit_sync(np.ones(2000) + 0.1 * rng.standard_normal(2000), 0) is None


def test_frame_sync_finds_subframes_and_tow():
    bits = _frame_bits()
    stream = bit_sync(_prompts(bits, phase=3, lead=40), start_epoch=1500)
    sfs = frame_sync(stream, prn=4)
    assert [s.sf_id for s in sfs] == [1, 2, 3]
    assert [s.tow_s for s in sfs] == [5044 * 6 - 6, 5045 * 6 - 6, 5046 * 6 - 6]
    assert sfs[0].epoch == 1503 and sfs[1].epoch == 1503 + 300 * 20


def test_decode_inverted_stream():
    bits = _frame_bits()
    stream = bit_sync(-_prompts(bits, phase=0), start_epoch=0)
    sfs = frame_sync(stream, prn=4)
    assert [s.sf_id for s in sfs] == [1, 2, 3] and sfs[1].fields["iode"] == 149


def test_frame_sync_rejects_chance_preamble():
    # random bits containing the preamble pattern but no valid parity: nothing decoded
    bits = rng.integers(0, 2, 1200).astype(np.uint8)
    bits[100:108] = [1, 0, 0, 0, 1, 0, 1, 1]
    bits[400:408] = [1, 0, 0, 0, 1, 0, 1, 1]
    stream = bit_sync(_prompts(bits, phase=0), 0)
    assert frame_sync(stream) == []


def test_decode_drops_corrupted_subframe():
    bits = _frame_bits()
    bits[300 + 150] ^= 1  # inside subframe 2
    stream = bit_sync(_prompts(bits, phase=0), 0)
    assert [s.sf_id for s in frame_sync(stream)] == [1, 3]


def test_frame_sync_handles_preamble_pair_at_stream_end():
    # a chance preamble pair whose second copy ends inside the last 30 bits must be
    # rejected, not read past the end of the stream
    bits = rng.integers(0, 2, 400).astype(np.uint8)
    bits[80:88] = [1, 0, 0, 0, 1, 0, 1, 1]
    bits[380:388] = [1, 0, 0, 0, 1, 0, 1, 1]
    stream = BitStream(bits, np.arange(400) * 20, 0, 1.0)
    assert frame_sync(stream) == []


def test_bit_sync_accepts_weak_signal():
    # noise sign flips spread over all 20 phases; the edge phase must still win by contrast
    bits = rng.integers(0, 2, 1400).astype(np.uint8)
    stream = bit_sync(_prompts(bits, phase=7, cn0_scale=0.6), start_epoch=0)
    assert stream is not None and stream.phase == 7
    np.testing.assert_array_equal(stream.bits[:1400], bits)


def test_bit_sync_rejects_noise_only():
    assert bit_sync(rng.standard_normal(2000), 0) is None


def test_frame_sync_reports_polarity():
    bits = _frame_bits()
    s = bit_sync(_prompts(bits, phase=0), 0)
    frame_sync(s, prn=4)
    assert s.inverted is False
    s = bit_sync(-_prompts(bits, phase=0), 0)
    frame_sync(s, prn=4)
    assert s.inverted is True
    s = bit_sync(_prompts(rng.integers(0, 2, 1200).astype(np.uint8), phase=0), 0)
    assert frame_sync(s) == [] and s.inverted is None


def test_frame_sync_accepts_the_week_rollover_pair():
    # the TOW count goes 100799 -> 0 once a week; a start verified on that pair must not be
    # rejected, and the decoded seconds of week wrap with it
    bits = _frame_bits(tow0=100799)
    stream = bit_sync(_prompts(bits, phase=0), 0)
    sfs = frame_sync(stream, prn=4)
    assert [s.tow_s for s in sfs] == [604788, 604794, 0]


def test_decode_drops_a_parity_clean_subframe_with_the_wrong_tow_count():
    # a bit-error pattern can leave a word parity-clean with a changed TOW count; the
    # count of each block is fixed by its position in the sequence, so such a block is
    # dropped on its own instead of poisoning the channel's time tags
    tow0 = 5044
    out, d29, d30 = [], 0, 0
    for i, count in ((1, tow0), (2, tow0 + 1), (3, tow0 + 3), (1, tow0 + 3), (2, tow0 + 4)):
        sf = encode_subframe(i, count, {1: SF1, 2: SF2, 3: SF3}[i], d29, d30)
        d29, d30 = int(sf[298]), int(sf[299])
        out.append(sf)
    stream = bit_sync(_prompts(np.concatenate(out), phase=0), 0)
    sfs = frame_sync(stream, prn=4)
    assert [(s.sf_id, s.tow_s) for s in sfs] == [
        (1, tow0 * 6 - 6), (2, (tow0 + 1) * 6 - 6), (1, (tow0 + 3) * 6 - 6), (2, (tow0 + 4) * 6 - 6)
    ]  # fmt: skip
