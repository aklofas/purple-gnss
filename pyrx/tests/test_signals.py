import numpy as np
import pytest

from pyrx.rinex import (
    read_rinex_header,
    read_rinex_obs,
    read_rinex_observations,
    write_rinex_observations,
)
from pyrx.signals import Observations, signal_frequency, to_epoch, wavelength

HEADER = """\
     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE
TEST                                                        MARKER NAME
                    SEPT POLARX5TR      5.4.0               REC # / TYPE / VERS
                    TRM59800.00     NONE                    ANT # / TYPE
  4628684.7755   119996.7157  4372109.7032                  APPROX POSITION XYZ
        0.4410        0.0000        0.0000                  ANTENNA: DELTA H/E/N
G    8 C1C L1C D1C S1C C2W L2W D2W S2W                      SYS / # / OBS TYPES
E    4 C1C L1C C5Q L5Q                                      SYS / # / OBS TYPES
  2021    12    26     0     0    0.0000000     GPS         TIME OF FIRST OBS
                                                            END OF HEADER
"""


def _rec(sat, *fields):
    """One observation record: each field is a value or (value, lli); None leaves it blank."""
    out = sat
    for f in fields:
        v, lli = f if isinstance(f, tuple) else (f, 0)
        out += " " * 16 if v is None else f"{v:14.3f}{lli or ' '} "
    return out.rstrip() + "\n"


RINEX = (
    HEADER
    + "> 2021 12 26 00 00  0.0000000  0  3\n"
    + _rec(
        "G04",
        22235454.255,
        (-116848778.204, 1),
        421.142,
        43.3,
        22235459.1,
        (-91052200.5, 2),
        -328.1,
        30.0,
    )
    + _rec("G16", 20653742.563, 108536284.235, -698.28, 46.9)
    + _rec("E11", 24000000.0, 126000000.0, 24000001.0, 94000000.0)
    + "> 2021 12 26 00 00 30.0000000  0  1\n"
    + _rec("G04", 22235373.83, -116848357.26, 420.9, 43.2, 22235378.7, -91051872.5, -327.9, 30.1)
)


def test_signal_frequency_and_wavelength():
    assert signal_frequency("G2W") == 1227.60e6 and signal_frequency("E5Q") == 1176.45e6
    assert wavelength("G1C") == pytest.approx(0.190293672798, rel=1e-9)
    with pytest.raises(KeyError):
        signal_frequency("G6X")


def test_read_rinex_observations_two_bands(tmp_path):
    (tmp_path / "t.obs").write_text(RINEX)
    a, b = read_rinex_observations(tmp_path / "t.obs", ("G1C", "G2W", "E1C", "E5Q"))
    assert (a.week, a.tow, a.sats) == (2190, 0.0, ("E11", "G04", "G16"))
    assert a.signals == ("G1C", "G2W", "E1C", "E5Q")
    g04 = a.sats.index("G04")
    assert a.pseudorange[g04, 0] == 22235454.255 and a.carrier_phase[g04, 1] == -91052200.5
    assert a.lli[g04, 0] == 1 and a.lli[g04, 1] == 2 and a.lli[a.sats.index("G16"), 0] == 0
    e11 = a.sats.index("E11")
    assert a.pseudorange[e11, 2] == 24000000.0 and np.isnan(a.pseudorange[e11, 0])
    assert np.isnan(a.pseudorange[a.sats.index("G16"), 1])  # no L2 on G16
    assert b.tow == 30.0 and b.sats == ("G04",)


def test_read_rinex_observations_missing_signal_is_nan(tmp_path):
    (tmp_path / "t.obs").write_text(RINEX)
    [a, _] = read_rinex_observations(tmp_path / "t.obs", ("G1C", "G5Q"))
    assert a.signals == ("G1C", "G5Q") and np.all(np.isnan(a.pseudorange[:, 1]))
    assert a.sats == ("G04", "G16")  # no Galileo signal requested, so E11 is not read
    assert a.lli.dtype.kind == "i" and a.lli[:, 1].tolist() == [0, 0]


def test_to_epoch(tmp_path):
    (tmp_path / "t.obs").write_text(RINEX)
    [a, _] = read_rinex_observations(tmp_path / "t.obs", ("G1C", "E1C"))
    ep = to_epoch(a)
    assert (ep.week, ep.tow, [o.prn for o in ep.obs]) == (2190, 0.0, [4, 16])
    assert ep.obs[0].pseudorange == 22235454.255 and ep.obs[0].doppler == 421.142


def test_read_rinex_header(tmp_path):
    (tmp_path / "t.obs").write_text(RINEX)
    h = read_rinex_header(tmp_path / "t.obs")
    assert h.marker == "TEST" and h.antenna == "TRM59800.00     NONE"
    assert h.receiver.startswith("SEPT POLARX5TR")
    np.testing.assert_array_equal(h.position, [4628684.7755, 119996.7157, 4372109.7032])
    np.testing.assert_array_equal(h.delta_enu, [0.0, 0.0, 0.441])


def test_write_rinex_observations_round_trip(tmp_path):
    (tmp_path / "t.obs").write_text(RINEX)
    signals = ("G1C", "G2W", "E1C", "E5Q")
    epochs = read_rinex_observations(tmp_path / "t.obs", signals)
    write_rinex_observations(tmp_path / "w.obs", epochs, "TEST")
    text = (tmp_path / "w.obs").read_text()
    assert "G    8 C1C L1C D1C S1C C2W L2W D2W S2W" in text and "E    8 C1C L1C D1C S1C C5Q" in text
    assert "nan" not in text
    back = read_rinex_observations(tmp_path / "w.obs", signals)
    assert [b.tow for b in back] == [0.0, 30.0] and back[0].sats == epochs[0].sats
    np.testing.assert_allclose(back[0].pseudorange, epochs[0].pseudorange, atol=5e-4, rtol=0)
    np.testing.assert_allclose(back[0].carrier_phase, epochs[0].carrier_phase, atol=5e-4, rtol=0)
    np.testing.assert_array_equal(back[0].lli, epochs[0].lli)


def test_read_rinex_observations_skips_event_records(tmp_path):
    event = (
        ">                              4  1\n"
        "antenna moved                                               COMMENT\n"
    )
    (tmp_path / "e.obs").write_text(HEADER + event + RINEX[len(HEADER) :])
    epochs = read_rinex_observations(tmp_path / "e.obs", ("G1C",))
    assert [e.tow for e in epochs] == [0.0, 30.0] and epochs[0].sats == ("G04", "G16")


END_OF_HEADER = "".ljust(60) + "END OF HEADER\n"
SCALE_FACTOR = "G   10  1 L1C".ljust(60) + "SYS / SCALE FACTOR\n"


@pytest.mark.parametrize(
    "change, message",
    [
        (("     3.04", "     2.11"), "RINEX 3"),
        (("OBSERVATION DATA", "N: GNSS NAV DATA"), "observation"),
        (("0.0000000     GPS", "0.0000000     GLO"), "time system"),
        ((END_OF_HEADER, SCALE_FACTOR + END_OF_HEADER), "scale factor"),
    ],
)
def test_readers_reject_profiles_they_do_not_implement(tmp_path, change, message):
    (tmp_path / "t.obs").write_text(RINEX.replace(*change, 1))
    with pytest.raises(ValueError, match=message):
        read_rinex_observations(tmp_path / "t.obs", ("G1C",))
    with pytest.raises(ValueError, match=message):
        read_rinex_obs(tmp_path / "t.obs")


def test_reader_and_writer_reject_a_system_they_do_not_handle(tmp_path):
    (tmp_path / "t.obs").write_text(RINEX)
    with pytest.raises(ValueError, match="R1C"):
        read_rinex_observations(tmp_path / "t.obs", ("G1C", "R1C"))
    a = np.array([[22235454.255]])
    glonass = Observations(2190, 0.0, ("R04",), ("R1C",), a, a, a, a, np.zeros((1, 1), dtype=int))
    with pytest.raises(ValueError, match="R"):
        write_rinex_observations(tmp_path / "w.obs", [glonass])


def test_reader_rejects_an_in_file_change_of_observation_types(tmp_path):
    event = (
        ">                              4  1\n"
        + "G    8 L1C C1C D1C S1C C2W L2W D2W S2W".ljust(60)
        + "SYS / # / OBS TYPES\n"
    )
    (tmp_path / "e.obs").write_text(HEADER + event + RINEX[len(HEADER) :])
    with pytest.raises(ValueError, match="observation types"):
        read_rinex_observations(tmp_path / "e.obs", ("G1C",))


def test_reader_accepts_galileo_time_and_rinex_4(tmp_path):
    text = RINEX.replace("     3.04", "     4.02", 1).replace("     GPS", "     GAL", 1)
    (tmp_path / "t.obs").write_text(text)
    assert [e.tow for e in read_rinex_observations(tmp_path / "t.obs", ("G1C",))] == [0.0, 30.0]


def test_reader_sorts_merges_and_normalises_epochs(tmp_path):
    # epochs out of order; the first repeated with one satellite repeated (its first record
    # wins) and one added; a satellite written as 'G 4'; a final epoch missing a record
    body = (
        "> 2021 12 26 00 00 30.0000000  0  1\n"
        + _rec("G04", 22235373.83, -116848357.26, 420.9, 43.2)
        + "> 2021 12 26 00 00  0.0000000  0  2\n"
        + _rec("G 4", 22235454.255, -116848778.204, 421.142, 43.3)
        + _rec("G16", 20653742.563, 108536284.235, -698.28, 46.9)
        + "> 2021 12 26 00 00  0.0000000  0  2\n"
        + _rec("G04", 1.0, 2.0, 3.0, 4.0)
        + _rec("G25", 23000000.0, 120000000.0, 100.0, 40.0)
        + "> 2021 12 26 00 01  0.0000000  0  2\n"
        + _rec("G04", 22235293.4, -116847936.3, 420.7, 43.1)
    )
    (tmp_path / "t.obs").write_text(HEADER + body)
    a, b = read_rinex_observations(tmp_path / "t.obs", ("G1C",))
    assert (a.tow, a.sats) == (0.0, ("G04", "G16", "G25")) and (b.tow, b.sats) == (30.0, ("G04",))
    assert a.pseudorange[0, 0] == 22235454.255 and a.pseudorange[2, 0] == 23000000.0


def test_reader_drops_a_final_record_cut_inside_a_field(tmp_path):
    body = RINEX[len(HEADER) :]
    (tmp_path / "t.obs").write_text(HEADER + body[: body.rindex("G04") + 12])
    [a] = read_rinex_observations(tmp_path / "t.obs", ("G1C",))
    assert a.tow == 0.0


def test_reader_drops_a_final_epoch_short_of_records(tmp_path):
    body = RINEX[len(HEADER) :].replace("0.0000000  0  1", "0.0000000  0  2", 1)
    (tmp_path / "t.obs").write_text(HEADER + body + "\n")  # a blank line is not a record
    [a] = read_rinex_observations(tmp_path / "t.obs", ("G1C",))
    assert a.tow == 0.0


def test_reader_ignores_a_scale_factor_of_a_system_it_does_not_read(tmp_path):
    glonass = "R   10  1 L1C".ljust(60) + "SYS / SCALE FACTOR\n"
    (tmp_path / "t.obs").write_text(RINEX.replace(END_OF_HEADER, glonass + END_OF_HEADER, 1))
    assert len(read_rinex_observations(tmp_path / "t.obs", ("G1C",))) == 2


def test_reader_survives_a_negative_epoch_count(tmp_path):
    body = RINEX[len(HEADER) :].replace("0.0000000  0  3", "0.0000000  0 -1", 1)
    (tmp_path / "t.obs").write_text(HEADER + body)
    assert [e.tow for e in read_rinex_observations(tmp_path / "t.obs", ("G1C",))] == [30.0]


def test_writer_declares_a_zero_phase_shift_per_phase_code(tmp_path):
    (tmp_path / "t.obs").write_text(RINEX)
    signals = ("G1C", "G2W", "E1C", "E5Q")
    epochs = read_rinex_observations(tmp_path / "t.obs", signals)
    write_rinex_observations(tmp_path / "w.obs", epochs)
    lines = (tmp_path / "w.obs").read_text().splitlines()
    shifts = [line[:5] for line in lines if line[60:].strip() == "SYS / PHASE SHIFT"]
    assert shifts == ["G L1C", "G L2W", "E L1C", "E L5Q"]
    assert lines.index("G L1C".ljust(60) + "SYS / PHASE SHIFT") < lines.index(END_OF_HEADER[:-1])
