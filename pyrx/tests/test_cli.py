import argparse
import json
import re
import sys
from dataclasses import replace

import numpy as np
import pytest
from support import AMB_B, AMB_R, ROVER, SIGNALS, check_golden, without_date, without_elapsed

from pyrx import channels
from pyrx.acquisition import AcqResult
from pyrx.cli import (
    _solution_line,
    acq_main,
    nav_main,
    parse_prns,
    pvt_main,
    rinex_main,
    rtk_main,
    trk_main,
)
from pyrx.pvt import solve
from pyrx.rinex import (
    read_rinex_nav,
    read_rinex_obs,
    write_rinex_nav,
    write_rinex_obs,
    write_rinex_observations,
)
from pyrx.signalspec import L5
from pyrx.synth import (
    TOW,
    USUD,
    constellation,
    frame,
    synth_l1ca,
    synth_pilot,
    synthetic_pair,
    truth_epoch,
    write_iq_int8,
)


def test_parse_prns():
    assert parse_prns("1-3,7") == [1, 2, 3, 7]
    assert parse_prns("1,1-3,3") == [1, 2, 3]  # repeats dropped, order kept
    for bad in ("", "0", "33", "5-1", "1-", "a", "1,,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_prns(bad)


def test_prn_argument_errors_are_usage_errors(tmp_path, capsys):
    f = tmp_path / "s.bin"
    write_iq_int8(np.zeros(1000, dtype=complex), f)
    with pytest.raises(SystemExit) as e:
        acq_main([str(f), "--fs", "4e6", "--iq", "--prn", "1-33"])
    assert e.value.code == 2 and "--prn" in capsys.readouterr().err


def test_acq_main_prints_table_and_plots(tmp_path, capsys):
    fs = 4e6
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 0.02), 45.0, True)
    f = tmp_path / "s.bin"
    write_iq_int8(x, f)
    rc = acq_main([str(f), "--fs", "4e6", "--iq", "--prn", "7,9", "--plot", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SIG= L1CA, PRN=   7, COFF=  0.25000 ms" in out
    assert "PRN=   9" in out and out.strip().endswith("s")
    assert (tmp_path / "cn0.png").exists() and (tmp_path / "acq_prn07.png").exists()
    assert not (tmp_path / "acq_prn09.png").exists()
    check_golden("cli_acq.out", without_elapsed(out))


def test_acq_main_defaults_to_first_20ms(tmp_path, capsys):
    # Signal at offset 3000 samples for 8 ms, then at offset 1000 for 40 ms: a 10 ms
    # window would pick the first part, the 20 ms default sees 12 ms of the second.
    fs = 4e6
    n_ms = int(fs * 1e-3)
    first = synth_l1ca(7, fs, 0.0, 0.0, 3000, 8 * n_ms, 45.0, True, seed=1)
    rest = synth_l1ca(7, fs, 0.0, 0.0, 1000, 40 * n_ms, 45.0, True)
    x = np.concatenate([first, rest])
    f = tmp_path / "s.bin"
    write_iq_int8(x, f)
    assert acq_main([str(f), "--fs", "4e6", "--iq", "--prn", "7"]) == 0
    assert "COFF=  0.25000 ms" in capsys.readouterr().out


def test_trk_main_tracks_and_logs(tmp_path, capsys):
    fs = 4e6
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 2.5), 45.0, True, data_bits=True)
    f = tmp_path / "s.bin"
    write_iq_int8(x, f)
    log = tmp_path / "logs" / "trk.csv"  # directory does not exist yet
    rc = trk_main(
        [str(f), "--fs", "4e6", "--iq", "--prn", "7,9", "--log", str(log), "--plot", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "PRN  7" in out and "LOCK" in out and "PRN  9" not in out
    assert "PLI 0." in out
    assert out.strip().endswith("s")
    lines = log.read_text().splitlines()
    assert lines[0] == "time_s,prn,cn0_dbhz,phase_lock,code_offset_ms,doppler_hz,carrier_cycles"
    assert 24 <= len(lines) - 1 <= 26  # 2.5 s at 0.1 s
    assert (tmp_path / "trk_prn07.png").exists()
    check_golden("cli_trk.out", without_elapsed(out))
    check_golden("cli_trk.csv", log.read_text())


def test_nav_main_prints_and_writes_json(tmp_path, capsys):
    fs = 2.5e6
    seq = np.concatenate([np.zeros(100, dtype=np.uint8), frame()])
    n = int(fs * (len(seq) * 0.020 + 0.1))
    x = synth_l1ca(4, fs, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    f = tmp_path / "s.bin"
    write_iq_int8(x, f)
    out_json = tmp_path / "nav.json"
    rc = nav_main([str(f), "--fs", "2.5e6", "--iq", "--prn", "4", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PRN  4  bits sync  subframes 1@30258,2@30264,3@30270  ephemeris IODE 149" in out
    data = json.loads(out_json.read_text())
    assert data["ephemeris"]["4"]["iode"] == 149 and data["ionosphere"] is None
    check_golden("cli_nav.out", without_elapsed(out))
    check_golden("cli_nav.json", out_json.read_text())


def test_rinex_main_writes_obs_and_nav(tmp_path, capsys):
    fs = 2.5e6
    seq = np.concatenate([np.zeros(100, dtype=np.uint8), frame()])
    n = int(fs * (len(seq) * 0.020 + 0.1))
    x = synth_l1ca(4, fs, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    f = tmp_path / "s.bin"
    write_iq_int8(x, f)
    obs, nav = tmp_path / "o.obs", tmp_path / "o.nav"
    rc = rinex_main(
        [str(f), "--fs", "2.5e6", "--iq", "--prn", "4", "--obs", str(obs), "--nav", str(nav)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    n_epochs = int(re.search(r"PRN  4  epochs (\d+)", out).group(1))
    assert 15 <= n_epochs <= 19 and f"EPOCHS {n_epochs}" in out
    # week 2048 + 142 = 2190, receiver time about 30256 + 0.068 + T_e: 2021-12-26 08:24:1x..3x
    assert obs.read_text().count("> 2021 12 26 08 24 ") == n_epochs
    assert nav.read_text().count("\nG04 ") == 1
    check_golden("cli_rinex.out", without_elapsed(out))
    check_golden("cli_rinex.obs", without_date(obs.read_text()))
    check_golden("cli_rinex.nav", without_date(nav.read_text()))
    assert rinex_main([str(f), "--fs", "2.5e6", "--iq", "--prn", "9", "--obs", str(obs)]) == 2


def test_pvt_main_solves_a_synthetic_file(tmp_path, capsys):
    ephs = constellation()
    epochs = [truth_epoch(ephs, USUD, 1e-3, tow=28800.0 + k) for k in range(3)]
    obs, nav, out = tmp_path / "s.obs", tmp_path / "s.nav", tmp_path / "s.pos"
    write_rinex_obs(obs, epochs)
    write_rinex_nav(nav, ephs, {e.prn: 28800.0 for e in ephs})
    assert pvt_main([str(obs), str(nav), "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "EPOCHS 3 of 3  mean lat 36.1331" in text and "clock 1.000 ms" in text
    lines = out.read_text().splitlines()
    assert lines[0].startswith("% week tow lat lon height clock_s ns gdop vx vy vz ddtr")
    assert len(lines) == 4 and lines[1].split()[:2] == ["2190", "28800.000"]
    check_golden("cli_pvt.out", without_elapsed(text))
    check_golden("cli_pvt.pos", out.read_text())
    assert pvt_main([str(obs), str(nav), "-o", str(out), "--weights", "rtklib"]) == 0
    check_golden("cli_pvt_rtklib.pos", out.read_text())
    capsys.readouterr()
    assert pvt_main([str(obs), str(nav), "--ts", "2021-12-26 09:00:00"]) == 2


def test_rtk_main_fixes_a_synthetic_pair(tmp_path, capsys):
    rng = np.random.default_rng(20)
    ephs = constellation()
    pairs = [
        synthetic_pair(ephs, USUD, ROVER, TOW + 30.0 * k, SIGNALS, AMB_R, AMB_B, rng)
        for k in range(4)
    ]
    rov, base, nav, out = (tmp_path / n for n in ("r.obs", "b.obs", "n.nav", "fix.txt"))
    write_rinex_observations(rov, [p[0] for p in pairs], "ROVR")
    write_rinex_observations(base, [p[1] for p in pairs], "BASE")
    write_rinex_nav(nav, ephs, {e.prn: TOW for e in ephs})
    args = [str(rov), str(base), str(nav), "--base-xyz", *map(str, USUD), "--signals", "G1C,G2W"]
    assert rtk_main([*args, "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "EPOCHS 3 fixed 1 float of 4" in text or "EPOCHS 4 fixed 0 float of 4" in text
    assert "baseline E" in text
    lines = out.read_text().splitlines()
    assert lines[0].startswith("% week tow x y z Q ns ratio age") and len(lines) == 5
    assert lines[-1].split()[5] == "1" and abs(float(lines[-1].split()[2]) - ROVER[0]) < 0.01
    check_golden("cli_rtk.out", without_elapsed(text))
    check_golden("cli_rtk.pos", out.read_text())
    assert rtk_main([*args, "--ts", "2021-12-26 09:00:00"]) == 2


def test_mains_report_no_detection_with_exit_2(tmp_path, capsys):
    fs = 4e6
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 0.02), 45.0, True)
    f = tmp_path / "s.bin"
    write_iq_int8(x, f)
    common = [str(f), "--fs", "4e6", "--iq", "--prn", "9"]  # PRN 9 is not in the file
    assert trk_main(common) == 2
    assert nav_main(common) == 2
    assert rinex_main([*common, "--obs", str(tmp_path / "o.obs")]) == 2
    err = capsys.readouterr().err
    assert err.count("no PRN detected") == 3 and "--f-if" in err
    assert not (tmp_path / "o.obs").exists()
    check_golden("cli_no_detection.err", err)


def test_mains_say_why_nothing_was_tracked(tmp_path, capsys, monkeypatch):
    fs = 4e6
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 0.02), 45.0, True)
    f = tmp_path / "s.bin"
    write_iq_int8(x, f)
    common = [str(f), "--fs", "4e6", "--iq", "--prn", "9"]
    # no 20 ms window fits between --t-start and the end of a 20 ms file
    assert trk_main([*common, "--t-start", "0.01"]) == 2
    err = capsys.readouterr().err
    assert "no 20 ms acquisition window fits between --t-start 0.01 s and the end" in err
    # a detection on every probe that the bit-sync trial turns down
    monkeypatch.setattr(
        channels, "acquire_l1ca", lambda samples, fs, f_if, prn, **kw: _always_detected(prn)
    )
    assert trk_main(common) == 2
    err = capsys.readouterr().err
    assert "every detection failed the 6 s bit-sync trial (PRN 9 at 0 s)" in err
    assert "no PRN detected" not in err


def _always_detected(prn):
    return AcqResult(prn, 0.0005, 0.0, 36.0, True, np.zeros((1, 1), np.float32), np.zeros(1))


def test_pvt_main_weights_by_cn0_unless_asked_for_rtklibs(tmp_path, capsys):
    # one satellite 20 m off at 30 dB-Hz: the default pulls the fix less toward it than
    # RTKLIB's weights, which are what the library's solver uses unless told otherwise
    ephs = constellation()
    ep = truth_epoch(ephs, USUD, 1e-3, tow=28800.0)
    bad = replace(ep.obs[2], pseudorange=ep.obs[2].pseudorange + 20.0, cn0=30.0)
    ep = replace(ep, obs=ep.obs[:2] + [bad] + ep.obs[3:])
    obs, nav = tmp_path / "s.obs", tmp_path / "s.nav"
    write_rinex_obs(obs, [ep])
    write_rinex_nav(nav, ephs, {e.prn: 28800.0 for e in ephs})
    lines = {}
    for mode in ("default", "cn0", "rtklib"):
        out = tmp_path / f"{mode}.pos"
        flags = [] if mode == "default" else ["--weights", mode]
        assert pvt_main([str(obs), str(nav), "-o", str(out), *flags]) == 0
        lines[mode] = out.read_text().splitlines()[1]
    capsys.readouterr()
    assert lines["default"] == lines["cn0"] != lines["rtklib"]
    epochs, (eph_read, iono) = read_rinex_obs(obs), read_rinex_nav(nav)
    assert lines["rtklib"] == _solution_line(solve(epochs, eph_read, iono)[0])
    assert lines["cn0"] == _solution_line(solve(epochs, eph_read, iono, weights="cn0")[0])


def test_pvt_and_rtk_report_a_rejected_file_with_exit_2(tmp_path, capsys):
    ephs = constellation()
    obs, nav = tmp_path / "s.obs", tmp_path / "s.nav"
    write_rinex_obs(obs, [truth_epoch(ephs, USUD, 1e-3, tow=28800.0)])
    write_rinex_nav(nav, ephs, {e.prn: 28800.0 for e in ephs})
    obs.write_text(obs.read_text().replace("     3.05", "     2.11", 1))
    assert pvt_main([str(obs), str(nav)]) == 2
    assert rtk_main([str(obs), str(obs), str(nav), "--base-xyz", *map(str, USUD)]) == 2
    err = capsys.readouterr().err
    assert err.count("RINEX 3") == 2 and "Traceback" not in err
    check_golden("cli_rejected.err", err)


def test_rtk_reports_a_malformed_header_with_exit_2(tmp_path, capsys):
    ephs = constellation()
    obs, nav = tmp_path / "s.obs", tmp_path / "s.nav"
    write_rinex_obs(obs, [truth_epoch(ephs, USUD, 1e-3, tow=28800.0)])
    write_rinex_nav(nav, ephs, {e.prn: 28800.0 for e in ephs})
    text = obs.read_text()
    assert "        0.0000        0.0000        0.0000                  ANTENNA: DELTA H/E/N" in text
    obs.write_text(text.replace("        0.0000        0.0000        0.0000                  ANTENNA", "        0.0000        x.xxxx        0.0000                  ANTENNA", 1))
    assert rtk_main([str(obs), str(obs), str(nav), "--base-xyz", *map(str, USUD)]) == 2
    err = capsys.readouterr().err
    assert "malformed ANTENNA: DELTA H/E/N header record" in err and "Traceback" not in err


@pytest.mark.parametrize("main", [acq_main, trk_main, nav_main, rinex_main, pvt_main, rtk_main])
def test_help_golden(main, capsys, monkeypatch):
    if sys.version_info[:2] != (3, 12):
        pytest.skip("the help goldens pin the argparse layout of Python 3.12")
    monkeypatch.setenv("COLUMNS", "80")
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    check_golden(f"help_{main.__name__.removesuffix('_main')}.txt", capsys.readouterr().out)


def test_trk_main_prints_a_line_per_segment(gap_file, tmp_path, capsys):
    log = tmp_path / "trk.csv"
    rc = trk_main(
        [
            str(gap_file),
            "--fs",
            "2.5e6",
            "--iq",
            "--prn",
            "4",
            "--plot",
            str(tmp_path),
            "--log",
            str(log),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    rows = log.read_text().splitlines()[1:]
    times = [float(r.split(",")[0]) for r in rows]
    assert len(set(times)) == len(times)  # one row per 0.1 s per segment, none repeated
    assert sum(t >= 30.0 for t in times) <= 100 and sum(t < 30.0 for t in times) <= 170
    lines = [line for line in out.splitlines() if line.startswith("PRN  4")]
    assert len(lines) == 2
    assert lines[0].startswith("PRN  4  C/N0") and "LOST at" in lines[0]
    assert re.match(r"PRN  4 from 30\.0 s  C/N0", lines[1]) and lines[1].endswith("LOCK")
    assert (tmp_path / "trk_prn04.png").exists() and (tmp_path / "trk_prn04_2.png").exists()


def test_rinex_main_writes_one_nav_record_from_two_segments(gap_file, tmp_path, capsys):
    obs, nav = tmp_path / "g.obs", tmp_path / "g.nav"
    rc = rinex_main(
        [str(gap_file), "--fs", "2.5e6", "--iq", "--prn", "4", "--obs", str(obs), "--nav", str(nav)]
        + ["--marker-type", "GROUND_CRAFT"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    lines = [line for line in out.splitlines() if line.startswith("PRN  4")]
    assert len(lines) == 2 and lines[1].startswith("PRN  4 from 30.0 s  epochs")
    # each line counts its own segment's observations: together they are the file's G04
    # records (16 and 8 of 24 measured; one count for both would double it)
    n1, n2 = (int(re.search(r"epochs (\d+)", line).group(1)) for line in lines)
    assert n1 > 0 and n2 > 0 and n1 + n2 == obs.read_text().count("\nG04 ")
    assert nav.read_text().count("\nG04 ") == 1
    assert "GROUND_CRAFT".ljust(60) + "MARKER TYPE" in obs.read_text()


PILOT_FS = 6e6


def _l5_file(tmp_path, seconds, prn=6):
    x = synth_pilot(L5, prn, PILOT_FS, 0.0, 1234.0, 1500, int(PILOT_FS * seconds), 45.0)
    f = tmp_path / "l5.bin"
    write_iq_int8(x, f)
    return f


def test_acq_main_takes_the_signal(tmp_path, capsys):
    f = _l5_file(tmp_path, 0.02)
    assert acq_main([str(f), "--fs", "6e6", "--iq", "--signal", "L5", "--prn", "6,11"]) == 0
    out = capsys.readouterr().out
    assert "SIG= L5  , PRN=   6, COFF=  0.25000 ms" in out and "PRN=  11" in out
    check_golden("cli_acq_l5.out", without_elapsed(out))


def test_acq_main_defaults_to_the_signals_prns(tmp_path, capsys):
    f = _l5_file(tmp_path, 0.02)
    assert acq_main([str(f), "--fs", "6e6", "--iq", "--signal", "E5A"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("SIG= E5A ")]
    assert len(lines) == 36 and "PRN=  36" in lines[-1]


def test_trk_main_tracks_a_pilot(tmp_path, capsys):
    f = _l5_file(tmp_path, 2.6)
    log = tmp_path / "trk.csv"
    rc = trk_main([str(f), "--fs", "6e6", "--iq", "--signal", "L5", "--prn", "6", "--log", str(log)])
    out = capsys.readouterr().out
    assert rc == 0 and "PRN  6" in out and "SYNC 1.5 s" in out and "LOCK" in out
    rows = log.read_text().splitlines()
    assert rows[0].endswith(",synced") and rows[1].endswith(",0") and rows[-1].endswith(",1")
    check_golden("cli_trk_l5.out", without_elapsed(out))
    check_golden("cli_trk_l5.csv", log.read_text())


@pytest.mark.parametrize("signal,prn,span", [("E5A", "37", "1-36"), ("L5", "40", "1-32"), ("L5", "0", "1-32")])
def test_prn_outside_the_signals_set_is_a_usage_error(tmp_path, capsys, signal, prn, span):
    f = tmp_path / "s.bin"
    f.write_bytes(b"\0" * 16)
    for main in (acq_main, trk_main):
        with pytest.raises(SystemExit) as e:
            main([str(f), "--fs", "6e6", "--iq", "--signal", signal, "--prn", prn])
        err = capsys.readouterr().err
        assert e.value.code == 2 and "--prn" in err and f"PRNs {span}" in err


def test_pilot_trial_failure_is_named(tmp_path, capsys, monkeypatch):
    f = _l5_file(tmp_path, 3.2)
    monkeypatch.setattr(channels, "acquire_signal", lambda *a, **k: _always_detected(11))
    assert trk_main([str(f), "--fs", "6e6", "--iq", "--signal", "L5", "--prn", "11"]) == 2
    err = capsys.readouterr().err
    assert "every detection failed the 6 s secondary-code trial (PRN 11 at 0 s)" in err
