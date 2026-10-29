import argparse
import json
import re
import sys

import numpy as np
import pytest
from support import AMB_B, AMB_R, ROVER, SIGNALS, check_golden, without_date, without_elapsed

from pyrx.cli import acq_main, nav_main, parse_prns, pvt_main, rinex_main, rtk_main, trk_main
from pyrx.rinex import write_rinex_nav, write_rinex_obs, write_rinex_observations
from pyrx.synth import (
    TOW,
    USUD,
    constellation,
    frame,
    synth_l1ca,
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
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 0.01), 45.0, True)
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


def test_acq_main_defaults_to_first_10ms(tmp_path, capsys):
    # Signal at offset 1000 samples for 10 ms, then at offset 3000 for 20 ms: the
    # default window must see only the first part.
    fs = 4e6
    n10 = int(fs * 0.01)
    first = synth_l1ca(7, fs, 0.0, 0.0, 1000, n10, 45.0, True)
    rest = synth_l1ca(7, fs, 0.0, 0.0, 3000, 2 * n10, 45.0, True, seed=1)
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
    assert out.strip().endswith("s")
    lines = log.read_text().splitlines()
    assert lines[0] == "time_s,prn,cn0_dbhz,code_offset_ms,doppler_hz,carrier_cycles"
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


@pytest.mark.parametrize("main", [acq_main, trk_main, nav_main, rinex_main, pvt_main, rtk_main])
def test_help_golden(main, capsys, monkeypatch):
    if sys.version_info[:2] != (3, 12):
        pytest.skip("the help goldens pin the argparse layout of Python 3.12")
    monkeypatch.setenv("COLUMNS", "80")
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    check_golden(f"help_{main.__name__.removesuffix('_main')}.txt", capsys.readouterr().out)
