import json
import re

import numpy as np
from test_navdecode import _frame

from pyrx.cli import acq_main, nav_main, parse_prns, rinex_main, trk_main
from pyrx.synth import synth_l1ca


def _write_iq_int8(x, path):
    """Write complex samples as Pocket SDR interleaved int8 I/Q (Q negated)."""
    raw = np.empty(2 * len(x), dtype=np.int8)
    raw[0::2] = x.real.clip(-127, 127)
    raw[1::2] = (-x.imag).clip(-127, 127)
    raw.tofile(path)


def test_parse_prns():
    assert parse_prns("1-3,7") == [1, 2, 3, 7]


def test_acq_main_prints_table_and_plots(tmp_path, capsys):
    fs = 4e6
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 0.01), 45.0, True)
    f = tmp_path / "s.bin"
    _write_iq_int8(x, f)
    rc = acq_main([str(f), "--fs", "4e6", "--iq", "--prn", "7,9", "--plot", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SIG= L1CA, PRN=   7, COFF=  0.25000 ms" in out
    assert "PRN=   9" in out and out.strip().endswith("s")
    assert (tmp_path / "cn0.png").exists() and (tmp_path / "acq_prn07.png").exists()
    assert not (tmp_path / "acq_prn09.png").exists()


def test_acq_main_defaults_to_first_10ms(tmp_path, capsys):
    # Signal at offset 1000 samples for 10 ms, then at offset 3000 for 20 ms: the
    # default window must see only the first part.
    fs = 4e6
    n10 = int(fs * 0.01)
    first = synth_l1ca(7, fs, 0.0, 0.0, 1000, n10, 45.0, True)
    rest = synth_l1ca(7, fs, 0.0, 0.0, 3000, 2 * n10, 45.0, True, seed=1)
    x = np.concatenate([first, rest])
    f = tmp_path / "s.bin"
    _write_iq_int8(x, f)
    assert acq_main([str(f), "--fs", "4e6", "--iq", "--prn", "7"]) == 0
    assert "COFF=  0.25000 ms" in capsys.readouterr().out


def test_trk_main_tracks_and_logs(tmp_path, capsys):
    fs = 4e6
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 2.5), 45.0, True, data_bits=True)
    f = tmp_path / "s.bin"
    _write_iq_int8(x, f)
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


def test_nav_main_prints_and_writes_json(tmp_path, capsys):
    fs = 2.5e6
    seq = np.concatenate([np.zeros(100, dtype=np.uint8), _frame()])
    n = int(fs * (len(seq) * 0.020 + 0.1))
    x = synth_l1ca(4, fs, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    f = tmp_path / "s.bin"
    _write_iq_int8(x, f)
    out_json = tmp_path / "nav.json"
    rc = nav_main([str(f), "--fs", "2.5e6", "--iq", "--prn", "4", "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PRN  4  bits sync  subframes 1@30258,2@30264,3@30270  ephemeris IODE 149" in out
    data = json.loads(out_json.read_text())
    assert data["ephemeris"]["4"]["iode"] == 149 and data["ionosphere"] is None


def test_rinex_main_writes_obs_and_nav(tmp_path, capsys):
    fs = 2.5e6
    seq = np.concatenate([np.zeros(100, dtype=np.uint8), _frame()])
    n = int(fs * (len(seq) * 0.020 + 0.1))
    x = synth_l1ca(4, fs, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    f = tmp_path / "s.bin"
    _write_iq_int8(x, f)
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
    assert rinex_main([str(f), "--fs", "2.5e6", "--iq", "--prn", "9", "--obs", str(obs)]) == 2
