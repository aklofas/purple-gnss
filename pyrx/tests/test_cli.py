import numpy as np

from pyrx.cli import acq_main, parse_prns
from pyrx.synth import synth_l1ca


def test_parse_prns():
    assert parse_prns("1-3,7") == [1, 2, 3, 7]


def test_acq_main_prints_table_and_plots(tmp_path, capsys):
    fs = 4e6
    x = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 0.01), 45.0, True)
    raw = np.empty(2 * len(x), dtype=np.int8)
    raw[0::2] = x.real.clip(-127, 127)
    raw[1::2] = (-x.imag).clip(-127, 127)
    f = tmp_path / "s.bin"
    raw.tofile(f)
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
    raw = np.empty(2 * len(x), dtype=np.int8)
    raw[0::2] = x.real.clip(-127, 127)
    raw[1::2] = (-x.imag).clip(-127, 127)
    f = tmp_path / "s.bin"
    raw.tofile(f)
    assert acq_main([str(f), "--fs", "4e6", "--iq", "--prn", "7"]) == 0
    assert "COFF=  0.25000 ms" in capsys.readouterr().out
