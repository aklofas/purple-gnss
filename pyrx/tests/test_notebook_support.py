"""The notebooks' support module: data-source detection and the synthetic inputs."""

import gzip
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyrx.pvt import solve
from pyrx.rtk import solve_rtk

# loaded by path: the tests have their own `support` module
_SPEC = importlib.util.spec_from_file_location(
    "nbsupport", Path(__file__).resolve().parents[1] / "notebooks" / "support.py"
)
nbsupport = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nbsupport)


def test_data_source_is_none_without_complete_captures(monkeypatch, tmp_path):
    monkeypatch.delenv("PYRX_CAPTURES", raising=False)
    assert nbsupport.data_source() is None
    monkeypatch.setenv("PYRX_CAPTURES", str(tmp_path))  # a directory without the files
    assert nbsupport.data_source() is None


def _capture_tree(root, skip=()):
    for name, rel in nbsupport._FILES.items():
        if name not in skip:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).touch()  # data_source checks only that the files exist


def test_data_source_prints_no_path(monkeypatch, tmp_path, capsys):
    _capture_tree(tmp_path, skip=("recording",))
    monkeypatch.setenv("PYRX_CAPTURES", str(tmp_path))
    assert nbsupport.data_source() is None
    out = capsys.readouterr().out
    assert nbsupport._FILES["recording"] in out and str(tmp_path) not in out


def test_data_source_names_every_file_of_a_complete_tree(monkeypatch, tmp_path, capsys):
    _capture_tree(tmp_path)
    monkeypatch.setenv("PYRX_CAPTURES", str(tmp_path))
    src = nbsupport.data_source()
    assert src.root == tmp_path and (src.fs, src.f_if, src.iq) == (24e6, 6e6, False)
    for name, rel in nbsupport._FILES.items():
        assert getattr(src, name) == tmp_path / rel
    assert str(tmp_path) not in capsys.readouterr().out


def test_synthetic_pvt_epochs_solve_to_the_truth():
    epochs, ephs, iono, truth = nbsupport.synthetic_pvt_epochs(n=6)
    sols = solve(epochs, ephs, iono, 15.0)
    assert len(sols) == 6
    assert all(np.linalg.norm(s.position - truth) < 1.5 for s in sols)
    assert sols[-1].clock_bias > sols[0].clock_bias  # the clock drifts


def test_synthetic_rtk_epochs_fix_and_carry_the_slip():
    rover, base, ephs, station, delta, truth = nbsupport.synthetic_rtk_epochs(n=8, slip_at=4)
    fixes = {(f.week, f.tow): f for f in solve_rtk(rover, base, ephs, station, delta)}
    before, after = (rover[3].week, rover[3].tow), (rover[-1].week, rover[-1].tow)
    assert fixes[before].status == "fix" and fixes[after].status == "fix"
    for f in fixes.values():
        if f.status == "fix":
            assert np.linalg.norm(f.position - truth) < 0.02
    i, j = rover[0].sats.index("G03"), rover[0].signals.index("G1C")
    r, jj = rover[0].sats.index("G01"), j
    before = (rover[3].carrier_phase[i, j] - base[3].carrier_phase[i, j]) - (
        rover[3].carrier_phase[r, jj] - base[3].carrier_phase[r, jj]
    )
    after = (rover[4].carrier_phase[i, j] - base[4].carrier_phase[i, j]) - (
        rover[4].carrier_phase[r, jj] - base[4].carrier_phase[r, jj]
    )
    assert abs((after - before) - 3.0) < 0.5  # the double difference jumps by the slip


def test_synthetic_recording_is_cached_and_sized(tmp_path):
    path, fs, f_if, iq, prns = nbsupport.synthetic_recording(tmp_path, seconds=0.2)
    assert iq and f_if == 0.0 and prns == [4, 7, 12, 25]
    assert path.stat().st_size == 2 * int(fs * 0.2)
    mtime = path.stat().st_mtime
    nbsupport.synthetic_recording(tmp_path, seconds=0.2)
    assert path.stat().st_mtime == mtime


def test_reference_runners_return_none_without_rtklib(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert nbsupport.rnx2rtkp_single("a.obs", "a.nav", tmp_path, nbsupport.PVT_SPAN) is None
    assert (
        nbsupport.rnx2rtkp_rtk(
            "r.obs", "b.obs", "a.nav", tmp_path, np.zeros(3), 0.0, 0.0, nbsupport.RTK_SPAN
        )
        is None
    )


# a stand-in for rnx2rtkp: "none" writes nothing and exits 0 (RTKLIB with no data), "header"
# writes a header only, "echo" writes one solution row whose position is the first input
# file's contents
_FAKE_RNX2RTKP = """\
import sys
from types import SimpleNamespace
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("-o") + 1])
if MODE == "header":
    out.write_text("% header\\n")
elif MODE == "echo":
    xyz = Path(args[args.index("-o") + 2]).read_text().split()
    out.write_text("% header\\n2190 0.000 " + " ".join(xyz) + " 1 8 0 0 0 0 0 0 0.0 3.5\\n")
"""


def _fake_rnx2rtkp(monkeypatch, tmp_path, mode):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "rnx2rtkp"
    exe.write_text(f"#!{sys.executable}\nMODE = {mode!r}\n" + _FAKE_RNX2RTKP)
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))


def test_reference_runners_never_return_a_previous_solution(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for mode in ("none", "header"):
        _fake_rnx2rtkp(monkeypatch, tmp_path, mode)
        (out / "rnx2rtkp.pos").write_text("2190 0.000 1 2 3 1 8 0 0 0 0 0 0 0.0 3.5\n")
        assert nbsupport.rnx2rtkp_single("a.obs", "a.nav", out, nbsupport.PVT_SPAN) is None
        (out / "rnx2rtkp.pos").write_text("2190 0.000 1 2 3 1 8 0 0 0 0 0 0 0.0 3.5\n")
        assert (
            nbsupport.rnx2rtkp_rtk(
                "r.obs", "b.obs", "a.nav", out, np.zeros(3), 0.0, 0.0, nbsupport.RTK_SPAN
            )
            is None
        )


def test_reference_runner_reads_the_current_source(monkeypatch, tmp_path):
    _fake_rnx2rtkp(monkeypatch, tmp_path, "echo")
    out = tmp_path / "out"
    for name, xyz in (("one", "1 2 3"), ("two", "4 5 6")):  # same file name, other contents
        src = tmp_path / name / "USUD00JPN_R_20213600000_01D_30S_MO.rnx.gz"
        src.parent.mkdir()
        src.write_bytes(gzip.compress(xyz.ljust(100).encode()))  # hatanaka rejects tiny files
        ref = nbsupport.rnx2rtkp_single(src, "a.nav", out, nbsupport.PVT_SPAN)
        assert ref[(2190, 0.0)].tolist() == [float(v) for v in xyz.split()]


def test_synthetic_recording_follows_its_recipe(monkeypatch, tmp_path):
    path, *_ = nbsupport.synthetic_recording(tmp_path, seconds=0.05)
    data = path.read_bytes()
    monkeypatch.setattr(nbsupport, "SYNTH_CN0", 30.0)
    path2, *_ = nbsupport.synthetic_recording(tmp_path, seconds=0.05)
    assert path2.read_bytes() != data
    monkeypatch.setattr(nbsupport, "SYNTH_PRNS", [4, 7])
    path3, _, _, _, prns = nbsupport.synthetic_recording(tmp_path, seconds=0.05)
    assert prns == [4, 7] and path3.read_bytes() not in (data, path2.read_bytes())


def test_reference_runners_read_the_solution_columns(monkeypatch, tmp_path):
    _fake_rnx2rtkp(monkeypatch, tmp_path, "echo")
    rover = tmp_path / "r.obs"
    rover.write_text("4641951.5 1393054.25 4133281.125")
    ref = nbsupport.rnx2rtkp_rtk(
        rover, "b.obs", "a.nav", tmp_path / "out", np.zeros(3), 0.0, 0.0, nbsupport.RTK_SPAN
    )
    ((key, row),) = ref.items()
    assert key == (2190, 0.0) and (row["q"], row["ns"], row["ratio"]) == (1, 8, 3.5)
    assert row["xyz"].tolist() == [4641951.5, 1393054.25, 4133281.125]


def test_elapsed_minutes_run_across_the_week_boundary():
    start = SimpleNamespace(week=2189, tow=604740.0)  # one minute before the week ends
    later = [start, SimpleNamespace(week=2190, tow=0.0), SimpleNamespace(week=2190, tow=90.0)]
    assert nbsupport.elapsed_minutes(later, start).tolist() == [0.0, 1.0, 2.5]
    assert nbsupport.time_note(SimpleNamespace(week=2190, tow=0.0), synthetic=False) == (
        "time: minutes from 2021-12-26 00:00:00 GPST"
    )
    assert "synthetic" in nbsupport.time_note(start, synthetic=True)


def test_figure_applies_the_package_style():
    nbsupport.plt.switch_backend("agg")  # no display in the test run
    fig, ax = nbsupport.figure()
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.title.get_fontweight() == "bold" and ax.xaxis.get_gridlines()[0].get_visible()
    nbsupport.plt.close(fig)


def test_diagram_arrows_meet_the_box_sides_and_control_paths_are_dashed():
    nbsupport.plt.switch_backend("agg")
    fig, ax = nbsupport.diagram_canvas(16, 8, "test")
    a = nbsupport.diagram_box(ax, 1, 2, 4, 2, "A", code="acquire_l1ca")
    b = nbsupport.diagram_box(ax, 9, 2, 4, 2, "B", role="measure")
    assert (a.right, b.left, a.top) == ((5, 3), (9, 3), (3, 4))
    nbsupport.diagram_arrow(
        ax, [b.top, (11, 6), (3, 6), a.top], "feedback", (7, 6.3), role="control"
    )
    line, head = ax.patches[-2:]
    assert line.get_linestyle() == "--" and head.get_linestyle() == "solid"
    nbsupport.plt.close(fig)
