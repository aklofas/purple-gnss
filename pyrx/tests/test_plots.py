import importlib

import matplotlib

from pyrx.acquisition import acquire_l1ca
from pyrx.synth import synth_l1ca


def test_plots_leave_backend_alone(tmp_path):
    matplotlib.use("svg")
    import pyrx.plots

    importlib.reload(pyrx.plots)
    fs = 4e6
    r = acquire_l1ca(synth_l1ca(7, fs, 0.0, 0.0, 1000, int(fs * 0.005), 45.0, True), fs, 0.0, 7)
    pyrx.plots.plot_cn0([r], 38.0, tmp_path / "cn0.png")
    pyrx.plots.plot_acquisition(r, fs, tmp_path / "acq.png")
    assert matplotlib.get_backend().lower() == "svg"
    assert (tmp_path / "cn0.png").stat().st_size > 0
