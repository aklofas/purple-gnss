import importlib
import io

import matplotlib
import matplotlib.image
import numpy as np
from matplotlib import font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyrx.acquisition import acquire_l1ca
from pyrx.codes import CA_CHIP_RATE, CA_PERIOD
from pyrx.synth import synth_l1ca
from pyrx.tracking import TrackResult


def test_plots_leave_backend_alone(tmp_path):
    matplotlib.use("svg")
    import pyrx.plots

    importlib.reload(pyrx.plots)
    before = dict(matplotlib.rcParams)
    fs = 4e6
    r = acquire_l1ca(synth_l1ca(7, fs, 0.0, 0.0, 1000, int(fs * 0.005), 45.0, True), fs, 0.0, 7)
    pyrx.plots.plot_cn0([r], 38.0, tmp_path / "cn0.png")
    pyrx.plots.plot_acquisition(r, fs, tmp_path / "acq.png")
    assert matplotlib.get_backend().lower() == "svg"
    assert dict(matplotlib.rcParams) == before  # the style applies inside the plots only
    assert (tmp_path / "cn0.png").stat().st_size > 0


def test_plot_style_uses_weights_its_font_has():
    # matplotlib falls back with a logged warning, which notebooks keep in their outputs
    from pyrx.plots import PLOT_STYLE

    family = PLOT_STYLE["font.family"]
    weights = {f.weight for f in font_manager.fontManager.ttflist if f.name == family}
    for key in ("font.weight", "axes.titleweight", "figure.titleweight", "axes.labelweight"):
        weight = PLOT_STYLE.get(key, "normal")
        assert font_manager.weight_dict.get(weight, weight) in weights, key


def test_plot_style_is_opaque_white():
    from pyrx.plots import PLOT_STYLE

    with rc_context(PLOT_STYLE):
        fig = Figure()
        FigureCanvasAgg(fig)
        fig.subplots()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    buf.seek(0)
    assert matplotlib.image.imread(buf)[0, 0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_acquisition_figure_overview_keeps_every_code_offset():
    from pyrx.plots import acquisition_figure

    fs = 2.5e6
    x = synth_l1ca(7, fs, 0.0, -1020.0, 5, int(fs * 0.01), 39.0, True, seed=3)
    r = acquire_l1ca(x, fs, 0.0, 7)
    fig = acquisition_figure(r, fs)
    ax_all, ax_zoom = fig.axes[:2]
    (line,) = ax_all.get_lines()  # one value per code offset: the strongest Doppler bin, dB
    y = line.get_ydata()
    assert len(y) == r.grid.shape[1]
    assert np.argmax(y) == round(r.code_offset * fs) and y.max() == 0.0
    # the zoom wraps around the code period: a peak at sample 5 keeps both sides
    (mesh,) = ax_zoom.collections
    assert mesh.get_array().shape == (len(r.dopplers), 2 * int(np.ceil(3 * fs / CA_CHIP_RATE)) + 1)


def _long_track(seconds=30.0):
    n = round(seconds / CA_PERIOD)
    z = np.zeros(n)
    corr = np.ones((n, 4), dtype=np.complex64)
    nan = np.full(n, np.nan)
    t = np.arange(n) * CA_PERIOD
    return TrackResult(7, t, corr, z, z + 1000.0, z, z, z + 45.0, True, None, nan, nan)


def test_tracking_plot_gives_the_pull_in_its_own_time_axis(monkeypatch, tmp_path):
    from pyrx.plots import plot_tracking

    figs = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figs.append(self))
    plot_tracking(_long_track(), tmp_path / "trk.png")
    ax_iq, ax_dop, ax_cn0, ax_epl = figs[0].axes
    lo, hi = ax_iq.get_xlim()
    assert hi - lo <= 2.0 + 1e-9
    assert ax_epl.get_xlim()[1] > 29.0
    shared = ax_dop.get_shared_x_axes()
    assert shared.joined(ax_dop, ax_cn0) and shared.joined(ax_dop, ax_epl)
    assert not shared.joined(ax_dop, ax_iq)
