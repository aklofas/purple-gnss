import importlib
import io

import matplotlib
import matplotlib.image
from matplotlib import font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyrx.acquisition import acquire_l1ca
from pyrx.synth import synth_l1ca


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
