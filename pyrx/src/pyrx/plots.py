"""Diagnostic plots written to image files."""

import os
from pathlib import Path

import numpy as np
from cycler import cycler
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyrx.acquisition import AcqResult
from pyrx.tracking import TrackResult

INK = "#243041"  # text and ticks
MUTED = "#637285"  # secondary text, spines, reference lines
BAR = "#CDD5DF"
BAR_DETECTED = "#0072B2"
LINES = ("#0072B2", "#D55E00", "#009E73")  # also east, north, up, with "-", "--", "-."
HIGHLIGHT = "#A64B00"  # exceptional events
GRID = "#DCE2E8"
# Okabe-Ito without its yellow, plus a dark wine: under simulated protanopia, deuteranopia
# and tritanopia every pair stays at least 16 CIELAB units apart. The orange and the light
# blue are too pale for text; identities still need a dash, marker or label as well.
CATEGORIES = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#882255",
    "#222222",
)
DASHES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1, 3, 1)),
    (0, (6, 2, 1, 2, 1, 2)),
)
# matplotlib rcParams of every figure, for `rc_context` or `rcParams.update`: opaque white,
# left-aligned bold titles (DejaVu Sans has no semibold), light grid, colour and dash cycled
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#FFFFFF",
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.prop_cycle": cycler(color=CATEGORIES) + cycler(linestyle=DASHES),
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.facecolor": "#FFFFFF",
    "figure.dpi": 100,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "savefig.facecolor": "#FFFFFF",
    "savefig.edgecolor": "#FFFFFF",
    "savefig.transparent": False,
}


def plot_acquisition(result: AcqResult, fs: float, path: str | os.PathLike) -> None:
    """Search grid (code offset x Doppler) and the Doppler cut through the peak."""
    n = result.grid.shape[1]
    offsets_ms = np.arange(n) / fs * 1e3
    fig = Figure(figsize=(8, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax_grid, ax_cut = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
    ax_grid.imshow(
        result.grid,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        extent=(offsets_ms[0], offsets_ms[-1], result.dopplers[0], result.dopplers[-1]),
        interpolation="nearest",
    )
    ax_grid.set_xlabel("code offset (ms)")
    ax_grid.set_ylabel("Doppler (Hz)")
    ax_grid.set_title(
        f"PRN {result.prn}: peak at {result.code_offset * 1e3:.5f} ms, "
        f"{result.doppler:.0f} Hz, C/N0 {result.cn0:.1f} dB-Hz"
    )
    col = round(result.code_offset * fs)
    ax_cut.plot(result.dopplers, result.grid[:, col], color=BAR_DETECTED, linewidth=2)
    ax_cut.set_xlabel("Doppler (Hz)")
    ax_cut.set_ylabel("normalised power")
    ax_cut.grid(color=MUTED, alpha=0.3)
    for ax in (ax_grid, ax_cut):
        ax.tick_params(colors=INK)
    fig.savefig(path, dpi=120)


def plot_cn0(results: list[AcqResult], threshold: float, path: str | os.PathLike) -> None:
    """C/N0 per PRN with the detection threshold; detected PRNs are darker and labelled."""
    with rc_context(PLOT_STYLE):
        prns = [r.prn for r in results]
        cn0 = [r.cn0 for r in results]
        colors = [BAR_DETECTED if r.detected else BAR for r in results]
        fig = Figure(figsize=(10, 4), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.bar(prns, cn0, color=colors, width=0.8)
        ax.axhline(threshold, color=MUTED, linestyle="--", linewidth=1)
        for r in results:
            if r.detected:
                ax.annotate(f"{r.cn0:.1f}", (r.prn, r.cn0), ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("PRN")
        ax.set_ylabel("C/N0 (dB-Hz)")
        ax.set_title(f"Acquisition C/N0 (threshold {threshold:.0f} dB-Hz)")
        ax.set_xticks(prns)
        ax.grid(axis="x", visible=False)
        fig.savefig(path, dpi=120)


def plot_tracking(result: TrackResult, path: str | os.PathLike) -> None:
    """Prompt I/Q (first 2 s), Doppler, C/N0 and |E|,|P|,|L| against time."""
    fig = Figure(figsize=(10, 10), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax_iq, ax_dop, ax_cn0, ax_epl = fig.subplots(4, 1, sharex=True)
    t = result.t
    head = t <= t[0] + 2.0
    ax_iq.plot(t[head], result.corr[head, 0].real, color=LINES[0], linewidth=1, label="I")
    ax_iq.plot(t[head], result.corr[head, 0].imag, color=LINES[1], linewidth=1, label="Q")
    ax_iq.set_ylabel("prompt (first 2 s)")
    ax_iq.legend(loc="upper right")
    ax_dop.plot(t, result.doppler, color=LINES[0], linewidth=1.5)
    ax_dop.set_ylabel("Doppler (Hz)")
    ax_cn0.plot(t, result.cn0, color=LINES[0], linewidth=1.5)
    ax_cn0.set_ylabel("C/N0 (dB-Hz)")
    mag = np.abs(result.corr)
    for col, name, color in ((1, "|E|", LINES[1]), (0, "|P|", LINES[0]), (2, "|L|", LINES[2])):
        ax_epl.plot(t, mag[:, col], color=color, linewidth=0.8, label=name)
    ax_epl.set_ylabel("correlator magnitude")
    ax_epl.set_xlabel("time (s)")
    ax_epl.legend(loc="upper right")
    state = "locked" if result.locked else f"lost at {result.lost_at:.1f} s"
    ax_iq.set_title(f"PRN {result.prn}: {state}, final C/N0 {result.cn0[-1]:.1f} dB-Hz")
    for ax in (ax_iq, ax_dop, ax_cn0, ax_epl):
        ax.grid(color=MUTED, alpha=0.3)
        ax.tick_params(colors=INK)
    fig.savefig(path, dpi=120)


def write_track_log(results: list[TrackResult], path: str | os.PathLike, interval: float = 0.1):
    """CSV of each channel's state every `interval` seconds (nearest epoch at or after)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("time_s,prn,cn0_dbhz,code_offset_ms,doppler_hz,carrier_cycles\n")
        for r in results:
            times = np.arange(0.0, r.t[-1] + 1e-9, interval)
            idx = np.searchsorted(r.t, times)
            f.writelines(
                f"{r.t[k]:.3f},{r.prn},{r.cn0[k]:.1f},{r.code_offset[k] * 1e3:.6f},"
                f"{r.doppler[k]:.1f},{r.carrier_phase[k]:.3f}\n"
                for k in idx[idx < len(r.t)]
            )
