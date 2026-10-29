"""Diagnostic plots written to image files."""

import os
from pathlib import Path

import numpy as np
from cycler import cycler
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyrx.acquisition import AcqResult
from pyrx.codes import CA_CHIP_RATE
from pyrx.tracking import T_FLL_PULLIN, TrackResult

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


def acquisition_figure(result: AcqResult, fs: float) -> Figure:
    """The L1 C/A search: the strongest Doppler bin at every code offset, the samples
    within 3 chips of the peak, and the code and Doppler cuts through it."""
    with rc_context(PLOT_STYLE):
        grid = result.grid
        n = grid.shape[1]
        row, col = np.unravel_index(np.argmax(grid), grid.shape)
        dopplers = result.dopplers
        step = dopplers[1] - dopplers[0] if len(dopplers) > 1 else 500.0
        doppler_edges = np.r_[dopplers - step / 2, dopplers[-1] + step / 2] / 1e3
        samples_per_chip = fs / CA_CHIP_RATE
        radius = int(np.ceil(3 * samples_per_chip))
        offsets = np.arange(-radius, radius + 1)
        columns = (col + offsets) % n  # wraps around the code period
        chip_edges = np.r_[offsets - 0.5, offsets[-1] + 0.5] / samples_per_chip
        floor = 10 ** (-25 / 10)

        fig = Figure(figsize=(10, 7), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax_all, ax_zoom, ax_code, ax_dop = fig.subplots(2, 2).flat

        ax_all.plot(
            np.arange(n) / fs * 1e3,
            10 * np.log10(np.maximum(grid.max(axis=0), floor)),
            color=LINES[0],
            linewidth=0.6,
        )
        ax_all.set(ylim=(-25, 1), xlabel="code offset (ms)", ylabel="best Doppler bin (dB)")
        ax_all.set_title("Best Doppler bin per code offset")
        ax_all.text(
            0.02,
            0.97,
            f"sampled peak: {result.code_offset * 1e3:.4f} ms",
            transform=ax_all.transAxes,
            va="top",
            fontsize=10,
        )

        mesh = ax_zoom.pcolormesh(
            chip_edges,
            doppler_edges,
            10 * np.log10(np.maximum(grid[:, columns], floor)),
            cmap="cividis",
            vmin=-25,
            vmax=0,
            shading="flat",
        )
        ax_zoom.grid(False)
        ax_zoom.set(xlim=(-3, 3), xlabel="offset from sampled peak (chips)", ylabel="Doppler (kHz)")
        ax_zoom.set_title("Around the peak, ±3 chips")
        fig.colorbar(
            mesh, ax=ax_zoom, label="power relative to peak (dB)", ticks=[-25, -20, -10, 0]
        )

        ax_code.plot(offsets / samples_per_chip, grid[row, columns], ".-", color=LINES[0])
        ax_code.set(
            xlim=(-3, 3),
            ylim=(0, 1.05),
            xlabel="offset from sampled peak (chips)",
            ylabel="normalised power",
        )
        ax_code.set_title(f"Code cut at {dopplers[row]:+.0f} Hz")

        ax_dop.plot(dopplers, grid[:, col], "o-", color=LINES[0], markersize=4)
        ax_dop.axvline(result.doppler, color=HIGHLIGHT, linestyle="--")
        ax_dop.set(ylim=(0, 1.05), xlabel="Doppler (Hz)", ylabel="normalised power")
        ax_dop.set_title("Doppler cut: sampled bins")
        ax_dop.text(
            0.98,
            0.95,
            f"interpolated {result.doppler:+.0f} Hz",
            transform=ax_dop.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
        fig.suptitle(f"G{result.prn:02d}: C/N0 {result.cn0:.1f} dB-Hz")
        return fig


def plot_acquisition(result: AcqResult, fs: float, path: str | os.PathLike) -> None:
    """Write `acquisition_figure` to an image file."""
    with rc_context(PLOT_STYLE):
        acquisition_figure(result, fs).savefig(path, dpi=120)


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
    """Prompt I/Q over the first 2 s on its own time axis; Doppler change, C/N0 and
    |E|, |P|, |L| over the whole channel."""
    with rc_context(PLOT_STYLE):
        fig = Figure(figsize=(10, 9), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax_iq, ax_dop, ax_cn0, ax_epl = fig.subplots(4, 1)
        ax_cn0.sharex(ax_dop)
        ax_epl.sharex(ax_dop)
        t = result.t
        head = t <= t[0] + 2.0
        ax_iq.set_xlim(t[0], t[0] + max(min(2.0, t[-1] - t[0]), 0.001))
        ax_iq.plot(t[head], result.corr[head, 0].real, color=LINES[0], linewidth=0.8, label="I")
        ax_iq.plot(
            t[head],
            result.corr[head, 0].imag,
            color=LINES[1],
            linestyle="--",
            linewidth=0.8,
            label="Q",
        )
        handover = t[0] + T_FLL_PULLIN
        if handover <= min(t[-1], t[0] + 2.0):
            ax_iq.axvline(handover, color=MUTED, linestyle=":")
            ax_iq.text(
                handover,
                1.01,
                "FLL → PLL",
                transform=ax_iq.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                color=MUTED,
            )
        ax_iq.set_xlabel("recording time (s), first 2 s of the channel")
        ax_iq.set_ylabel("prompt sum")
        ax_iq.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)

        initial = result.doppler[0]
        ax_dop.plot(t, result.doppler - initial, color=LINES[0])
        ax_dop.axhline(0, color=MUTED, linewidth=0.8)
        ax_dop.set_ylabel("Doppler change (Hz)")
        ax_dop.set_title(f"Doppler change from the seed's {initial:+.1f} Hz", loc="right")
        ax_cn0.step(t, result.cn0, where="post", color=LINES[0])
        ax_cn0.set_ylabel("C/N0 (dB-Hz)")
        mag = np.abs(result.corr)
        for col, name, color, style in (
            (1, "|E|", LINES[1], "--"),
            (0, "|P|", LINES[0], "-"),
            (2, "|L|", LINES[2], "-."),
        ):
            ax_epl.plot(t, mag[:, col], color=color, linestyle=style, linewidth=0.8, label=name)
        ax_epl.set_ylabel("correlator magnitude")
        ax_epl.set_xlabel("recording time (s)")
        ax_epl.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
        for ax in (ax_dop, ax_cn0):
            ax.tick_params(labelbottom=False)
        state = "locked" if result.locked else f"lost at {result.lost_at:.1f} s"
        fig.suptitle(f"G{result.prn:02d}: {state}, final C/N0 {result.cn0[-1]:.1f} dB-Hz")
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
