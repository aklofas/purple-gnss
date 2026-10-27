"""Diagnostic plots written to image files."""

import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyrx.acquisition import AcqResult
from pyrx.tracking import TrackResult

_INK = "#333333"
_MUTED = "#9aa5b1"
_BAR = "#9ecae1"
_BAR_DETECTED = "#2171b5"
_LINES = ("#2171b5", "#6baed6", "#08306b")


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
    ax_cut.plot(result.dopplers, result.grid[:, col], color=_BAR_DETECTED, linewidth=2)
    ax_cut.set_xlabel("Doppler (Hz)")
    ax_cut.set_ylabel("normalised power")
    ax_cut.grid(color=_MUTED, alpha=0.3)
    for ax in (ax_grid, ax_cut):
        ax.tick_params(colors=_INK)
    fig.savefig(path, dpi=120)


def plot_cn0(results: list[AcqResult], threshold: float, path: str | os.PathLike) -> None:
    """C/N0 per PRN with the detection threshold; detected PRNs are darker and labelled."""
    prns = [r.prn for r in results]
    cn0 = [r.cn0 for r in results]
    colors = [_BAR_DETECTED if r.detected else _BAR for r in results]
    fig = Figure(figsize=(10, 4), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.bar(prns, cn0, color=colors, width=0.8)
    ax.axhline(threshold, color=_MUTED, linestyle="--", linewidth=1)
    for r in results:
        if r.detected:
            ax.annotate(f"{r.cn0:.1f}", (r.prn, r.cn0), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("PRN")
    ax.set_ylabel("C/N0 (dB-Hz)")
    ax.set_title(f"Acquisition C/N0 (threshold {threshold:.0f} dB-Hz)")
    ax.set_xticks(prns)
    ax.grid(axis="y", color=_MUTED, alpha=0.3)
    fig.savefig(path, dpi=120)


def plot_tracking(result: TrackResult, path: str | os.PathLike) -> None:
    """Prompt I/Q (first 2 s), Doppler, C/N0 and |E|,|P|,|L| against time."""
    fig = Figure(figsize=(10, 10), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax_iq, ax_dop, ax_cn0, ax_epl = fig.subplots(4, 1, sharex=True)
    t = result.t
    head = t <= t[0] + 2.0
    ax_iq.plot(t[head], result.corr[head, 0].real, color=_LINES[0], linewidth=1, label="I")
    ax_iq.plot(t[head], result.corr[head, 0].imag, color=_LINES[1], linewidth=1, label="Q")
    ax_iq.set_ylabel("prompt (first 2 s)")
    ax_iq.legend(loc="upper right")
    ax_dop.plot(t, result.doppler, color=_LINES[0], linewidth=1.5)
    ax_dop.set_ylabel("Doppler (Hz)")
    ax_cn0.plot(t, result.cn0, color=_LINES[0], linewidth=1.5)
    ax_cn0.set_ylabel("C/N0 (dB-Hz)")
    mag = np.abs(result.corr)
    for col, name, color in ((1, "|E|", _LINES[1]), (0, "|P|", _LINES[0]), (2, "|L|", _LINES[2])):
        ax_epl.plot(t, mag[:, col], color=color, linewidth=0.8, label=name)
    ax_epl.set_ylabel("correlator magnitude")
    ax_epl.set_xlabel("time (s)")
    ax_epl.legend(loc="upper right")
    state = "locked" if result.locked else f"lost at {result.lost_at:.1f} s"
    ax_iq.set_title(f"PRN {result.prn}: {state}, final C/N0 {result.cn0[-1]:.1f} dB-Hz")
    for ax in (ax_iq, ax_dop, ax_cn0, ax_epl):
        ax.grid(color=_MUTED, alpha=0.3)
        ax.tick_params(colors=_INK)
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
