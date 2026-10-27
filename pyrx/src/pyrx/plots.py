"""Diagnostic plots written to image files."""

import os

import matplotlib
import numpy as np

from pyrx.acquisition import AcqResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_INK = "#333333"
_MUTED = "#9aa5b1"
_BAR = "#9ecae1"
_BAR_DETECTED = "#2171b5"


def plot_acquisition(result: AcqResult, fs: float, path: str | os.PathLike) -> None:
    """Search grid (code offset x Doppler) and the Doppler cut through the peak."""
    n = result.grid.shape[1]
    offsets_ms = np.arange(n) / fs * 1e3
    fig, (ax_grid, ax_cut) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )
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
    plt.close(fig)


def plot_cn0(results: list[AcqResult], threshold: float, path: str | os.PathLike) -> None:
    """C/N0 per PRN with the detection threshold; detected PRNs are darker and labelled."""
    prns = [r.prn for r in results]
    cn0 = [r.cn0 for r in results]
    colors = [_BAR_DETECTED if r.detected else _BAR for r in results]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
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
    plt.close(fig)
