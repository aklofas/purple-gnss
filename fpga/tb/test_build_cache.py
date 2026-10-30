"""The bench build cache keys on everything that changes what gets compiled."""

from pathlib import Path

from tb.conftest import build_dir_for

SRC = [Path("rtl/channel.v")]


def test_waves_setting_selects_its_own_build(monkeypatch):
    monkeypatch.delenv("WAVES", raising=False)
    plain = build_dir_for("tb_channel", SRC, {})
    monkeypatch.setenv("WAVES", "1")
    assert build_dir_for("tb_channel", SRC, {}) != plain


def test_parameters_and_sources_select_their_own_builds():
    base = build_dir_for("tb_channel", SRC, {})
    assert build_dir_for("tb_channel", SRC, {"CODE_LEN": 1024}) != base
    assert build_dir_for("tb_channel", SRC + [Path("tb/tb_channel.v")], {}) != base
    assert build_dir_for("tb_channel", SRC, {}) == base
