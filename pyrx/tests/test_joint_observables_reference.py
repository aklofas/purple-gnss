"""GPS L5 pilot observations beside L1 from the two recordings of 2021-12-26: the L1 file
and the L5 band file hold the same 30 s on one sample clock."""

import numpy as np
from conftest import capture_file
from rtklib import rnx2rtkp_single

from pyrx.acquisition import acquire_l1ca, acquire_signal
from pyrx.navdecode import decode_lnav
from pyrx.observables import LLI_HALF_CYCLE, LLI_SLIP, make_joint_observables, make_observables
from pyrx.rinex import read_rinex_observations, write_rinex_obs, write_rinex_observations
from pyrx.samples import read_samples
from pyrx.signals import signal_frequency, wavelength
from pyrx.signalspec import L5
from pyrx.tracking import track_channels

L1_FILE = "L1_20211226_082401_24MHz_I.bin"
L5_FILE = "L5_20211226_082401_24MHz_IQ.bin"
BRDC = "brdc/BRDC00WRD_R_20213600000_01D_GN.rnx"
FS, L1_IF = 24e6, 6.0e6
L1_PRNS = [4, 16, 18, 26, 27, 29, 31]
BOTH = ["G04", "G18", "G26", "G27"]  # tracked on L1 and on L5


def _records(path):
    """The observation records (not the header lines that also start with the system letter)."""
    return [ln for ln in path.read_text().splitlines() if ln[:1] == "G" and ln[1:3].isdigit()]


def test_joint_file_from_the_2021_pair(captures, tmp_path):
    l1, l5 = capture_file(captures, L1_FILE), capture_file(captures, L5_FILE)
    head = read_samples(l1, FS, False, 0.0, 0.020)
    seeds = [
        (a.prn, a.code_offset, a.doppler, a.cn0)
        for a in (acquire_l1ca(head, FS, L1_IF, p) for p in L1_PRNS)
    ]
    tracks = track_channels(l1, FS, False, L1_IF, seeds)
    lnavs = [decode_lnav(r) for r in tracks]
    head = read_samples(l5, FS, True, 0.0, 0.020)
    seeds = [
        (a.prn, a.code_offset, a.doppler, a.cn0)
        for a in (acquire_signal(head, FS, 0.0, L5, int(s[1:])) for s in BOTH)
    ]
    pilots = track_channels(l5, FS, True, 0.0, seeds, signal="L5")

    joint_file, l1_file = tmp_path / "joint.obs", tmp_path / "l1.obs"
    write_rinex_observations(joint_file, make_joint_observables(tracks, lnavs, pilots))
    write_rinex_obs(l1_file, make_observables(tracks, lnavs))

    # the L1 fields are those of the L1-only file, character for character
    joint_records, l1_records = _records(joint_file), _records(l1_file)
    assert [r[: 3 + 4 * 16].rstrip() for r in joint_records] == [r.rstrip() for r in l1_records]
    text = joint_file.read_text()
    assert "G    8 C1C L1C D1C S1C C5Q L5Q D5Q S5Q" in text and "G L5Q  0.00000" in text

    epochs = read_rinex_observations(joint_file, ("G1C", "G5Q"))
    assert len(epochs) >= 25
    ratio = signal_frequency("G5Q") / signal_frequency("G1C")
    offsets = {}
    for sat in BOTH:
        rows = [e for e in epochs if sat in e.sats]
        values = {
            f: np.array([getattr(e, f)[e.sats.index(sat)] for e in rows])
            for f in ("pseudorange", "carrier_phase", "doppler", "lli")
        }
        tow = np.array([e.tow for e in rows])
        # both signals at every epoch, both phases too
        assert len(rows) >= 25 and np.isfinite(values["pseudorange"]).all(), sat
        assert np.isfinite(values["carrier_phase"]).all(), sat
        offsets[sat] = np.mean(values["pseudorange"][:, 1] - values["pseudorange"][:, 0])
        # the Doppler in the ratio of the carriers; the L1 Costas loop scatters by up to
        # 0.9 Hz at 41 to 43 dB-Hz
        d = values["doppler"][:, 1] - ratio * values["doppler"][:, 0]
        assert abs(d.mean()) < 0.3 and np.abs(d).max() < 1.5, (sat, d.mean(), np.abs(d).max())
        # the geometry-free combination of the two phases changes only with the ionosphere:
        # a line over the half minute, and 0.8 to 2.3 mm of scatter about it
        gf = (
            wavelength("G1C") * values["carrier_phase"][:, 0]
            - wavelength("G5Q") * values["carrier_phase"][:, 1]
        )
        residual = gf - np.polyval(np.polyfit(tow - tow[0], gf, 1), tow - tow[0])
        assert residual.std() < 0.005 and np.abs(residual).max() < 0.01, (sat, residual.std())
        # the pilot's phase: a slip flag on the first observation only, never a half cycle
        lli = values["lli"][:, 1].astype(int)
        assert lli[0] == LLI_SLIP and not lli[1:].any() and not (lli & LLI_HALF_CYCLE).any(), sat
    # C5Q minus C1C is the front end's delay between its two channels (the L5 path is about
    # 97 ns shorter), common to the satellites; they differ by the ionosphere and the code
    # noise, 2.3 m at most here
    common = np.mean(list(offsets.values()))
    assert -40.0 < common < -20.0, common
    assert max(abs(v - common) for v in offsets.values()) < 5.0, offsets
    # a satellite without L1 time gets no L5 observation, one without L5 no L5 fields
    assert {s for e in epochs for s in e.sats} == {f"G{p:02d}" for p in L1_PRNS}

    # RTKLIB's single-point fix is the same from either file: the L5 columns disturb nothing
    brdc = capture_file(captures, BRDC)
    joint_fix, _ = rnx2rtkp_single(joint_file, brdc, tmp_path / "joint")
    l1_fix, _ = rnx2rtkp_single(l1_file, brdc, tmp_path / "l1")
    assert len(joint_fix) >= 25 and joint_fix == l1_fix
