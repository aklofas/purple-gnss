"""Parse Pocket SDR pocket_trk.py logs ($CH lines) for reference comparisons."""

from collections import defaultdict

import numpy as np


def load_pocketsdr_trk(path):
    """PRN -> array of (t_s, cn0_dbhz, code_offset_s, doppler_hz, carrier_cycles) rows."""
    rows = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.startswith("$CH,"):
                continue
            p = line.strip().split(",")
            rows[int(p[3])].append(
                (float(p[1]), float(p[5]), float(p[6]) * 1e-3, float(p[7]), float(p[8]))
            )
    return {prn: np.array(r) for prn, r in rows.items()}
