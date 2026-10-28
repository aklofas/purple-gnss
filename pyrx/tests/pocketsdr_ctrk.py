"""Reader for Pocket SDR's C tracker (`pocket_trk`) log, for the observables tests."""

from pathlib import Path

import numpy as np

# $CH,time,ch,rfch,sat,sig,prn,lock,cn0,coff_ms,dop,adr,ssync,bsync,fsync,rev,srev,
#     err_phas,err_code,tow_v,tow_ms,week,type,nnav,nerr,nlol,nfec
COLUMNS = ("time", "coff_ms", "dop", "adr", "tow_v", "tow_ms")
_INDEX = {"time": 1, "coff_ms": 9, "dop": 10, "adr": 11, "tow_v": 19, "tow_ms": 20}


def load_ctrk_ch(path) -> dict[int, np.ndarray]:
    """L1CA `$CH` rows per PRN as an array with columns `COLUMNS`."""
    rows: dict[int, list] = {}
    for line in Path(path).read_text().splitlines():
        f = line.split(",")
        if f[0] != "$CH" or f[5] != "L1CA":
            continue
        rows.setdefault(int(f[6]), []).append([float(f[_INDEX[c]]) for c in COLUMNS])
    return {prn: np.array(r) for prn, r in rows.items()}
