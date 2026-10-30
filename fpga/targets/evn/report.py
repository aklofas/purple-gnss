"""Print the utilization and fmax line of a nextpnr --report JSON file."""

import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
used = {k: v.get("used", 0) for k, v in report["utilization"].items()}
dsp = sum(used.get(k, 0) for k in ("MULT18_CORE", "MULT18X36_CORE", "MULT36_CORE"))
fmax = min((v["achieved"] for v in report.get("fmax", {}).values()), default=0.0)
print(
    f"N_CHANNELS={sys.argv[2]}: LUT4 {used.get('OXIDE_COMB', 0)}  FF {used.get('OXIDE_FF', 0)}  "
    f"EBR {used.get('OXIDE_EBR', 0)}  DSP {dsp}  fmax {fmax:.1f} MHz"
)
