#!/usr/bin/env bash
# Build the EVN bitstream: build.sh [N_CHANNELS] -> build/n<N>/top.bit and report.txt
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-1}
OUT=build/n$N
RTL=../../rtl
mkdir -p "$OUT"

yosys -q -l "$OUT/yosys.log" -p "
    read_verilog $RTL/channel.v $RTL/playback.v $RTL/rx_core.v $RTL/spi_slave.v $RTL/rx_spi.v top.v;
    chparam -set N_CHANNELS $N top;
    synth_nexus -top top -json $OUT/top.json"

# nextpnr's exit code is checked (set -e): it can abort while writing the PLL's FASM section
# and leave a truncated file that prjoxide would pack without complaint.
nextpnr-nexus --device LIFCL-40-9BG400C --json "$OUT/top.json" --pdc evn.pdc \
    --fasm "$OUT/top.fasm" --report "$OUT/report.json" -l "$OUT/nextpnr.log" -q
pll_lines=$(grep -c "PLL" "$OUT/top.fasm" || true)
if [ "$pll_lines" -lt 100 ]; then
    echo "PLL section of $OUT/top.fasm has only $pll_lines lines: truncated FASM" >&2
    exit 1
fi

prjoxide pack --compress "$OUT/top.fasm" "$OUT/top.bit"
python3 report.py "$OUT/report.json" "$N" | tee "$OUT/report.txt"
ls -l "$OUT/top.bit"
