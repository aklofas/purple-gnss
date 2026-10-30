#!/usr/bin/env bash
# Build the EVN bitstream: build.sh [N_CHANNELS] -> build/n<N>/top.bit and report.txt
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-1}
OUT=build/n$N
RTL=../../rtl
rm -rf "$OUT"   # a failed run must not leave the previous run's bitstream behind
mkdir -p "$OUT"

yosys -q -l "$OUT/yosys.log" -p "
    read_verilog $RTL/channel.v $RTL/playback.v $RTL/rx_core.v $RTL/spi_slave.v $RTL/rx_spi.v top.v;
    chparam -set N_CHANNELS $N top;
    synth_nexus -top top -json $OUT/top.json"

# nextpnr can abort while writing the PLL's FASM section and leave a truncated file that
# prjoxide would pack without complaint: its exit code (set -e), its closing log line and the
# FASM's PLL and oscillator sections are all checked.
nextpnr-nexus --device LIFCL-40-9BG400C --json "$OUT/top.json" --pdc evn.pdc \
    --fasm "$OUT/top.fasm" --report "$OUT/report.json" -l "$OUT/nextpnr.log" -q
grep -q "Program finished normally" "$OUT/nextpnr.log" || {
    echo "nextpnr did not finish normally, see $OUT/nextpnr.log" >&2; exit 1; }
pll_lines=$(grep -c "^IP_PLL_" "$OUT/top.fasm" || true)
if [ "$pll_lines" -lt 100 ]; then
    echo "PLL section of $OUT/top.fasm has only $pll_lines lines: truncated FASM" >&2
    exit 1
fi
grep -q "OSC_CORE.HF_OSC_EN.ENABLED" "$OUT/top.fasm" || {
    echo "$OUT/top.fasm has no oscillator configuration: truncated FASM" >&2; exit 1; }

# nextpnr drops a port constraint whose selector matches nothing without a warning, so the
# pulls in the FASM are counted against the ones evn.pdc asks for (one port per line there).
check_pulls() {
    want=$(grep -c "PULLMODE=$1" evn.pdc || true)
    got=$(grep -c "PULLMODE\.$1\$" "$OUT/top.fasm" || true)
    if [ "$want" -ne "$got" ]; then
        echo "evn.pdc asks for PULLMODE=$1 on $want pins, $OUT/top.fasm has it on $got" >&2
        exit 1
    fi
}
check_pulls UP
check_pulls DOWN

prjoxide pack --compress "$OUT/top.fasm" "$OUT/top.bit"
{
    python3 report.py "$OUT/report.json" "$N"
    echo "source $(git describe --always --dirty --tags)"
} | tee "$OUT/report.txt"
ls -l "$OUT/top.bit"
