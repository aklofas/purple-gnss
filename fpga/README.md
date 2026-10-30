# FPGA

The hardware receiver's FPGA side: shared Verilog cores, a bit-true Python model of them, a
cocotb bench that holds the two to exact agreement, and one top level per board.

```
fpga/
  rtl/        shared Verilog modules
  model/      fpgamodel, the bit-true Python model (installable package)
  tb/         cocotb benches, run as pytest tests
  targets/    one directory per board: top level, pin file, build script
```

## Running the bench

From the repository root, with the project venv:

```
uv pip install --python .venv/bin/python -e fpga/model
.venv/bin/pytest fpga
```

Tests on the recordings need `PYRX_CAPTURES` set to the directory holding them, exactly as the
`pyrx` suite does; without it they skip.

## Quantization loss

The channel works on 2-bit samples with a 3-level local oscillator. Its cost against
`pyrx`'s float correlator, measured by `fpgamodel.compare` as the ratio of post-correlation
SNRs on the same samples (prompt power on the tracked PRN over the mean prompt power on
eight PRNs absent from the signal, 100 one-period windows, fixed NCO):

| Signal | What the number measures | Loss |
|---|---|---|
| Synthetic L1 C/A at 45 dB-Hz (`pyrx.synth`), float samples for `pyrx`, 2-bit for the model | 2-bit quantization plus the 3-level LO | 0.86 dB |
| Recording `L1_20211226_082401_24MHz_I.bin`, PRN 26 at 48.7 dB-Hz | The 3-level LO alone: the recording already holds 2-bit samples (values ±1 and ±3), so `pyrx` sees the same quantization | 0.27 dB |

The bench bounds are 1.5 dB total on synthetic signal and 0.8 dB on the recording
(`fpga/model/tests/test_compare.py`).
