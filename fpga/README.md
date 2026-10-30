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
