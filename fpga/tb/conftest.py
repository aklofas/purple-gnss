"""cocotb benches as pytest tests: build once per (toplevel, sources, parameters), then run
one cocotb test case per pytest function. SIM selects the simulator (icarus, verilator)."""

import hashlib
import os
from pathlib import Path

from cocotb_tools.runner import Runner, get_results, get_runner

TB = Path(__file__).resolve().parent
FPGA = TB.parent
RTL = FPGA / "rtl"
SIM = os.environ.get("SIM", "icarus")

_built: dict[Path, Runner] = {}


def build_dir_for(toplevel: str, sources: list[Path], parameters: dict) -> Path:
    """One build directory per (toplevel, sources, parameters, simulator, WAVES): the waves
    dump module is compiled in only at build time, so a WAVES toggle must not reuse a build."""
    key = (toplevel, tuple(str(s) for s in sources), tuple(sorted(parameters.items())), SIM, waves())
    digest = hashlib.sha1(repr(key).encode()).hexdigest()[:8]
    return TB / "sim_build" / f"{toplevel}-{SIM}-{digest}"


def waves() -> bool:
    return bool(os.environ.get("WAVES"))


def build(toplevel: str, sources: list[Path], parameters: dict | None = None) -> Runner:
    """Build `toplevel` from `sources`; returns the runner that built it (cached per session)."""
    parameters = parameters or {}
    build_dir = build_dir_for(toplevel, sources, parameters)
    if build_dir not in _built:
        runner = get_runner(SIM)
        runner.build(
            sources=list(sources),
            hdl_toplevel=toplevel,
            parameters=parameters,
            build_dir=build_dir,
            timescale=("1ns", "1ps"),
            waves=waves(),
        )
        _built[build_dir] = runner
    return _built[build_dir]


def run_cocotb(runner: Runner, toplevel: str, module: str, testcase: str) -> None:
    """Run one cocotb test case from `module` (importable from fpga/) and fail on any error."""
    test_dir = Path(runner.build_dir) / f"test-{testcase}"
    test_dir.mkdir(exist_ok=True)
    xml = runner.test(
        hdl_toplevel=toplevel,
        test_module=module,
        testcase=testcase,
        test_dir=test_dir,
        timescale=("1ns", "1ps"),
        waves=waves(),
    )
    n_tests, n_failed = get_results(xml)
    assert n_tests == 1 and n_failed == 0, f"{testcase}: {n_failed} of {n_tests} failed, see {xml}"
