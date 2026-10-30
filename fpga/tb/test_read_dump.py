"""The dump read protocol: a dump that lands during the read is detected and discarded."""

import asyncio

from fpgamodel import regmap as R
from fpgamodel.channel import Dump
from tb.support import read_dump, read_dump_retrying

K = 1
SUMS = {R.IE: 1, R.QE: -2, R.IP: 3, R.QP: -4, R.IL: 5, R.QL: -6}
SUM_ADDRS = [R.ch(K, off) for off in (R.IE, R.QE, R.IP, R.QP, R.IL, R.QL)]


class FakeBus:
    """Returns scripted EPOCH values in order and logs every access in sequence."""

    def __init__(self, epochs: list[int]):
        self.epochs = list(epochs)
        self.log: list[tuple[str, int]] = []

    async def read(self, addr: int) -> int:
        self.log.append(("read", addr))
        if addr == R.ch(K, R.EPOCH):
            return self.epochs.pop(0)
        return SUMS[addr - R.ch(K, 0)] & 0xFFFF_FFFF

    async def write(self, addr: int, data: int) -> None:
        self.log.append(("write", addr))
        assert (addr, data) == (R.DUMP_CLEAR, 1 << K)


ONE_READ = [("read", R.ch(K, R.EPOCH))] + [("read", a) for a in SUM_ADDRS]
ONE_READ += [("write", R.DUMP_CLEAR), ("read", R.ch(K, R.EPOCH))]


def test_stable_read_returns_the_dump_in_the_documented_order():
    bus = FakeBus([7, 7])
    assert asyncio.run(read_dump(bus, K)) == Dump(7, 1, -2, 3, -4, 5, -6, -1)
    assert bus.log == ONE_READ  # EPOCH, the six sums, the clear, EPOCH again


def test_dump_landing_during_the_read_is_discarded_after_the_clear():
    bus = FakeBus([7, 8])
    assert asyncio.run(read_dump(bus, K)) is None
    assert bus.log == ONE_READ  # the clear still went out before the second EPOCH read


def test_retry_reads_again_until_a_read_is_coherent():
    bus = FakeBus([7, 8, 8, 8])
    assert asyncio.run(read_dump_retrying(bus, K)) == Dump(8, 1, -2, 3, -4, 5, -6, -1)
    assert bus.log == ONE_READ * 2
