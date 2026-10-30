"""The dump read protocol: a dump that lands during or around the read is never lost unseen."""

import asyncio

from fpgamodel import regmap as R
from fpgamodel.channel import Dump
from tb.support import read_dump, read_dumps_now

K = 1
SUMS = {R.IE: 1, R.QE: -2, R.IP: 3, R.QP: -4, R.IL: 5, R.QL: -6}
EPOCH = ("read", R.ch(K, R.EPOCH))
SUM_READS = [("read", R.ch(K, off)) for off in (R.IE, R.QE, R.IP, R.QP, R.IL, R.QL)]
CLEAR = ("write", R.DUMP_CLEAR)
FULL = [EPOCH, *SUM_READS, EPOCH, CLEAR, EPOCH]  # a read with nothing landing on it
TORN = [EPOCH, *SUM_READS, EPOCH]  # a dump replaced the registers: no clear


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


def dump(epoch):
    return Dump(epoch, 1, -2, 3, -4, 5, -6, -1)


def test_quiet_read_returns_the_dump_once():
    bus = FakeBus([7, 7, 7])
    assert asyncio.run(read_dump(bus, K)) == (dump(7), False)
    assert bus.log == FULL


def test_dump_landing_during_the_read_discards_it_and_does_not_clear():
    bus = FakeBus([7, 8])
    assert asyncio.run(read_dump(bus, K)) == (None, True)
    assert bus.log == TORN  # the new dump stays pending, DUMP_MISSED counted the lost one


def test_dump_landing_around_the_clear_keeps_the_read_and_asks_for_another():
    bus = FakeBus([7, 7, 8])
    assert asyncio.run(read_dump(bus, K)) == (dump(7), True)
    assert bus.log == FULL


def test_read_dumps_now_collects_until_the_registers_are_quiet():
    bus = FakeBus([7, 8, 8, 8, 8])  # torn once, then a clean read of the newer dump
    assert asyncio.run(read_dumps_now(bus, K)) == [dump(8)]
    assert bus.log == TORN + FULL
    bus = FakeBus([7, 7, 8, 8, 8, 8])  # a dump around the clear: both come back
    assert asyncio.run(read_dumps_now(bus, K)) == [dump(7), dump(8)]
    assert bus.log == FULL + FULL
