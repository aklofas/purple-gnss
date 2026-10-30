"""The dump read protocol: a dump that lands during the read is detected and discarded."""

import asyncio

from fpgamodel import regmap as R
from fpgamodel.channel import Dump
from tb.support import read_dump, read_dump_retrying

K = 1
SUMS = {R.IE: 1, R.QE: -2, R.IP: 3, R.QP: -4, R.IL: 5, R.QL: -6}


class FakeBus:
    """Returns scripted DUMP_MISSED values in order and records every write."""

    def __init__(self, missed: list[int]):
        self.missed = list(missed)
        self.writes: list[tuple[int, int]] = []

    async def read(self, addr: int) -> int:
        if addr == R.ch(K, R.DUMP_MISSED):
            return self.missed.pop(0)
        if addr == R.ch(K, R.EPOCH):
            return 7
        return SUMS[addr - R.ch(K, 0)] & 0xFFFF_FFFF

    async def write(self, addr: int, data: int) -> None:
        self.writes.append((addr, data))


def test_stable_read_returns_the_dump_and_clears_it():
    bus = FakeBus([3, 3])
    assert asyncio.run(read_dump(bus, K)) == Dump(7, 1, -2, 3, -4, 5, -6, -1)
    assert bus.writes == [(R.DUMP_CLEAR, 1 << K)]
    assert bus.missed == []  # DUMP_MISSED read before and after


def test_dump_landing_during_the_read_is_discarded_after_the_clear():
    bus = FakeBus([3, 4])
    assert asyncio.run(read_dump(bus, K)) is None
    assert bus.writes == [(R.DUMP_CLEAR, 1 << K)]  # the clear comes before the second read


def test_retry_reads_again_until_a_read_is_coherent():
    bus = FakeBus([3, 4, 4, 4])
    assert asyncio.run(read_dump_retrying(bus, K)) == Dump(7, 1, -2, 3, -4, 5, -6, -1)
    assert bus.writes == [(R.DUMP_CLEAR, 1 << K)] * 2
