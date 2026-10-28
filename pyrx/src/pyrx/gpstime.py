"""GPS time: the week and seconds-of-week convention, its epoch, and the constants every
stage shares. Imports nothing from the signal chain or the positioning."""

from datetime import datetime, timedelta

import numpy as np

C_LIGHT = 299792458.0  # m/s
WEEK_S = 604800
GPS_EPOCH = datetime(1980, 1, 6)
GPS_WEEK_ROLLOVER = 2048  # first week of the current 1024-week era (2019-04-07)


def full_week(wn10: int) -> int:
    """Full GPS week from the broadcast 10-bit week number (valid until 2038-11-20)."""
    return wn10 + GPS_WEEK_ROLLOVER


def gps_datetime(week: int, tow: float) -> datetime:
    """Naive datetime in the GPS time scale for a week and seconds of week."""
    return GPS_EPOCH + timedelta(weeks=week, seconds=tow)


def week_tow(t: np.datetime64) -> tuple[int, float]:
    """GPS week and seconds of week of a numpy time stamp in the GPS time scale."""
    ns = int((t - np.datetime64(GPS_EPOCH)) / np.timedelta64(1, "ns"))
    week = ns // (WEEK_S * 10**9)
    return week, (ns - week * WEEK_S * 10**9) / 1e9
