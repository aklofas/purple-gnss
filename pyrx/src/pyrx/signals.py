"""Carrier frequencies and wavelengths of the signals of several systems and bands.

A signal is the RINEX 3 system letter plus the two-character observation attribute
(`G1C`, `G2W`, `E5Q`).
"""



from pyrx.gpstime import C_LIGHT

# carrier frequencies (Hz) by system letter and band digit
FREQUENCY = {
    "G1": 1575.42e6,
    "G2": 1227.60e6,
    "G5": 1176.45e6,
    "E1": 1575.42e6,
    "E5": 1176.45e6,
    "E7": 1207.14e6,
}
L1_FREQ = FREQUENCY["G1"]


def signal_frequency(signal: str) -> float:
    """Carrier frequency (Hz) of a signal such as `G2W`."""
    return FREQUENCY[signal[:2]]


def wavelength(signal: str) -> float:
    """Carrier wavelength (m) of a signal."""
    return C_LIGHT / signal_frequency(signal)
