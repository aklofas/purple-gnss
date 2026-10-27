# purple-gps

An open RTK GNSS receiver built from scratch: the antenna and processing hardware, the
firmware that runs on it, and the positioning math that turns raw observations into a
centimetre-level fix. `pyrx/` holds the offline Python receiver used to develop and verify
every algorithm on recorded signals before it is ported to the FPGA and MCU.
