`timescale 1ns/1ps
// rx_core behind the SPI slave. The four data lanes come as input, output and output-enable
// buses, so a top level wraps them in tristate buffers and a bench can see when the slave
// drives; lane 0 is MOSI and lane 1 MISO of a single-lane master.
module rx_spi #(
    parameter N_CHANNELS = 1,
    parameter CODE_LEN   = 1023,
    parameter ACC_W      = 25,
    parameter PB_DEPTH   = 147456
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       spi_sck,
    input  wire       spi_cs_n,
    input  wire [3:0] spi_io_i,
    output wire [3:0] spi_io_o,
    output wire [3:0] spi_io_oe,
    input  wire       pin_valid,
    input  wire [1:0] pin_sample,
    input  wire       pll_lock,
    input  wire       evt_in,
    output wire       dump_irq
);
    wire [15:0] bus_addr;
    wire [31:0] bus_wdata, bus_rdata;
    wire        bus_we, bus_re;

    wire [15:0] bad_frames;
    wire [7:0]  overruns, underruns;
    wire        link_clear;

    spi_slave spi (
        .clk(clk), .rst(rst), .clr(link_clear),
        .sck(spi_sck), .cs_n(spi_cs_n), .io_i(spi_io_i), .io_o(spi_io_o), .io_oe(spi_io_oe),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_we(bus_we), .bus_re(bus_re),
        .bus_rdata(bus_rdata),
        .bad_frames(bad_frames), .overruns(overruns), .underruns(underruns)
    );

    rx_core #(
        .N_CHANNELS(N_CHANNELS), .CODE_LEN(CODE_LEN), .ACC_W(ACC_W), .PB_DEPTH(PB_DEPTH)
    ) core (
        .clk(clk), .rst(rst),
        .pin_valid(pin_valid), .pin_sample(pin_sample), .pll_lock(pll_lock), .evt_in(evt_in),
        .link_bad_frames(bad_frames), .link_overruns(overruns), .link_underruns(underruns),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_we(bus_we), .bus_re(bus_re),
        .bus_rdata(bus_rdata), .link_clear(link_clear), .dump_irq(dump_irq)
    );
endmodule
