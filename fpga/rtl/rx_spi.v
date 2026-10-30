`timescale 1ns/1ps
// rx_core behind the single-lane SPI slave: the bring-up interface for the EVN.
module rx_spi #(
    parameter N_CHANNELS = 1,
    parameter CODE_LEN   = 1023,
    parameter CODE_AW    = 10,
    parameter ACC_W      = 20,
    parameter PB_DEPTH   = 147456
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       spi_sck,
    input  wire       spi_mosi,
    input  wire       spi_cs_n,
    output wire       spi_miso,
    input  wire       pin_valid,
    input  wire [1:0] pin_sample,
    input  wire       pll_lock,
    output wire       dump_irq
);
    wire [15:0] bus_addr;
    wire [31:0] bus_wdata, bus_rdata;
    wire        bus_we, bus_re;

    spi_slave spi (
        .clk(clk), .rst(rst),
        .sck(spi_sck), .mosi(spi_mosi), .cs_n(spi_cs_n), .miso(spi_miso),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_we(bus_we), .bus_re(bus_re),
        .bus_rdata(bus_rdata)
    );

    rx_core #(
        .N_CHANNELS(N_CHANNELS), .CODE_LEN(CODE_LEN), .CODE_AW(CODE_AW),
        .ACC_W(ACC_W), .PB_DEPTH(PB_DEPTH)
    ) core (
        .clk(clk), .rst(rst),
        .pin_valid(pin_valid), .pin_sample(pin_sample), .pll_lock(pll_lock),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_we(bus_we), .bus_re(bus_re),
        .bus_rdata(bus_rdata), .dump_irq(dump_irq)
    );
endmodule
