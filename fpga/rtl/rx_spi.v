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

    wire miso;
    assign spi_io_o  = {2'b00, miso, 1'b0};
    assign spi_io_oe = {2'b00, !spi_cs_n, 1'b0};
    wire _unused = &{1'b0, spi_io_i[3:1]};

    spi_slave spi (
        .clk(clk), .rst(rst),
        .sck(spi_sck), .mosi(spi_io_i[0]), .cs_n(spi_cs_n), .miso(miso),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_we(bus_we), .bus_re(bus_re),
        .bus_rdata(bus_rdata)
    );

    rx_core #(
        .N_CHANNELS(N_CHANNELS), .CODE_LEN(CODE_LEN), .ACC_W(ACC_W), .PB_DEPTH(PB_DEPTH)
    ) core (
        .clk(clk), .rst(rst),
        .pin_valid(pin_valid), .pin_sample(pin_sample), .pll_lock(pll_lock), .evt_in(evt_in),
        .link_bad_frames(16'd0), .link_overruns(8'd0), .link_underruns(8'd0),
        .bus_addr(bus_addr), .bus_wdata(bus_wdata), .bus_we(bus_we), .bus_re(bus_re),
        .bus_rdata(bus_rdata), .dump_irq(dump_irq)
    );
endmodule
