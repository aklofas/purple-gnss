`timescale 1ns/1ps
// Bench wrapper around rx_spi: streams 2-bit samples from samples.hex (16 per 32-bit word,
// read when `load` rises) into the live sample pins, one per clock. Test-only.
module tb_rx_spi #(
    parameter N_CHANNELS = 1,
    parameter MAX_WORDS = 5_000_000
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        load,
    input  wire [31:0] n_samples,
    input  wire        stream_en,
    output reg  [31:0] consumed,
    output wire        done,
    input  wire        spi_sck,
    input  wire        spi_mosi,
    input  wire        spi_cs_n,
    output wire        spi_miso,
    output wire        dump_irq
);
    reg [31:0] mem [0:MAX_WORDS-1];
    always @(posedge load) $readmemh("samples.hex", mem);

    assign done = consumed == n_samples;
    wire valid = stream_en && !done;
    wire [31:0] word = mem[consumed[$clog2(MAX_WORDS)+3:4]];
    wire [1:0] sample = word[consumed[3:0] * 2 +: 2];
    always @(posedge clk) begin
        if (rst) consumed <= 32'd0;
        else if (valid) consumed <= consumed + 32'd1;
    end

    rx_spi #(.N_CHANNELS(N_CHANNELS)) rx (
        .clk(clk), .rst(rst),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_cs_n(spi_cs_n), .spi_miso(spi_miso),
        .pin_valid(valid), .pin_sample(sample), .pll_lock(1'b1), .dump_irq(dump_irq)
    );
endmodule
