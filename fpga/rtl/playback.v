`timescale 1ns/1ps
// Sample playback RAM: PB_DEPTH 2-bit samples held 16 to a 32-bit word, written over the
// register bus and streamed one per clock from sample 0 for `len` samples after `start`.
// `len` is clamped to PB_DEPTH; a start with len 0 is done at once. The RAM contents
// survive reset; the counters do not.
module playback #(
    parameter PB_DEPTH = 147456,
    parameter AW = $clog2(PB_DEPTH / 16)
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        wr_we,
    input  wire [AW-1:0] wr_addr,
    input  wire [31:0] wr_data,
    input  wire        start,
    input  wire [31:0] len,
    output reg         out_valid,
    output wire [1:0]  out_sample,
    output reg         running,
    output reg         done
);
    localparam WORDS = PB_DEPTH / 16;
    (* syn_ramstyle = "block_ram" *) reg [31:0] mem [0:WORDS-1];
    always @(posedge clk) if (wr_we) mem[wr_addr] <= wr_data;

    reg  [31:0] idx, len_r;
    reg  [31:0] rd_word;
    reg  [3:0]  lo_d;
    wire [31:0] len_clamped = (len > PB_DEPTH) ? PB_DEPTH : len;
    wire [31:0] idx_next = idx + 32'd1;

    always @(posedge clk) begin
        out_valid <= 1'b0;
        if (rst) begin
            running <= 1'b0;
            done    <= 1'b0;
            idx     <= 32'd0;
            len_r   <= 32'd0;
        end else if (start) begin
            idx     <= 32'd0;
            len_r   <= len_clamped;
            running <= (len_clamped != 32'd0);
            done    <= (len_clamped == 32'd0);
        end else if (running) begin
            rd_word   <= mem[idx[AW+3:4]];
            lo_d      <= idx[3:0];
            out_valid <= 1'b1;
            idx       <= idx_next;
            if (idx_next == len_r) begin
                running <= 1'b0;
                done    <= 1'b1;
            end
        end
    end
    assign out_sample = rd_word[lo_d * 2 +: 2];
endmodule
