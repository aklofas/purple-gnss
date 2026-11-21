`timescale 1ns/1ps
// Small FIFO between two unrelated clocks: gray-coded pointers, each carried into the other
// domain through two flops, and a memory read without a clock (an entry is only read after
// its write pointer has crossed, long after the data settled).
//
// A write with the FIFO full and a read with it empty are ignored. Each side has its own
// asynchronous reset, which also clears that side's copy of the other pointer. Resetting one
// side alone leaves the pointers apart, so both are reset, not necessarily at the same
// instant: between the two resets the FIFO's contents and flags mean nothing, and the side
// reset later must not rely on them.
module cdc_fifo #(
    parameter WIDTH      = 32,
    parameter DEPTH_LOG2 = 2
) (
    input  wire             wclk,
    input  wire             wrst,
    input  wire             we,
    input  wire [WIDTH-1:0] wdata,
    output wire             full,
    input  wire             rclk,
    input  wire             rrst,
    input  wire             re,
    output wire [WIDTH-1:0] rdata,
    output wire             empty
);
    localparam AW = DEPTH_LOG2;
    localparam [AW:0] ONE = 1;
    localparam [AW:0] FULL_DIFF = {(AW+1){1'b1}} << (AW - 1);  // the top two bits: a lap apart

    reg [WIDTH-1:0] mem [0:(1 << AW) - 1];
    reg [AW:0] wbin, wgray, rgray_w1, rgray_w2;  // write side, with its copy of the read pointer
    reg [AW:0] rbin, rgray, wgray_r1, wgray_r2;  // read side, with its copy of the write pointer

    // ---- write side ----
    wire [AW:0] wbin_next = wbin + ONE;
    assign full = (wgray ^ rgray_w2) == FULL_DIFF;
    wire push = we && !full;
    always @(posedge wclk) if (push) mem[wbin[AW-1:0]] <= wdata;
    always @(posedge wclk or posedge wrst) begin
        if (wrst) begin
            wbin <= {(AW+1){1'b0}}; wgray <= {(AW+1){1'b0}};
            rgray_w1 <= {(AW+1){1'b0}}; rgray_w2 <= {(AW+1){1'b0}};
        end else begin
            rgray_w1 <= rgray;
            rgray_w2 <= rgray_w1;
            if (push) begin
                wbin  <= wbin_next;
                wgray <= (wbin_next >> 1) ^ wbin_next;
            end
        end
    end

    // ---- read side ----
    wire [AW:0] rbin_next = rbin + ONE;
    assign empty = rgray == wgray_r2;
    assign rdata = mem[rbin[AW-1:0]];
    always @(posedge rclk or posedge rrst) begin
        if (rrst) begin
            rbin <= {(AW+1){1'b0}}; rgray <= {(AW+1){1'b0}};
            wgray_r1 <= {(AW+1){1'b0}}; wgray_r2 <= {(AW+1){1'b0}};
        end else begin
            wgray_r1 <= wgray;
            wgray_r2 <= wgray_r1;
            if (re && !empty) begin
                rbin  <= rbin_next;
                rgray <= (rbin_next >> 1) ^ rbin_next;
            end
        end
    end
endmodule
