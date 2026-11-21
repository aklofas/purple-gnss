`timescale 1ns/1ps
// Stamps an edge on an input pin with the count of valid samples.
//
// evt_in passes a two-flop synchronizer; the chosen edge (rising, or falling with `falling`
// set) seen while `enable` is high latches sample_cnt as it stands on that clock, which is the
// index of the sample then going in, and increments seq: the stamp is the sample two clocks
// after the pin changed. An edge that arrives before 16 valid samples have gone in since the
// last stamp is dropped and counted in `missed`, so the stamp holds still long enough to be
// read as seq, stamp, seq. The pin is followed while disabled too, so enabling with the pin
// already at its active level stamps nothing.
module event_stamp (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,
    input  wire        falling,
    input  wire        evt_in,
    input  wire        sample_valid,
    input  wire [47:0] sample_cnt,
    output reg  [31:0] seq,
    output reg  [47:0] stamp,
    output reg  [31:0] missed
);
    localparam [4:0] MIN_SPACING = 5'd16;

    reg [1:0] sync;
    reg       level;
    always @(posedge clk) begin
        sync  <= {sync[0], evt_in};
        level <= sync[1];
    end
    wire edge_seen = enable && (falling ? (level && !sync[1]) : (!level && sync[1]));

    reg [4:0] since;  // valid samples since the last stamp, saturating at MIN_SPACING
    always @(posedge clk) begin
        if (rst) begin
            seq <= 32'd0; stamp <= 48'd0; missed <= 32'd0; since <= MIN_SPACING;
        end else if (edge_seen && (since == MIN_SPACING)) begin
            seq   <= seq + 32'd1;
            stamp <= sample_cnt;
            since <= sample_valid ? 5'd1 : 5'd0;
        end else begin
            if (edge_seen) missed <= missed + 32'd1;
            if (sample_valid && (since != MIN_SPACING)) since <= since + 5'd1;
        end
    end
endmodule
