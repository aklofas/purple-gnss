`timescale 1ns/1ps
// Bench wrapper around one channel: streams 2-bit samples from samples.hex (read when
// `load` rises), one per clock with `gap` idle cycles between them, holds the stream
// while `pause_en` and consumed == pause_at, and reports the consumed-sample count as it
// stood when each dump's last sample went in (dump_sample). Test-only.
module tb_channel #(
    parameter CODE_LEN = 1023,
    parameter CODE_AW = $clog2(CODE_LEN),
    parameter ACC_W = 20,
    parameter MAX_SAMPLES = 5_000_000,
    parameter DUMP_LATENCY = 2   // clock edges from the one taking the wrap sample to dump high
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        load,
    input  wire [31:0] n_samples,
    input  wire [7:0]  gap,
    input  wire        stream_en,
    input  wire        pause_en,
    input  wire [31:0] pause_at,
    output reg  [31:0] consumed,
    output wire        paused,
    output wire        done,
    output wire [31:0] dump_sample,
    input  wire        enable,
    input  wire [31:0] carr_step,
    input  wire [31:0] code_step,
    input  wire [1:0]  spacing_q,
    input  wire        load_phase,
    input  wire [31:0] carr_phase_in,
    input  wire [31:0] code_frac_in,
    input  wire [CODE_AW+1:0] chip_q_in,
    input  wire        code_we,
    input  wire [CODE_AW-6:0] code_waddr,
    input  wire [31:0] code_wdata,
    input  wire        meas_strobe,
    output wire        dump,
    output wire signed [ACC_W-1:0] ie, qe, ip, qp, il, ql,
    output wire [31:0] epoch,
    output wire [31:0] meas_carr_phase, meas_code_frac, meas_epoch,
    output wire [CODE_AW+1:0] meas_chip_q
);
    reg [1:0] mem [0:MAX_SAMPLES-1];
    always @(posedge load) $readmemh("samples.hex", mem);

    reg [7:0] gapcnt;
    assign paused = pause_en && (consumed == pause_at);
    assign done = consumed == n_samples;
    wire valid = stream_en && !paused && !done && (gapcnt == 0);
    wire [1:0] sample = mem[consumed[$clog2(MAX_SAMPLES)-1:0]];

    always @(posedge clk) begin
        if (rst) begin
            consumed <= 0;
            gapcnt <= 0;
        end else if (valid) begin
            consumed <= consumed + 1;
            gapcnt <= gap;
        end else if (gapcnt != 0) begin
            gapcnt <= gapcnt - 1;
        end
    end

    // consumed as it stood DUMP_LATENCY edges ago: the count right after a dump's last
    // sample went in
    reg [31:0] consumed_d [1:DUMP_LATENCY];
    integer i;
    always @(posedge clk) begin
        consumed_d[1] <= consumed;
        for (i = 2; i <= DUMP_LATENCY; i = i + 1) consumed_d[i] <= consumed_d[i-1];
    end
    assign dump_sample = consumed_d[DUMP_LATENCY];

    channel #(.CODE_LEN(CODE_LEN), .CODE_AW(CODE_AW), .ACC_W(ACC_W)) ch (
        .clk(clk), .rst(rst),
        .sample_valid(valid), .sample(sample),
        .enable(enable), .carr_step(carr_step), .code_step(code_step), .spacing_q(spacing_q),
        .load_phase(load_phase), .carr_phase_in(carr_phase_in), .code_frac_in(code_frac_in),
        .chip_q_in(chip_q_in),
        .code_we(code_we), .code_waddr(code_waddr), .code_wdata(code_wdata),
        .meas_strobe(meas_strobe),
        .dump(dump), .ie(ie), .qe(qe), .ip(ip), .qp(qp), .il(il), .ql(ql), .epoch(epoch),
        .meas_carr_phase(meas_carr_phase), .meas_code_frac(meas_code_frac),
        .meas_epoch(meas_epoch), .meas_chip_q(meas_chip_q)
    );
endmodule
