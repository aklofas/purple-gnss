`timescale 1ns/1ps
// One GNSS tracking channel: carrier wipe-off with a 3-level local oscillator, early /
// prompt / late code correlation at quarter-chip tap spacing, integrate-and-dump once per
// code period. Everything with a feedback loop lives in software.
//
// Samples are 2-bit {sign, mag} (values +/-1, +/-3) and count only while sample_valid and
// enable are both high; a disabled channel freezes. The code is held as CODE_LEN bits in
// 32-bit words (bit set = chip value -1), written over code_we. The code NCO counts quarter
// chips and ticks at most once per sample, so the sample rate must exceed four times the
// chip rate; the chip index wrapping at 4 * CODE_LEN dumps the six sums into ie..ql, raises
// dump for one clock, increments epoch and copies the step inputs into the active NCO
// steps (while disabled the inputs are copied every clock, so a write takes effect at once).
// load_phase, honored only while disabled, sets the NCO accumulators directly. After a code
// write or a phase load the channel needs about eight clocks before enable to refresh the
// chips around the current code phase. meas_strobe, raised with a sample, latches the NCO
// phases, chip index and epoch count as they stood before that sample into the meas_*
// outputs; every channel sees the same strobe, so the latched states are simultaneous.
//
// Timing: a sample taken at clock edge E shows in a dump raised DUMP_LATENCY edges later.
module channel #(
    parameter CODE_LEN = 1023,
    parameter CODE_AW  = 10,
    parameter ACC_W    = 20
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        sample_valid,
    input  wire [1:0]  sample,            // {sign, mag}
    input  wire        enable,
    input  wire [31:0] carr_step,
    input  wire [31:0] code_step,         // quarter chips per sample, x 2^32
    input  wire [1:0]  spacing_q,         // early/late offset in quarter chips: 1 or 2
    input  wire        load_phase,
    input  wire [31:0] carr_phase_in,
    input  wire [31:0] code_frac_in,
    input  wire [CODE_AW+1:0] chip_q_in,  // quarter-chip index
    input  wire        code_we,
    input  wire [CODE_AW-6:0] code_waddr, // 32-chip word index
    input  wire [31:0] code_wdata,
    input  wire        meas_strobe,
    output reg         dump,
    output reg  signed [ACC_W-1:0] ie, qe, ip, qp, il, ql,
    output reg  [31:0] epoch,
    output wire [31:0] meas_carr_phase,
    output wire [31:0] meas_code_frac,
    output wire [31:0] meas_epoch,
    output wire [CODE_AW+1:0] meas_chip_q
);
    /* verilator lint_off UNUSEDPARAM */
    localparam DUMP_LATENCY = 2;  // read by the bench, not by the logic
    /* verilator lint_on UNUSEDPARAM */
    localparam [CODE_AW+1:0] N_Q = 4 * CODE_LEN;
    localparam CODE_WORDS = (CODE_LEN + 31) / 32;

    // ---- code memory and the chips around the current one ---------------------------
    (* syn_ramstyle = "block_ram" *) reg [31:0] code_mem [0:CODE_WORDS-1];
    always @(posedge clk) if (code_we) code_mem[code_waddr] <= code_wdata;

    reg  [CODE_AW-1:0] rd_chip;
    reg  [31:0] rd_word;
    reg  [4:0]  rd_bit;
    always @(posedge clk) begin
        rd_word <= code_mem[rd_chip[CODE_AW-1:5]];
        rd_bit  <= rd_chip[4:0];
    end
    wire rd_data = rd_word[rd_bit];

    reg [31:0] carr_acc, code_acc, carr_step_a, code_step_a;
    reg [CODE_AW+1:0] chip_q;
    reg [31:0] epoch0;  // dumps so far, counted where the wrap happens (epoch lags it by the pipeline)
    wire [CODE_AW-1:0] chip_idx = chip_q[CODE_AW+1:2];
    wire [CODE_AW-1:0] idx_prev  = (chip_idx == 0) ? CODE_LEN - 1 : chip_idx - 1;
    wire [CODE_AW-1:0] idx_next  = (chip_idx == CODE_LEN - 1) ? 0 : chip_idx + 1;
    wire [CODE_AW-1:0] idx_next2 = (chip_idx >= CODE_LEN - 2) ? chip_idx + 2 - CODE_LEN : chip_idx + 2;

    wire [32:0] code_sum = {1'b0, code_acc} + {1'b0, code_step_a};
    wire tick     = code_sum[32];
    wire chip_inc = tick && (chip_q[1:0] == 2'b11);
    wire wrap     = tick && (chip_q == N_Q - 1);
    wire v0       = sample_valid && enable;

    // c_prev / c_cur / c_next hold the chips at chip_idx - 1, chip_idx, chip_idx + 1. While
    // enabled, a chip increment shifts them and prefetches the new next chip; while disabled
    // all three are re-read in rotation (reads issued while disabled are dropped if they
    // land after enable, so a stale value never overwrites a shifted one).
    reg c_prev, c_cur, c_next;
    reg [1:0] refill;
    reg rd_pend, rd_pend_d, rd_rot, rd_rot_d;
    reg [1:0] rd_tag, rd_tag_d;
    always @(posedge clk) begin
        rd_pend   <= 1'b0;
        rd_pend_d <= rd_pend;
        rd_rot_d  <= rd_rot;
        rd_tag_d  <= rd_tag;
        if (rst) begin
            refill <= 2'd0;
            rd_pend_d <= 1'b0;
        end else begin
            if (!enable) begin
                rd_pend <= 1'b1;
                rd_rot  <= 1'b1;
                rd_tag  <= refill;
                rd_chip <= (refill == 2'd0) ? idx_prev : (refill == 2'd1) ? chip_idx : idx_next;
                refill  <= (refill == 2'd2) ? 2'd0 : refill + 2'd1;
            end else if (v0 && chip_inc) begin
                c_prev  <= c_cur;
                c_cur   <= c_next;
                rd_pend <= 1'b1;
                rd_rot  <= 1'b0;
                rd_tag  <= 2'd2;
                rd_chip <= idx_next2;
            end
            if (rd_pend_d && !(rd_rot_d && enable)) begin
                case (rd_tag_d)
                    2'd0: c_prev <= rd_data;
                    2'd1: c_cur  <= rd_data;
                    default: c_next <= rd_data;
                endcase
            end
        end
    end

    // ---- stage 0: NCOs and tap selection ---------------------------------------------
    wire [1:0] sp = (spacing_q == 2'd2) ? 2'd2 : 2'd1;
    wire [2:0] qf_plus  = {1'b0, chip_q[1:0]} + {1'b0, sp};
    wire tap_e = (qf_plus > 3'd3) ? c_next : c_cur;
    wire tap_l = (chip_q[1:0] < sp) ? c_prev : c_cur;

    reg        s1_v, s1_wrap;
    reg [2:0]  s1_ph;
    reg [1:0]  s1_sample;
    reg        s1_e, s1_p, s1_l;
    always @(posedge clk) begin
        s1_v    <= 1'b0;
        s1_wrap <= 1'b0;
        if (rst) begin
            carr_acc <= 32'd0; code_acc <= 32'd0; chip_q <= {(CODE_AW+2){1'b0}};
            carr_step_a <= 32'd0; code_step_a <= 32'd0; epoch0 <= 32'd0;
        end else begin
            if (!enable) begin
                carr_step_a <= carr_step;
                code_step_a <= code_step;
            end
            if (load_phase && !enable) begin
                carr_acc <= carr_phase_in;
                code_acc <= code_frac_in;
                chip_q   <= chip_q_in;
            end else if (v0) begin
                s1_v      <= 1'b1;
                s1_wrap   <= wrap;
                s1_ph     <= carr_acc[31:29];
                s1_sample <= sample;
                s1_e      <= tap_e;
                s1_p      <= c_cur;
                s1_l      <= tap_l;
                carr_acc  <= carr_acc + carr_step_a;
                code_acc  <= code_sum[31:0];
                chip_q    <= wrap ? {(CODE_AW+2){1'b0}} : chip_q + {{(CODE_AW+1){1'b0}}, tick};
                if (wrap) begin
                    carr_step_a <= carr_step;
                    code_step_a <= code_step;
                    epoch0      <= epoch0 + 32'd1;
                end
            end
        end
    end

    // ---- stage 1: 3-level LO and mixer ------------------------------------------------
    // LO phase p covers the octant centered on (p + 0.5) * 45 degrees; cos/sin in {+/-1, +/-2}.
    reg lo_c_neg, lo_c_big, lo_s_neg, lo_s_big;
    always @* case (s1_ph)                                                         //  cos sin
        3'd0:    begin lo_c_neg = 0; lo_c_big = 1; lo_s_neg = 0; lo_s_big = 0; end //   2   1
        3'd1:    begin lo_c_neg = 0; lo_c_big = 0; lo_s_neg = 0; lo_s_big = 1; end //   1   2
        3'd2:    begin lo_c_neg = 1; lo_c_big = 0; lo_s_neg = 0; lo_s_big = 1; end //  -1   2
        3'd3:    begin lo_c_neg = 1; lo_c_big = 1; lo_s_neg = 0; lo_s_big = 0; end //  -2   1
        3'd4:    begin lo_c_neg = 1; lo_c_big = 1; lo_s_neg = 1; lo_s_big = 0; end //  -2  -1
        3'd5:    begin lo_c_neg = 1; lo_c_big = 0; lo_s_neg = 1; lo_s_big = 1; end //  -1  -2
        3'd6:    begin lo_c_neg = 0; lo_c_big = 0; lo_s_neg = 1; lo_s_big = 1; end //   1  -2
        default: begin lo_c_neg = 0; lo_c_big = 1; lo_s_neg = 1; lo_s_big = 0; end //   2  -1
    endcase

    // product of a {+/-1, +/-3} sample and a {+/-1, +/-2} LO value: +/-1, 2, 3 or 6
    function signed [4:0] prod(input s_neg, input s_big, input l_neg, input l_big);
        reg [2:0] m;
        begin
            m = s_big ? (l_big ? 3'd6 : 3'd3) : (l_big ? 3'd2 : 3'd1);
            prod = (s_neg ^ l_neg) ? -$signed({2'b0, m}) : $signed({2'b0, m});
        end
    endfunction

    reg s2_v, s2_wrap, s2_e, s2_p, s2_l;
    reg signed [4:0] mix_i, mix_q;
    always @(posedge clk) begin
        s2_v    <= s1_v && !rst;
        s2_wrap <= s1_wrap;
        s2_e <= s1_e; s2_p <= s1_p; s2_l <= s1_l;
        mix_i <=  prod(s1_sample[1], s1_sample[0], lo_c_neg, lo_c_big);
        mix_q <= -prod(s1_sample[1], s1_sample[0], lo_s_neg, lo_s_big);  // replica exp(-j phase)
    end

    // ---- stage 2: correlate and dump ------------------------------------------------
    reg signed [ACC_W-1:0] a_ie, a_qe, a_ip, a_qp, a_il, a_ql;
    wire signed [ACC_W-1:0] mi = {{(ACC_W-5){mix_i[4]}}, mix_i};
    wire signed [ACC_W-1:0] mq = {{(ACC_W-5){mix_q[4]}}, mix_q};
    wire signed [ACC_W-1:0] n_ie = a_ie + (s2_e ? -mi : mi);
    wire signed [ACC_W-1:0] n_qe = a_qe + (s2_e ? -mq : mq);
    wire signed [ACC_W-1:0] n_ip = a_ip + (s2_p ? -mi : mi);
    wire signed [ACC_W-1:0] n_qp = a_qp + (s2_p ? -mq : mq);
    wire signed [ACC_W-1:0] n_il = a_il + (s2_l ? -mi : mi);
    wire signed [ACC_W-1:0] n_ql = a_ql + (s2_l ? -mq : mq);

    always @(posedge clk) begin
        dump <= 1'b0;
        if (rst) begin
            a_ie <= 0; a_qe <= 0; a_ip <= 0; a_qp <= 0; a_il <= 0; a_ql <= 0;
            ie <= 0; qe <= 0; ip <= 0; qp <= 0; il <= 0; ql <= 0;
            epoch <= 32'd0;
        end else if (s2_v) begin
            if (s2_wrap) begin
                ie <= n_ie; qe <= n_qe; ip <= n_ip; qp <= n_qp; il <= n_il; ql <= n_ql;
                a_ie <= 0; a_qe <= 0; a_ip <= 0; a_qp <= 0; a_il <= 0; a_ql <= 0;
                dump  <= 1'b1;
                epoch <= epoch + 32'd1;
            end else begin
                a_ie <= n_ie; a_qe <= n_qe; a_ip <= n_ip; a_qp <= n_qp; a_il <= n_il; a_ql <= n_ql;
            end
        end
    end

    // ---- measurement latch: the state before the strobed sample ----------------------
    reg [31:0] meas_carr_r, meas_frac_r, meas_epoch_r;
    reg [CODE_AW+1:0] meas_chip_r;
    always @(posedge clk) begin
        if (rst) begin
            meas_carr_r <= 32'd0; meas_frac_r <= 32'd0; meas_epoch_r <= 32'd0;
            meas_chip_r <= {(CODE_AW+2){1'b0}};
        end else if (meas_strobe) begin
            meas_carr_r  <= carr_acc;
            meas_frac_r  <= code_acc;
            meas_chip_r  <= chip_q;
            meas_epoch_r <= epoch0;
        end
    end
    assign meas_carr_phase = meas_carr_r;
    assign meas_code_frac  = meas_frac_r;
    assign meas_epoch      = meas_epoch_r;
    assign meas_chip_q     = meas_chip_r;
endmodule
