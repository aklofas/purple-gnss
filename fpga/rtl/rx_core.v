`timescale 1ns/1ps
// The receiver core: N tracking channels behind a 16-bit-address / 32-bit-data register
// bus, with the global registers, the sample source mux (front-end pins or the playback
// RAM) and a 48-bit count of the valid samples. The register map is documented in
// fpga/README.md and mirrored by fpgamodel.regmap.
//
// Bus: a write lands on the clock edge where bus_we is high; a read presents bus_rdata on
// the edge after bus_re. Unmapped addresses read zero and ignore writes. Up to 16 channels
// at 0x1000 + k * 0x100.
module rx_core #(
    parameter N_CHANNELS = 1,
    parameter CODE_LEN   = 1023,
    parameter CODE_AW    = 10,
    parameter ACC_W      = 20,
    parameter PB_DEPTH   = 147456
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        pin_valid,
    input  wire [1:0]  pin_sample,
    input  wire        pll_lock,
    input  wire [15:0] bus_addr,
    input  wire [31:0] bus_wdata,
    input  wire        bus_we,
    input  wire        bus_re,
    output reg  [31:0] bus_rdata,
    output wire        dump_irq
);
    localparam PB_AW = $clog2(PB_DEPTH / 16);
    localparam [31:0] ID_VALUE = 32'h50474E53;
    localparam [31:0] VERSION  = 32'h00000100;

    // register offsets (fpgamodel.regmap)
    localparam [11:0] A_ID = 12'h000, A_VERSION = 12'h001, A_CTRL = 12'h002, A_STATUS = 12'h003,
                      A_CNT_LO = 12'h004, A_CNT_HI = 12'h005, A_PENDING = 12'h006, A_CLEAR = 12'h007,
                      A_MEAS_TRIG = 12'h008, A_MEAS_PERIOD = 12'h009, A_NCH = 12'h00A,
                      A_PB_CTRL = 12'h00B, A_PB_LEN = 12'h00C, A_PB_DEPTH = 12'h00D,
                      A_PB_ADDR = 12'h100, A_PB_DATA = 12'h101;
    localparam [7:0] C_CTRL = 8'h00, C_START_LO = 8'h01, C_START_HI = 8'h02, C_CARR_STEP = 8'h03,
                     C_CODE_STEP = 8'h04, C_CARR_PHASE = 8'h05, C_CODE_FRAC = 8'h06, C_CODE_CHIP = 8'h07,
                     C_SPACING = 8'h08, C_CODE_WR_ADDR = 8'h09, C_CODE_WR_DATA = 8'h0A, C_EPOCH = 8'h0B,
                     C_MISSED = 8'h0C, C_IE = 8'h10, C_QE = 8'h11, C_IP = 8'h12, C_QP = 8'h13,
                     C_IL = 8'h14, C_QL = 8'h15;

    // ---- decode ----------------------------------------------------------------------
    wire        glob     = bus_addr[15:12] == 4'h0;
    wire        chsel    = bus_addr[15:12] == 4'h1;
    wire [3:0]  ch_k     = bus_addr[11:8];
    wire [7:0]  ch_off   = bus_addr[7:0];
    wire [11:0] g_off    = bus_addr[11:0];
    wire        ch_valid = chsel && ({28'd0, ch_k} < N_CHANNELS);
    wire        wr_g     = bus_we && glob;
    wire        wr_c     = bus_we && ch_valid;

    // ---- global state --------------------------------------------------------------------
    reg soft_rst;
    wire rst_i = rst | soft_rst;
    reg g_enable, source;
    reg [31:0] meas_period, pb_len, pb_addr;
    reg pb_start, pb_we;
    reg [31:0] pb_wdata;
    reg [PB_AW-1:0] pb_waddr;
    wire pb_running, pb_done, pb_valid;
    wire [1:0] pb_sample;

    always @(posedge clk) begin
        soft_rst <= wr_g && (g_off == A_CTRL) && bus_wdata[1];
        pb_start <= 1'b0;
        pb_we    <= 1'b0;
        if (rst_i) begin
            g_enable <= 1'b0; source <= 1'b0;
            meas_period <= 32'd0; pb_len <= 32'd0; pb_addr <= 32'd0;
        end else if (wr_g) begin
            case (g_off)
                A_CTRL:        begin g_enable <= bus_wdata[0]; source <= bus_wdata[2]; end
                A_MEAS_PERIOD: meas_period <= bus_wdata;
                A_PB_CTRL:     pb_start <= bus_wdata[0];
                A_PB_LEN:      pb_len <= bus_wdata;
                A_PB_ADDR:     pb_addr <= bus_wdata;
                A_PB_DATA:     begin
                    pb_we <= 1'b1; pb_wdata <= bus_wdata; pb_waddr <= pb_addr[PB_AW-1:0];
                    pb_addr <= pb_addr + 32'd1;
                end
                A_MEAS_TRIG:   ;  // measurement latch not built yet
                default: ;
            endcase
        end
    end

    playback #(.PB_DEPTH(PB_DEPTH)) pb (
        .clk(clk), .rst(rst_i),
        .wr_we(pb_we), .wr_addr(pb_waddr), .wr_data(pb_wdata),
        .start(pb_start), .len(pb_len),
        .out_valid(pb_valid), .out_sample(pb_sample), .running(pb_running), .done(pb_done)
    );

    // ---- sample source and count ---------------------------------------------------------
    wire       src_valid  = g_enable && (source ? pb_valid : pin_valid);
    wire [1:0] src_sample = source ? pb_sample : pin_sample;
    reg [47:0] sample_cnt;
    reg [15:0] cnt_hi_latch;
    always @(posedge clk) begin
        if (rst_i) sample_cnt <= 48'd0;
        else if (src_valid) sample_cnt <= sample_cnt + 48'd1;
        if (bus_re && glob && (g_off == A_CNT_LO)) cnt_hi_latch <= sample_cnt[47:32];
    end

    // ---- channels ------------------------------------------------------------------------
    reg        ch_en        [0:N_CHANNELS-1];
    reg [31:0] ch_carr_step [0:N_CHANNELS-1];
    reg [31:0] ch_code_step [0:N_CHANNELS-1];
    reg [31:0] ch_carr_ph   [0:N_CHANNELS-1];
    reg [31:0] ch_code_frac [0:N_CHANNELS-1];
    reg [CODE_AW+1:0] ch_chip_q [0:N_CHANNELS-1];
    reg [1:0]  ch_spacing   [0:N_CHANNELS-1];
    reg [31:0] ch_code_addr [0:N_CHANNELS-1];
    reg        ch_load      [0:N_CHANNELS-1];
    reg        ch_code_we   [0:N_CHANNELS-1];
    reg [31:0] ch_code_wdata[0:N_CHANNELS-1];
    reg [CODE_AW-6:0] ch_code_waddr [0:N_CHANNELS-1];
    reg        pending      [0:N_CHANNELS-1];
    reg [31:0] missed       [0:N_CHANNELS-1];

    wire [N_CHANNELS-1:0]       ch_dump;
    wire [32*N_CHANNELS-1:0]    ch_epoch;
    wire [ACC_W*N_CHANNELS-1:0] ch_ie, ch_qe, ch_ip, ch_qp, ch_il, ch_ql;
    reg  [N_CHANNELS-1:0]       pending_vec;

    genvar k;
    generate for (k = 0; k < N_CHANNELS; k = k + 1) begin : g
        wire sel = wr_c && (ch_k == k);
        always @(posedge clk) begin
            ch_load[k]    <= 1'b0;
            ch_code_we[k] <= 1'b0;
            if (rst_i) begin
                ch_en[k] <= 1'b0; ch_carr_step[k] <= 32'd0; ch_code_step[k] <= 32'd0;
                ch_carr_ph[k] <= 32'd0; ch_code_frac[k] <= 32'd0; ch_chip_q[k] <= {(CODE_AW+2){1'b0}};
                ch_spacing[k] <= 2'd1; ch_code_addr[k] <= 32'd0;
            end else if (sel) begin
                case (ch_off)
                    C_CTRL:      ch_en[k] <= bus_wdata[0];
                    C_CARR_STEP: ch_carr_step[k] <= bus_wdata;
                    C_CODE_STEP: ch_code_step[k] <= bus_wdata;
                    C_CARR_PHASE: if (!ch_en[k]) begin ch_carr_ph[k] <= bus_wdata; ch_load[k] <= 1'b1; end
                    C_CODE_FRAC:  if (!ch_en[k]) begin ch_code_frac[k] <= bus_wdata; ch_load[k] <= 1'b1; end
                    C_CODE_CHIP:  if (!ch_en[k]) begin ch_chip_q[k] <= bus_wdata[CODE_AW+1:0]; ch_load[k] <= 1'b1; end
                    C_SPACING:   ch_spacing[k] <= bus_wdata[1:0];
                    C_START_LO, C_START_HI: ;  // placeholders until a live sample source exists
                    C_CODE_WR_ADDR: ch_code_addr[k] <= bus_wdata;
                    C_CODE_WR_DATA: begin
                        ch_code_we[k] <= 1'b1; ch_code_wdata[k] <= bus_wdata;
                        ch_code_waddr[k] <= ch_code_addr[k][CODE_AW-6:0];
                        ch_code_addr[k] <= ch_code_addr[k] + 32'd1;
                    end
                    default: ;
                endcase
            end
        end

        // a dump sets the pending bit (and counts a miss if it was still set); the clear
        // register clears it; a dump in the same clock as a clear wins
        always @(posedge clk) begin
            if (rst_i) begin pending[k] <= 1'b0; missed[k] <= 32'd0; end
            else if (ch_dump[k]) begin
                pending[k] <= 1'b1;
                if (pending[k]) missed[k] <= missed[k] + 32'd1;
            end else if (wr_g && (g_off == A_CLEAR) && bus_wdata[k]) pending[k] <= 1'b0;
        end
        always @* pending_vec[k] = pending[k];

        channel #(.CODE_LEN(CODE_LEN), .CODE_AW(CODE_AW), .ACC_W(ACC_W)) ch (
            .clk(clk), .rst(rst_i),
            .sample_valid(src_valid), .sample(src_sample),
            .enable(ch_en[k] && g_enable),
            .carr_step(ch_carr_step[k]), .code_step(ch_code_step[k]), .spacing_q(ch_spacing[k]),
            .load_phase(ch_load[k]), .carr_phase_in(ch_carr_ph[k]), .code_frac_in(ch_code_frac[k]),
            .chip_q_in(ch_chip_q[k]),
            .code_we(ch_code_we[k]), .code_waddr(ch_code_waddr[k]), .code_wdata(ch_code_wdata[k]),
            .meas_strobe(1'b0),
            /* verilator lint_off PINCONNECTEMPTY */
            .meas_carr_phase(), .meas_code_frac(), .meas_epoch(), .meas_chip_q(),
            /* verilator lint_on PINCONNECTEMPTY */
            .dump(ch_dump[k]),
            .ie(ch_ie[k*ACC_W +: ACC_W]), .qe(ch_qe[k*ACC_W +: ACC_W]),
            .ip(ch_ip[k*ACC_W +: ACC_W]), .qp(ch_qp[k*ACC_W +: ACC_W]),
            .il(ch_il[k*ACC_W +: ACC_W]), .ql(ch_ql[k*ACC_W +: ACC_W]),
            .epoch(ch_epoch[k*32 +: 32])
        );
    end endgenerate

    assign dump_irq = |pending_vec;

    // ---- readback ---------------------------------------------------------------------------
    function [31:0] sext;
        input [ACC_W-1:0] v;
        sext = {{(32-ACC_W){v[ACC_W-1]}}, v};
    endfunction
    localparam KW = (N_CHANNELS > 1) ? $clog2(N_CHANNELS) : 1;
    wire [KW-1:0] rk = ch_valid ? ch_k[KW-1:0] : {KW{1'b0}};

    always @(posedge clk) begin
        if (bus_re) begin
            bus_rdata <= 32'd0;
            if (glob) begin
                case (g_off)
                    A_ID:          bus_rdata <= ID_VALUE;
                    A_VERSION:     bus_rdata <= VERSION;
                    A_CTRL:        bus_rdata <= {29'd0, source, 1'b0, g_enable};
                    A_STATUS:      bus_rdata <= {29'd0, pb_done, pb_running, pll_lock};
                    A_CNT_LO:      bus_rdata <= sample_cnt[31:0];
                    A_CNT_HI:      bus_rdata <= {16'd0, cnt_hi_latch};
                    A_PENDING:     bus_rdata <= {{(32-N_CHANNELS){1'b0}}, pending_vec};
                    A_MEAS_PERIOD: bus_rdata <= meas_period;
                    A_NCH:         bus_rdata <= N_CHANNELS;
                    A_PB_LEN:      bus_rdata <= pb_len;
                    A_PB_DEPTH:    bus_rdata <= PB_DEPTH;
                    A_PB_ADDR:     bus_rdata <= pb_addr;
                    default: ;
                endcase
            end else if (ch_valid) begin
                case (ch_off)
                    C_CTRL:         bus_rdata <= {31'd0, ch_en[rk]};
                    C_CARR_STEP:    bus_rdata <= ch_carr_step[rk];
                    C_CODE_STEP:    bus_rdata <= ch_code_step[rk];
                    C_CARR_PHASE:   bus_rdata <= ch_carr_ph[rk];
                    C_CODE_FRAC:    bus_rdata <= ch_code_frac[rk];
                    C_CODE_CHIP:    bus_rdata <= {{(30-CODE_AW){1'b0}}, ch_chip_q[rk]};
                    C_SPACING:      bus_rdata <= {30'd0, ch_spacing[rk]};
                    C_CODE_WR_ADDR: bus_rdata <= ch_code_addr[rk];
                    C_EPOCH:        bus_rdata <= ch_epoch[rk*32 +: 32];
                    C_MISSED:       bus_rdata <= missed[rk];
                    C_IE:           bus_rdata <= sext(ch_ie[rk*ACC_W +: ACC_W]);
                    C_QE:           bus_rdata <= sext(ch_qe[rk*ACC_W +: ACC_W]);
                    C_IP:           bus_rdata <= sext(ch_ip[rk*ACC_W +: ACC_W]);
                    C_QP:           bus_rdata <= sext(ch_qp[rk*ACC_W +: ACC_W]);
                    C_IL:           bus_rdata <= sext(ch_il[rk*ACC_W +: ACC_W]);
                    C_QL:           bus_rdata <= sext(ch_ql[rk*ACC_W +: ACC_W]);
                    default: ;   // CH_START_SAMPLE and MEAS_* read zero until built
                endcase
            end
        end
    end
endmodule
