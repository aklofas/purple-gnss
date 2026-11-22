`timescale 1ns/1ps
// SPI slave (mode 0, MSB first, CS active low) driving the register bus, clocked by SCK.
//
// Frame: a command byte and a 16-bit word address on lane 0, then 32-bit words, high byte
// first, for as long as CS stays low. Command bit 7 is write, bit 6 auto-increments the word
// address, bit 5 puts the data on all four lanes (high nibble first, lane 3 the most
// significant bit) instead of lane 0 in and lane 1 out, and bits 4:0 must be the signature
// 10110: a frame without it, as one shifted by a stray clock almost always is, is ignored
// and counted. A read has eight dummy clocks between the address and the data, in which
// nobody drives. The slave drives lane 1, or all four, only while it sends read data.
//
// The frame logic runs on SCK itself (inputs taken on the rising edge, outputs changed on
// the falling edge). A frame ends, and a partial word is dropped, when the fabric clock has
// seen CS high on two edges in a row; the frame logic is then held in reset until CS falls.
// A high pulse on CS shorter than a fabric clock period therefore does not end a frame:
// with several lanes switching at once on a marginal ground, such pulses happen. CS falling
// starts the next frame at once, and the lanes are released by CS itself, not by the reset. Two small FIFOs cross to the fabric clock. Requests carry the frame's command and
// address, then each write word; the fabric side issues the writes in order. For a read
// the fabric side fetches the word at the address as soon as the request arrives and keeps
// the response FIFO filled, the address advancing per word in auto-increment mode, so a
// read fetches up to three words past its last one: nothing on the bus may react to a read.
//
// Rates. The first read word has to cross within the eight dummy clocks: up to five fabric
// clocks for the request to cross and the bus to answer, then two SCK clocks for the
// answer to cross back, so six SCK periods must be longer than five fabric clock periods. A
// word that is not there in time is sent as zero, with every word after it in that frame,
// and the frame is counted as an underrun. A request finding its
// FIFO full (the fabric takes one per clock, so only a stalled fabric clock does this) is
// dropped with every later request of that frame, since a later word would land on an
// earlier one's address, and the frame is counted as an overrun.
//
// CS must stay high for at least five fabric clocks between frames, with SCK idle, so
// that every frame end is seen: after a shorter gap the next frame's bytes continue the frame
// before it (a write goes on writing), and nothing counts that.
//
// bad_frames, overruns and underruns count frames and saturate; clr zeroes them. rst resets
// both clock domains and must not come while a frame is in flight.
module spi_slave (
    input  wire        clk,
    input  wire        rst,
    input  wire        clr,
    input  wire        sck,
    input  wire        cs_n,
    input  wire [3:0]  io_i,
    output wire [3:0]  io_o,
    output wire [3:0]  io_oe,
    output reg  [15:0] bus_addr,
    output reg  [31:0] bus_wdata,
    output reg         bus_we,
    output reg         bus_re,
    input  wire [31:0] bus_rdata,
    output reg  [15:0] bad_frames,
    output reg  [7:0]  overruns,
    output reg  [7:0]  underruns
);
    // The frame end, as the fabric clock sees it: CS high on two consecutive edges behind a
    // synchronizer. The SCK domain's reset is that flop gated by the pin, so it holds from the
    // confirmed end until CS falls; rst resets both domains.
    /* verilator lint_off SYNCASYNCNET */
    reg [2:0] cs_q;
    reg       frame_end;
    always @(posedge clk) begin
        cs_q      <= {cs_q[1:0], cs_n};
        frame_end <= cs_q[1] && cs_q[2];
    end
    wire frame_rst = (frame_end && cs_n) | rst;

    wire        req_full, req_empty, resp_full, resp_empty;
    wire [32:0] req_rdata;
    wire [31:0] resp_rdata;

    // ================= SCK domain =================================================
    localparam [2:0] P_HEADER = 3'd0, P_DUMMY = 3'd1, P_READ = 3'd2, P_WRITE = 3'd3,
                     P_IGNORE = 3'd4;
    reg [2:0]  phase;
    reg [4:0]  cnt;       // clocks into the header, the dummy phase or the current word
    reg [30:0] sh;
    reg        c_read, c_incr, c_quad;
    reg        starved;   // a read word was not there: zeros until the frame ends
    reg        dropped;   // a request found its FIFO full: none after it until the frame ends
    reg [31:0] hold;      // the next word to send, taken over on the falling edge
    reg        load, drive;

    wire [7:0]  cmd_now   = {sh[6:0], io_i[0]};
    wire        sig_bad   = (phase == P_HEADER) && (cnt == 5'd7) && (cmd_now[4:0] != 5'b10110);
    wire [31:0] word_now  = c_quad ? {sh[27:0], io_i} : {sh[30:0], io_i[0]};
    wire        word_last = c_quad ? (cnt[2:0] == 3'd7) : (cnt == 5'd31);
    wire        hdr_done  = (phase == P_HEADER) && (cnt == 5'd23);
    wire        wr_word   = (phase == P_WRITE) && word_last;
    wire        rd_pop    = ((phase == P_DUMMY) && (cnt[2:0] == 3'd7))
                         || ((phase == P_READ) && word_last);

    // a request is written on the very edge that brings its last bit
    wire        req_we    = (hdr_done || wr_word) && !dropped;
    wire [32:0] req_wdata = hdr_done ? {1'b1, 14'd0, c_read, c_incr, sh[14:0], io_i[0]}
                                     : {1'b0, word_now};

    always @(posedge sck or posedge frame_rst) begin
        if (frame_rst) begin
            phase <= P_HEADER; cnt <= 5'd0;
            c_read <= 1'b0; c_incr <= 1'b0; c_quad <= 1'b0;
            starved <= 1'b0; dropped <= 1'b0; load <= 1'b0; drive <= 1'b0;
        end else begin
            load <= 1'b0;
            if (req_we && req_full) dropped <= 1'b1;
            case (phase)
                P_HEADER: begin
                    sh  <= {sh[29:0], io_i[0]};
                    cnt <= cnt + 5'd1;
                    if (cnt == 5'd7) begin
                        c_read <= !cmd_now[7]; c_incr <= cmd_now[6]; c_quad <= cmd_now[5];
                        if (sig_bad) phase <= P_IGNORE;
                    end
                    if (hdr_done) begin
                        phase <= c_read ? P_DUMMY : P_WRITE;
                        cnt   <= 5'd0;
                    end
                end
                P_WRITE: begin
                    sh  <= word_now[30:0];
                    cnt <= word_last ? 5'd0 : cnt + 5'd1;
                end
                P_DUMMY, P_READ: begin
                    cnt <= rd_pop ? 5'd0 : cnt + 5'd1;
                    if (rd_pop) begin
                        phase <= P_READ;
                        drive <= 1'b1;
                        load  <= 1'b1;
                        if (resp_empty || starved) begin
                            hold    <= 32'd0;
                            starved <= 1'b1;
                        end else begin
                            hold <= resp_rdata;
                        end
                    end
                end
                default: ;  // P_IGNORE: until CS rises
            endcase
        end
    end

    // read data out, changed on the falling edge
    reg [31:0] osr;
    reg        oe_r;
    always @(negedge sck or posedge frame_rst) begin
        if (frame_rst) begin
            oe_r <= 1'b0;
        end else begin
            oe_r <= drive;
            if (load) osr <= hold;
            else      osr <= c_quad ? {osr[27:0], 4'b0000} : {osr[30:0], 1'b0};
        end
    end
    assign io_o  = c_quad ? osr[31:28] : {2'b00, osr[31], 1'b0};
    assign io_oe = (!oe_r || cs_n) ? 4'b0000 : c_quad ? 4'b1111 : 4'b0010;

    // events, counted per frame in gray code for the fabric side to follow (exact for up to
    // 255 of a kind between two fabric clock edges); they outlive the frame
    reg [7:0] bad_bin, bad_gray, ovr_bin, ovr_gray, und_bin, und_gray;
    wire [7:0] bad_next = bad_bin + 8'd1;
    wire [7:0] ovr_next = ovr_bin + 8'd1;
    wire [7:0] und_next = und_bin + 8'd1;
    always @(posedge sck or posedge rst) begin
        if (rst) begin
            bad_bin <= 8'd0; bad_gray <= 8'd0; ovr_bin <= 8'd0; ovr_gray <= 8'd0;
            und_bin <= 8'd0; und_gray <= 8'd0;
        end else begin
            if (sig_bad) begin
                bad_bin <= bad_next; bad_gray <= (bad_next >> 1) ^ bad_next;
            end
            if (req_we && req_full) begin
                ovr_bin <= ovr_next; ovr_gray <= (ovr_next >> 1) ^ ovr_next;
            end
            if (rd_pop && resp_empty && !starved) begin
                und_bin <= und_next; und_gray <= (und_next >> 1) ^ und_next;
            end
        end
    end

    // ================= fabric domain ==============================================
    wire cs_high = frame_end;

    reg        rd_open, incr_r;
    reg [15:0] addr_r;
    reg [1:0]  fetch;     // 0 idle, 1 bus_re is out, 2 the word is on bus_rdata
    wire       req_re  = !req_empty;
    wire       resp_we = rd_open && (fetch == 2'd2) && !cs_high;

    always @(posedge clk) begin
        bus_we <= 1'b0;
        bus_re <= 1'b0;
        if (rst) begin
            rd_open <= 1'b0; incr_r <= 1'b0; addr_r <= 16'd0; fetch <= 2'd0;
        end else if (req_re) begin
            if (req_rdata[32]) begin          // a frame's command and address
                addr_r  <= req_rdata[15:0];
                incr_r  <= req_rdata[16];
                rd_open <= req_rdata[17];
                fetch   <= 2'd0;
                if (req_rdata[17]) begin      // a read: its first fetch goes out at once
                    bus_re   <= 1'b1;
                    bus_addr <= req_rdata[15:0];
                    fetch    <= 2'd1;
                end
            end else begin                    // a write word
                bus_we    <= 1'b1;
                bus_addr  <= addr_r;
                bus_wdata <= req_rdata[31:0];
                if (incr_r) addr_r <= addr_r + 16'd1;
            end
        end else if (cs_high) begin           // the frame is over: stop fetching (a read whose
                                              // request arrives after its end stops here too)
            rd_open <= 1'b0;
            fetch   <= 2'd0;
        end else if (rd_open) begin
            case (fetch)
                2'd0: if (!resp_full) begin
                    bus_re   <= 1'b1;
                    bus_addr <= addr_r;
                    fetch    <= 2'd1;
                end
                2'd1: fetch <= 2'd2;
                default: begin
                    fetch <= 2'd0;
                    if (incr_r) addr_r <= addr_r + 16'd1;
                end
            endcase
        end
    end

    cdc_fifo #(.WIDTH(33), .DEPTH_LOG2(2)) req (
        .wclk(sck), .wrst(rst), .we(req_we), .wdata(req_wdata), .full(req_full),
        .rclk(clk), .rrst(rst), .re(req_re), .rdata(req_rdata), .empty(req_empty)
    );
    // emptied between frames from both sides: words fetched for a read that ended are gone
    // before the next one starts
    wire resp_wrst = rst | cs_high;
    cdc_fifo #(.WIDTH(32), .DEPTH_LOG2(1)) resp (
        .wclk(clk), .wrst(resp_wrst), .we(resp_we), .wdata(bus_rdata), .full(resp_full),
        .rclk(sck), .rrst(frame_rst), .re(rd_pop && !starved), .rdata(resp_rdata),
        .empty(resp_empty)
    );

    // the event counts, followed from the SCK domain and added up with saturation
    function [7:0] gray_to_bin(input [7:0] g);
        gray_to_bin = g ^ (g >> 1) ^ (g >> 2) ^ (g >> 3) ^ (g >> 4) ^ (g >> 5) ^ (g >> 6)
                    ^ (g >> 7);
    endfunction
    reg [7:0] bad_s1, bad_s2, bad_seen, ovr_s1, ovr_s2, ovr_seen, und_s1, und_s2, und_seen;
    wire [7:0] bad_now = gray_to_bin(bad_s2);
    wire [7:0] bad_new = bad_now - bad_seen;
    wire [7:0] ovr_now = gray_to_bin(ovr_s2);
    wire [7:0] und_now = gray_to_bin(und_s2);
    wire [7:0] ovr_new = ovr_now - ovr_seen;
    wire [7:0] und_new = und_now - und_seen;
    wire [16:0] bad_sum = {1'b0, bad_frames} + {9'd0, bad_new};
    wire [8:0]  ovr_sum = {1'b0, overruns} + {1'b0, ovr_new};
    wire [8:0]  und_sum = {1'b0, underruns} + {1'b0, und_new};
    always @(posedge clk) begin
        bad_s1 <= bad_gray; bad_s2 <= bad_s1;
        ovr_s1 <= ovr_gray; ovr_s2 <= ovr_s1;
        und_s1 <= und_gray; und_s2 <= und_s1;
        if (rst) begin
            bad_seen <= 8'd0; ovr_seen <= 8'd0; und_seen <= 8'd0;
            bad_frames <= 16'd0; overruns <= 8'd0; underruns <= 8'd0;
        end else begin
            bad_seen <= bad_now;
            ovr_seen <= ovr_now;
            und_seen <= und_now;
            if (clr) begin
                bad_frames <= 16'd0; overruns <= 8'd0; underruns <= 8'd0;
            end else begin
                bad_frames <= bad_sum[16] ? 16'hFFFF : bad_sum[15:0];
                overruns   <= ovr_sum[8] ? 8'hFF : ovr_sum[7:0];
                underruns  <= und_sum[8] ? 8'hFF : und_sum[7:0];
            end
        end
    end
    /* verilator lint_on SYNCASYNCNET */
endmodule
