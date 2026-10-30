`timescale 1ns/1ps
// Single-lane SPI slave (mode 0, MSB first, CS active low) driving the register bus.
//
// Transaction: command byte (bit 7 = write, bit 6 = auto-increment the word address),
// two address bytes, then 32-bit words. A write commits each word to the bus as its last
// bit arrives; a read inserts one dummy byte after the address while the first word is
// fetched, and fetches each following word while the previous one is shifting out. CS
// rising ends the transaction and drops any partial word.
//
// SCK, MOSI and CS are oversampled through two-flop synchronizers; MISO takes its next bit
// right after a detected SCK rising edge, so it is stable long before the master's next
// sample. In fabric clocks the slave needs: SCK high and low at least 2 each; MOSI held at
// least 2 after the SCK rise; CS low at least 2 before the first SCK rise and at least 2 after
// the last (a CS rise within one clock of the last SCK rise drops that word); CS high at least
// 2 between transactions. SCK at the fabric clock divided by 8 with equal high and low times
// meets all of these.
module spi_slave (
    input  wire        clk,
    input  wire        rst,
    input  wire        sck,
    input  wire        mosi,
    input  wire        cs_n,
    output reg         miso,
    output reg  [15:0] bus_addr,
    output reg  [31:0] bus_wdata,
    output reg         bus_we,
    output reg         bus_re,
    input  wire [31:0] bus_rdata
);
    // ---- synchronizers and edge detection ----
    reg [1:0] sck_q, mosi_q, cs_q;
    reg sck_q3;
    always @(posedge clk) begin
        sck_q  <= {sck_q[0], sck};
        mosi_q <= {mosi_q[0], mosi};
        cs_q   <= {cs_q[0], cs_n};
        sck_q3 <= sck_q[1];
    end
    wire rise     = sck_q[1] && !sck_q3;
    wire selected = !cs_q[1];
    wire mosi_s   = mosi_q[1];

    // ---- transaction state ----
    reg [7:0]  nbit;        // rising edges seen in the command, address and dummy bytes (at most 32)
    reg        is_write, incr;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [31:0] shift;       // bits shifted in (command/address/write data) or out (read data);
                            // on reads bit 31 is the bit already on MISO
    /* verilator lint_on UNUSEDSIGNAL */
    reg [31:0] rd_word;
    reg        rd_pend;
    reg [4:0]  wbit;        // bit index within the current data word
    reg        in_data;     // past command, address (and dummy for reads)

    wire [31:0] shift_in = {shift[30:0], mosi_s};

    always @(posedge clk) begin
        bus_we  <= 1'b0;
        bus_re  <= 1'b0;
        rd_pend <= bus_re;
        if (rd_pend) rd_word <= bus_rdata;
        if (bus_we) bus_addr <= bus_addr + {15'd0, incr};  // after the word has landed

        if (rst || !selected) begin
            nbit <= 8'd0; wbit <= 5'd0; in_data <= 1'b0; miso <= 1'b0;
            is_write <= 1'b0; incr <= 1'b0;
        end else if (rise) begin
            if (!in_data) begin
                shift <= shift_in;
                nbit  <= nbit + 8'd1;
                if (nbit == 8'd7) begin
                    is_write <= shift_in[7];
                    incr     <= shift_in[6];
                end
                if (nbit == 8'd23) begin
                    bus_addr <= shift_in[15:0];
                    if (is_write) begin
                        in_data <= 1'b1;
                        wbit    <= 5'd0;
                    end else begin
                        bus_re <= 1'b1;    // first word fetched during the dummy byte
                    end
                end
                if (nbit == 8'd31 && !is_write) begin
                    // last dummy bit: present the first data bit
                    shift   <= rd_word;
                    miso    <= rd_word[31];
                    in_data <= 1'b1;
                    wbit    <= 5'd0;
                end
            end else if (is_write) begin
                shift <= shift_in;
                wbit  <= wbit + 5'd1;
                if (wbit == 5'd31) begin
                    bus_wdata <= shift_in;
                    bus_we    <= 1'b1;
                end
            end else begin
                // read: the master just sampled shift[31]; present the next bit
                wbit <= wbit + 5'd1;
                if (wbit == 5'd23) begin
                    bus_addr <= bus_addr + {15'd0, incr};
                    bus_re   <= 1'b1;    // next word, ready long before it is needed
                end
                if (wbit == 5'd31) begin
                    shift <= rd_word;
                    miso  <= rd_word[31];
                end else begin
                    shift <= {shift[30:0], 1'b0};
                    miso  <= shift[30];
                end
            end
        end
    end
endmodule
