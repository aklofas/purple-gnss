`timescale 1ns/1ps
// LIFCL-40-EVN top level: rx_spi behind the board's Raspberry Pi and PMOD headers.
//
// Clocking. The board's only fabric clock input is a 12 MHz oscillator, below the GPLL's
// 18 MHz minimum reference (FPGA-DS-02049, fIN), so the fabric runs from the device's
// internal oscillator: 450 MHz / 12 = 37.5 MHz nominal, 34.9 to 40.1 MHz over the oscillator's
// specified range. The receiver does not depend on
// the clock rate (the NCO step words absorb it and the SPI link is asynchronous). A GPLL
// referenced from the same oscillator runs alongside, locked at a 1125 MHz VCO with a
// 112.5 MHz output, purely to check that a bitstream from the open toolchain configures a
// PLL that locks: its LOCK drives an LED and STATUS bit 0, its output a second heartbeat LED.
//
// LEDs are active low: 0 PLL lock, 1 fabric heartbeat (about 1.1 Hz), 2 dump pending
// (stretched), 3 SPI chip select active (stretched), 4 PLL heartbeat (about 1.7 Hz).
// SW2 is reset.
module top #(
    parameter N_CHANNELS = 1
) (
    input  wire       btn_rst_n,
    input  wire       spi_sck,
    input  wire       spi_mosi,
    input  wire       spi_cs_n,
    output wire       spi_miso,
    output wire       dump_irq,
    input  wire       sample_sign,
    input  wire       sample_mag,
    output wire [4:0] led_n
);
    // ---- clocks -------------------------------------------------------------------------
    wire clk;
    OSCA #(
        .HF_CLK_DIV("11"),          // 450 MHz / (11 + 1)
        .HF_OSC_EN("ENABLED"),
        .LF_OUTPUT_EN("DISABLED")
    ) osc (
        .HFOUTEN(1'b1), .HFSDSCEN(1'b0),
        .HFCLKOUT(clk), .LFCLKOUT(), .HFCLKCFG(), .HFSDCOUT()
    );

    // ---- reset: power-on counter, then SW2 -------------------------------------------------
    reg [15:0] por_cnt = 16'd0;
    reg        por_done = 1'b0;
    reg [1:0]  btn_q = 2'b11;
    always @(posedge clk) begin
        if (!por_done) begin
            por_cnt <= por_cnt + 16'd1;
            if (&por_cnt) por_done <= 1'b1;
        end
        btn_q <= {btn_q[0], btn_rst_n};
    end
    wire rst = !por_done || !btn_q[1];

    // PLL settings as found on the board (README, "PLL settings"). PLLPD_N is the PLL power
    // bit (TN-02095 lmmi_pllpd_n: 0 = PLL not used) and the toolchain's default, UNUSED, leaves
    // the PLL powered down whatever else is set. With the primitive's FBK_MASK default of 8 the
    // feedback divides by 8, not by FBK_MMD_DIG = 3 (TN-02095 Table C.3); the PLL still locked
    // that way, with the heartbeat visibly faster (rate not measured). The analog set is the
    // 1.0 V-core one LiteX's NXPLL uses (the primitive defaults to the 0.9 V values); the PLL
    // also locked with the defaults, so the set is for the core voltage, not for lock.
    wire pll_clkop, pll_lock;
    PLL #(
        .REF_MMD_DIG("1"), .FBK_MMD_DIG("3"),            // 37.5 MHz x 3 = 112.5 MHz on CLKOP
        .FBK_MASK("0b00000000"), .FBK_MMD_PULS_CTL("0b0001"),
        .FBK_INTEGER_MODE("ENABLED"), .REF_INTEGER_MODE("ENABLED"),
        .CLKMUX_FB("CMUX_CLKOP"), .SEL_FBK("DIVA"),
        .V2I_KVCO_SEL("60"),                              // explicit: nextpnr drops the PLL's FASM otherwise
        .KP_VCO("0b00011"), .V2I_1V_EN("ENABLED"),
        .PLLRESET_ENA("ENABLED"), .PLLPD_N("USED"),
        .DIVA("9"), .DELA("0"), .ENCLK_CLKOP("ENABLED")   // VCO = 112.5 MHz x (9 + 1) = 1125 MHz
    ) pll (
        .PLLPOWERDOWN_N(1'b1),
        .REFCK(clk),
        .CLKOP(pll_clkop),
        .ENCLKOP(1'b1),
        .LOCK(pll_lock),
        .PLLRESET(!por_done)
    );
    reg [25:0] pll_hb = 26'd0;
    always @(posedge pll_clkop) pll_hb <= pll_hb + 26'd1;
    reg [1:0] lock_q = 2'b00;  // LOCK into the fabric clock domain
    always @(posedge clk) lock_q <= {lock_q[0], pll_lock};

    // ---- receiver ----------------------------------------------------------------------------
    rx_spi #(.N_CHANNELS(N_CHANNELS)) rx (
        .clk(clk), .rst(rst),
        .spi_sck(spi_sck), .spi_mosi(spi_mosi), .spi_cs_n(spi_cs_n), .spi_miso(spi_miso),
        .pin_valid(1'b1), .pin_sample({sample_sign, sample_mag}),
        .pll_lock(lock_q[1]), .dump_irq(dump_irq)
    );

    // ---- LEDs -----------------------------------------------------------------------------------
    reg [24:0] hb = 25'd0;
    always @(posedge clk) hb <= hb + 25'd1;

    // activity indicators held about 56 ms past the last event (2^21 clocks at 37.5 MHz)
    reg [20:0] dump_hold = 21'd0, spi_hold = 21'd0;
    always @(posedge clk) begin
        if (dump_irq) dump_hold <= {21{1'b1}};
        else if (dump_hold != 21'd0) dump_hold <= dump_hold - 21'd1;
        if (!spi_cs_n) spi_hold <= {21{1'b1}};
        else if (spi_hold != 21'd0) spi_hold <= spi_hold - 21'd1;
    end

    assign led_n = ~{pll_hb[25], spi_hold != 21'd0, dump_hold != 21'd0, hb[24], lock_q[1]};
endmodule
