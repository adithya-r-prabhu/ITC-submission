`default_nettype none
`timescale 1ns / 1ps






module cw305_ibex_top #(
    parameter pBYTECNT_SIZE = 7,
    parameter pADDR_WIDTH   = 21
)(

    input  wire                          usb_clk,
    inout  wire [7:0]                    usb_data,
    input  wire [pADDR_WIDTH-1:0]        usb_addr,
    input  wire                          usb_rdn,
    input  wire                          usb_wrn,
    input  wire                          usb_cen,
    input  wire                          usb_trigger,


    input  wire                          j16_sel,
    input  wire                          k16_sel,
    input  wire                          k15_sel,
    input  wire                          l14_sel,
    input  wire                          pushbutton,
    output wire                          led1,
    output wire                          led2,
    output wire                          led3,


    input  wire                          pll_clk1,


    output wire                          tio_trigger,
    output wire                          tio_clkout,
    input  wire                          tio_clkin
);

    wire usb_clk_buf;
    wire [7:0] usb_dout;
    wire isout;
    assign usb_data = isout ? usb_dout : 8'bZ;

    wire [pADDR_WIDTH-pBYTECNT_SIZE-1:0] reg_address;
    wire [pBYTECNT_SIZE-1:0] reg_bytecnt;
    wire reg_addrvalid;
    wire [7:0] write_data;
    wire [7:0] read_data;
    wire reg_read;
    wire reg_write;
    wire [4:0] clk_settings;
    wire crypt_clk;

    wire resetn = pushbutton;
    wire reset  = !resetn;


    reg [24:0] usb_timer_heartbeat;
    always @(posedge usb_clk_buf) usb_timer_heartbeat <= usb_timer_heartbeat + 25'd1;
    assign led1 = usb_timer_heartbeat[24];
    reg [22:0] crypt_clk_heartbeat;
    always @(posedge crypt_clk) crypt_clk_heartbeat <= crypt_clk_heartbeat + 23'd1;
    assign led2 = crypt_clk_heartbeat[22];

    cw305_usb_reg_fe #(
        .pBYTECNT_SIZE (pBYTECNT_SIZE),
        .pADDR_WIDTH   (pADDR_WIDTH)
    ) U_usb_reg_fe (
        .rst           (reset),
        .usb_clk       (usb_clk_buf),
        .usb_din       (usb_data),
        .usb_dout      (usb_dout),
        .usb_rdn       (usb_rdn),
        .usb_wrn       (usb_wrn),
        .usb_cen       (usb_cen),
        .usb_alen      (1'b0),
        .usb_addr      (usb_addr),
        .usb_isout     (isout),
        .reg_address   (reg_address),
        .reg_bytecnt   (reg_bytecnt),
        .reg_datao     (write_data),
        .reg_datai     (read_data),
        .reg_read      (reg_read),
        .reg_write     (reg_write),
        .reg_addrvalid (reg_addrvalid)
    );

    cw305_reg_ibex #(
        .pBYTECNT_SIZE (pBYTECNT_SIZE),
        .pADDR_WIDTH   (pADDR_WIDTH)
    ) U_reg_ibex (
        .reset_i       (reset),
        .crypto_clk    (crypt_clk),
        .usb_clk       (usb_clk_buf),
        .reg_address   (reg_address[pADDR_WIDTH-pBYTECNT_SIZE-1:0]),
        .reg_bytecnt   (reg_bytecnt),
        .read_data     (read_data),
        .write_data    (write_data),
        .reg_read      (reg_read),
        .reg_write     (reg_write),
        .reg_addrvalid (reg_addrvalid),
        .exttrigger_in (usb_trigger),
        .O_clksettings (clk_settings),
        .O_user_led    (led3),
        .tio_trigger   (tio_trigger)
    );

    clocks U_clocks (
        .usb_clk       (usb_clk),
        .usb_clk_buf   (usb_clk_buf),
        .I_j16_sel     (j16_sel),
        .I_k16_sel     (k16_sel),
        .I_clock_reg   (clk_settings),
        .I_cw_clkin    (tio_clkin),
        .I_pll_clk1    (pll_clk1),
        .O_cw_clkout   (tio_clkout),
        .O_cryptoclk   (crypt_clk)
    );

endmodule
`default_nettype wire
