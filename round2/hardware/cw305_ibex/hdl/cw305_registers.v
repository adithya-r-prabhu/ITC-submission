`default_nettype none
`timescale 1ns / 1ps
`include "register_map.v"






module cw305_reg_ibex #(
    parameter pADDR_WIDTH   = 21,
    parameter pBYTECNT_SIZE = 7,
    parameter pCORE_TYPE    = 8'h49,
    parameter pCORE_REV     = 8'h01,
    parameter pIDENTIFY     = 8'h32
)(
    input  wire                                  usb_clk,
    input  wire                                  crypto_clk,
    input  wire                                  reset_i,
    input  wire [pADDR_WIDTH-pBYTECNT_SIZE-1:0]  reg_address,
    input  wire [pBYTECNT_SIZE-1:0]              reg_bytecnt,
    output reg  [7:0]                            read_data,
    input  wire [7:0]                            write_data,
    input  wire                                  reg_read,
    input  wire                                  reg_write,
    input  wire                                  reg_addrvalid,
    input  wire                                  exttrigger_in,
    output reg  [4:0]                            O_clksettings,
    output reg                                   O_user_led,
    output wire                                  tio_trigger
);


    reg [31:0] fw_addr_usb, fw_wdata_usb;
    reg        start_tog, fwcommit_tog, reset_tog, clear_tog;
    reg        fw_autoinc;
    reg        irq_en_usb;
    reg [31:0] fault_word_usb;


    (* ASYNC_REG="TRUE" *) reg [2:0] start_sync, fwcommit_sync, reset_sync, clear_sync;
    (* ASYNC_REG="TRUE" *) reg [2:0] exttrig_sync;
    (* ASYNC_REG="TRUE" *) reg [31:0] fw_addr_c, fw_wdata_c;
    (* ASYNC_REG="TRUE" *) reg        irq_en_c0, irq_en_c1;
    (* ASYNC_REG="TRUE" *) reg [31:0] fault_c0, fault_c;
    wire start_pulse    = start_sync[2]    ^ start_sync[1];
    wire fwcommit_pulse = fwcommit_sync[2] ^ fwcommit_sync[1];
    wire reset_pulse    = reset_sync[2]    ^ reset_sync[1];
    wire clear_pulse    = clear_sync[2]    ^ clear_sync[1];

    wire exttrig_pulse  = exttrig_sync[1] & ~exttrig_sync[2];


    wire [31:0] soc_result, soc_pc, soc_cycle;
    wire        soc_busy, soc_done, soc_trap, soc_trig;
    reg         soc_rst;


    (* ASYNC_REG="TRUE" *) reg [31:0] result_usb0, result_usb, pc_usb0, pc_usb, cyc_usb0, cyc_usb;
    (* ASYNC_REG="TRUE" *) reg [3:0]  status_usb0, status_usb;
    wire [31:0] buildtime;


    always @(posedge usb_clk) begin
        if (reset_i) begin
            O_clksettings <= 0; O_user_led <= 0;
            fw_addr_usb <= 0; fw_wdata_usb <= 0; fw_autoinc <= 0; irq_en_usb <= 0;
            start_tog <= 0; fwcommit_tog <= 0; reset_tog <= 0; clear_tog <= 0;
            fault_word_usb <= 0;
        end else begin
            if (reg_addrvalid && reg_write) begin
                case (reg_address)
                    `REG_CLKSETTINGS: O_clksettings <= write_data[4:0];
                    `REG_USER_LED:    O_user_led    <= write_data[0];
                    `REG_FW_ADDR:     fw_addr_usb[reg_bytecnt*8 +: 8]  <= write_data;
                    `REG_FW_WDATA:    fw_wdata_usb[reg_bytecnt*8 +: 8] <= write_data;
                    `REG_FW_LOAD: begin
                        fwcommit_tog <= ~fwcommit_tog;
                        fw_autoinc   <= write_data[1];
                    end
                    `REG_CONTROL: begin
                        if (write_data[0]) start_tog <= ~start_tog;
                        if (write_data[1]) begin reset_tog <= ~reset_tog; end
                        if (write_data[2]) begin reset_tog <= ~reset_tog; end
                        if (write_data[3]) clear_tog <= ~clear_tog;
                        irq_en_usb <= write_data[4];
                    end
                    `REG_FAULT:       fault_word_usb[reg_bytecnt*8 +: 8] <= write_data;
                endcase
            end

            if (reg_addrvalid && reg_write && (reg_address == `REG_FW_LOAD) && write_data[1])
                fw_addr_usb <= fw_addr_usb + 32'd4;
        end
    end


    reg [7:0] rd;
    always @(*) begin
        rd = 8'd0;
        if (reg_addrvalid && reg_read) begin
            case (reg_address)
                `REG_CLKSETTINGS: rd = {3'b0, O_clksettings};
                `REG_USER_LED:    rd = {7'b0, O_user_led};
                `REG_CORE_TYPE:   rd = pCORE_TYPE;
                `REG_CORE_REV:    rd = pCORE_REV;
                `REG_IDENTIFY:    rd = pIDENTIFY;
                `REG_STATUS:      rd = {2'b0, status_usb};
                `REG_FW_ADDR:     rd = fw_addr_usb[reg_bytecnt*8 +: 8];
                `REG_FW_WDATA:    rd = fw_wdata_usb[reg_bytecnt*8 +: 8];
                `REG_RESULT:      rd = result_usb[reg_bytecnt*8 +: 8];
                `REG_PC:          rd = pc_usb[reg_bytecnt*8 +: 8];
                `REG_CYCLE:       rd = cyc_usb[reg_bytecnt*8 +: 8];
                `REG_BUILDTIME:   rd = buildtime[reg_bytecnt*8 +: 8];
                `REG_FAULT:       rd = fault_word_usb[reg_bytecnt*8 +: 8];
                default:          rd = 8'd0;
            endcase
        end
    end
    always @(posedge usb_clk) read_data <= rd;


    always @(posedge usb_clk) begin
        result_usb0 <= soc_result; result_usb <= result_usb0;
        pc_usb0     <= soc_pc;     pc_usb     <= pc_usb0;
        cyc_usb0    <= soc_cycle;  cyc_usb    <= cyc_usb0;
        status_usb0 <= {soc_trig, soc_trap, soc_done, soc_busy};
        status_usb  <= status_usb0;
    end


    always @(posedge crypto_clk) begin
        start_sync    <= {start_sync[1:0],    start_tog};
        fwcommit_sync <= {fwcommit_sync[1:0], fwcommit_tog};
        reset_sync    <= {reset_sync[1:0],    reset_tog};
        clear_sync    <= {clear_sync[1:0],    clear_tog};
        exttrig_sync  <= {exttrig_sync[1:0],  exttrigger_in};
        fw_addr_c     <= fw_addr_usb;
        fw_wdata_c    <= fw_wdata_usb;
        irq_en_c0     <= irq_en_usb; irq_en_c1 <= irq_en_c0;
        fault_c0      <= fault_word_usb; fault_c <= fault_c0;

        soc_rst <= reset_i | reset_pulse | clear_pulse;
    end

    ibex_soc_bram #(
        .MEM_WORDS(4096), .RUN_CYCLES(600)
    ) U_soc (
        .clk       (crypto_clk),
        .rst       (soc_rst),
        .start     (start_pulse | exttrig_pulse),
        .irq_en    (irq_en_c1),
        .fw_commit (fwcommit_pulse),
        .fw_addr   (fw_addr_c),
        .fw_wdata  (fw_wdata_c),
        .fault_word(fault_c),
        .result    (soc_result),
        .last_pc   (soc_pc),
        .cycle_count(soc_cycle),
        .busy      (soc_busy),
        .done      (soc_done),
        .core_trap (soc_trap),
        .trigger_active (soc_trig)
    );

    assign tio_trigger = soc_trig;

`ifndef __ICARUS__
    `ifndef VERILATOR
        USR_ACCESSE2 U_buildtime (.CFGCLK(), .DATA(buildtime), .DATAVALID());
    `else
        assign buildtime = 32'h0;
    `endif
`else
    assign buildtime = 32'h0;
`endif

endmodule
`default_nettype wire
