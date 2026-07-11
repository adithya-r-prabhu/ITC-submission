`default_nettype none
`timescale 1ns / 1ps







module ibex_soc_bram #(
    parameter integer MEM_WORDS   = 4096,
    parameter [31:0]  MMIO_RESULT = 32'h1000_0000,
    parameter integer RUN_CYCLES  = 600
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire        irq_en,
    input  wire        fw_commit,
    input  wire [31:0] fw_addr,
    input  wire [31:0] fw_wdata,
    input  wire [31:0] fault_word,
    output reg  [31:0] result,
    output reg  [31:0] last_pc,
    output reg  [31:0] cycle_count,
    output reg         busy,
    output reg         done,
    output reg         core_trap,
    output wire        trigger_active
);
    localparam integer AW = $clog2(MEM_WORDS);
    (* ram_style = "block" *) reg [31:0] mem [0:MEM_WORDS-1];

    reg run; reg [15:0] runcnt;
    wire core_rstn = run & ~rst;


    wire        instr_req, instr_gnt, instr_rvalid;
    wire [31:0] instr_addr, instr_rdata;
    wire        data_req, data_gnt, data_rvalid, data_we;
    wire [3:0]  data_be;
    wire [31:0] data_addr, data_wdata, data_rdata;
    wire        alert;

    ibex_bare u_core (
        .clk_i (clk), .rst_ni (core_rstn),
        .instr_req_o (instr_req), .instr_gnt_i (instr_gnt),
        .instr_rvalid_i (instr_rvalid), .instr_addr_o (instr_addr),
        .instr_rdata_i (instr_rdata),
        .data_req_o (data_req), .data_gnt_i (data_gnt),
        .data_rvalid_i (data_rvalid), .data_we_o (data_we), .data_be_o (data_be),
        .data_addr_o (data_addr), .data_wdata_o (data_wdata),
        .data_rdata_i (data_rdata), .alert_major_o (alert)
    );

    wire [AW-1:0] iidx = instr_addr[AW+1:2];
    wire [AW-1:0] didx = data_addr[AW+1:2];
    wire          is_mmio = (data_addr == MMIO_RESULT);


    assign instr_gnt = instr_req & run;
    assign data_gnt  = data_req  & run;






    wire        f_en   = fault_word[0] & run;
    wire [1:0]  f_tgt  = fault_word[2:1];
    wire        f_pol  = fault_word[4];
    wire [4:0]  f_bit  = fault_word[12:8];
    function [31:0] stuck_at;
        input [31:0] v; input en; input [4:0] b; input pol;
        stuck_at = en ? (pol ? (v | (32'd1 << b)) : (v & ~(32'd1 << b))) : v;
    endfunction


    reg [31:0] instr_rd; reg instr_rv;
    always @(posedge clk) begin
        instr_rd <= mem[iidx];
        instr_rv <= instr_gnt;
    end
    assign instr_rdata  = stuck_at(instr_rd, f_en & (f_tgt == 2'd1), f_bit, f_pol);
    assign instr_rvalid = instr_rv;


    wire        dwr      = data_gnt & data_we & ~is_mmio;
    wire [AW-1:0] portb_addr = run ? didx : fw_addr[AW+1:2];
    wire [31:0]   portb_data = run ? data_wdata : fw_wdata;
    wire [3:0]    portb_be   = run ? (dwr ? data_be : 4'b0000)
                                   : (fw_commit ? 4'b1111 : 4'b0000);
    reg [31:0] data_rd; reg data_rv, data_mmio_d;
    always @(posedge clk) begin
        if (portb_be[0]) mem[portb_addr][ 7: 0] <= portb_data[ 7: 0];
        if (portb_be[1]) mem[portb_addr][15: 8] <= portb_data[15: 8];
        if (portb_be[2]) mem[portb_addr][23:16] <= portb_data[23:16];
        if (portb_be[3]) mem[portb_addr][31:24] <= portb_data[31:24];
        data_rd     <= mem[portb_addr];
        data_rv     <= data_gnt & ~data_we;
        data_mmio_d <= data_gnt & ~data_we & is_mmio;
    end

    reg data_rv_any;
    always @(posedge clk) data_rv_any <= data_gnt;
    assign data_rvalid = data_rv_any;
    assign data_rdata  = stuck_at(data_mmio_d ? result : data_rd,
                                  f_en & (f_tgt == 2'd2), f_bit, f_pol);


    always @(posedge clk) begin
        if (rst) begin
            run<=1'b0; runcnt<=16'd0; busy<=1'b0; done<=1'b0; core_trap<=1'b0;
            result<=32'd0; last_pc<=32'd0; cycle_count<=32'd0;
        end else begin
            if (start && !run) begin
                run<=1'b1; runcnt<=16'd0; busy<=1'b1; done<=1'b0;
                core_trap<=1'b0; cycle_count<=32'd0;
            end
            if (run) begin
                cycle_count <= cycle_count + 32'd1;
                runcnt      <= runcnt + 16'd1;
                if (instr_gnt) last_pc <= instr_addr;
                if (data_gnt && data_we && is_mmio) begin
                    result <= data_wdata; done <= 1'b1;
                end
                if (alert) core_trap <= 1'b1;
                if (runcnt == RUN_CYCLES[15:0]) begin run<=1'b0; busy<=1'b0; end
            end
        end
    end
    assign trigger_active = run;

endmodule
`default_nettype wire
