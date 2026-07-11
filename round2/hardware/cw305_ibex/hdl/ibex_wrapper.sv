module ibex_bare import ibex_pkg::*; (
  input  logic        clk_i,
  input  logic        rst_ni,


  output logic        instr_req_o,
  input  logic        instr_gnt_i,
  input  logic        instr_rvalid_i,
  output logic [31:0] instr_addr_o,
  input  logic [31:0] instr_rdata_i,


  output logic        data_req_o,
  input  logic        data_gnt_i,
  input  logic        data_rvalid_i,
  output logic        data_we_o,
  output logic  [3:0] data_be_o,
  output logic [31:0] data_addr_o,
  output logic [31:0] data_wdata_o,
  input  logic [31:0] data_rdata_i,


  output logic        alert_major_o
);


  logic [IC_TAG_SIZE-1:0]  ic_tag_rdata  [IC_NUM_WAYS];
  logic [IC_LINE_SIZE-1:0] ic_data_rdata [IC_NUM_WAYS];
  assign ic_tag_rdata  = '{default: '0};
  assign ic_data_rdata = '{default: '0};


  logic        dummy_instr_id, dummy_instr_wb;
  logic [4:0]  rf_raddr_a, rf_raddr_b, rf_waddr_wb;
  logic        rf_we_wb;
  logic [31:0] rf_wdata_wb, rf_rdata_a, rf_rdata_b;

  ibex_core #(
    .PMPEnable         (1'b0),
    .ICache            (1'b0),
    .ICacheECC         (1'b0),
    .BranchPredictor   (1'b0),
    .WritebackStage    (1'b0),
    .SecureIbex        (1'b0),
    .DummyInstructions (1'b0),
    .RegFileECC        (1'b0),
    .MemECC            (1'b0),
    .DbgTriggerEn      (1'b0),
    .BranchTargetALU   (1'b0)
  ) u_ibex_core (
    .clk_i,
    .rst_ni,
    .hart_id_i           (32'h0),
    .boot_addr_i         (32'h00100000),


    .instr_req_o,
    .instr_gnt_i,
    .instr_rvalid_i,
    .instr_addr_o,
    .instr_rdata_i,
    .instr_err_i         (1'b0),


    .data_req_o,
    .data_gnt_i,
    .data_rvalid_i,
    .data_we_o,
    .data_be_o,
    .data_addr_o,
    .data_wdata_o,
    .data_rdata_i,
    .data_err_i          (1'b0),


    .dummy_instr_id_o    (dummy_instr_id),
    .dummy_instr_wb_o    (dummy_instr_wb),
    .rf_raddr_a_o        (rf_raddr_a),
    .rf_raddr_b_o        (rf_raddr_b),
    .rf_waddr_wb_o       (rf_waddr_wb),
    .rf_we_wb_o          (rf_we_wb),
    .rf_wdata_wb_ecc_o   (rf_wdata_wb),
    .rf_rdata_a_ecc_i    (rf_rdata_a),
    .rf_rdata_b_ecc_i    (rf_rdata_b),


    .ic_tag_req_o        (),
    .ic_tag_write_o      (),
    .ic_tag_addr_o       (),
    .ic_tag_wdata_o      (),
    .ic_tag_rdata_i      (ic_tag_rdata),
    .ic_data_req_o       (),
    .ic_data_write_o     (),
    .ic_data_addr_o      (),
    .ic_data_wdata_o     (),
    .ic_data_rdata_i     (ic_data_rdata),
    .ic_scr_key_valid_i  (1'b0),
    .ic_scr_key_req_o    (),


    .irq_software_i      (1'b0),
    .irq_timer_i         (1'b0),
    .irq_external_i      (1'b0),
    .irq_fast_i          (15'b0),
    .irq_nm_i            (1'b0),
    .irq_pending_o       (),


    .debug_req_i         (1'b0),
    .crash_dump_o        (),
    .double_fault_seen_o (),


    .fetch_enable_i      (IbexMuBiOn),
    .alert_minor_o       (),
    .alert_major_internal_o (alert_major_o),
    .alert_major_bus_o   (),
    .core_busy_o         ()
  );

  ibex_register_file_ff #(
    .RV32E             (1'b0),
    .DataWidth         (32),
    .DummyInstructions (1'b0)
  ) u_reg_file (
    .clk_i,
    .rst_ni,
    .test_en_i       (1'b0),
    .dummy_instr_id_i (dummy_instr_id),
    .dummy_instr_wb_i (dummy_instr_wb),
    .raddr_a_i        (rf_raddr_a),
    .rdata_a_o        (rf_rdata_a),
    .raddr_b_i        (rf_raddr_b),
    .rdata_b_o        (rf_rdata_b),
    .waddr_a_i        (rf_waddr_wb),
    .wdata_a_i        (rf_wdata_wb),
    .we_a_i           (rf_we_wb)
  );

endmodule
