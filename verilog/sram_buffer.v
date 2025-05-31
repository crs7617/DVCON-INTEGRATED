module shared_sram_buffer #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 22,  // ~3MB addressing (2^22 = 4M addresses)
    parameter NUM_PORTS = 3     // For Vision, Audio, and Motion processing units
)(
    input wire clk,
    input wire rst_n,
    
    // Port 1 interface (e.g., Vision Processing Unit)
    input wire port1_en,
    input wire port1_we,
    input wire [ADDR_WIDTH-1:0] port1_addr,
    input wire [DATA_WIDTH-1:0] port1_din,
    output reg [DATA_WIDTH-1:0] port1_dout,
    
    // Port 2 interface (e.g., Audio Processing Unit)
    input wire port2_en,
    input wire port2_we,
    input wire [ADDR_WIDTH-1:0] port2_addr,
    input wire [DATA_WIDTH-1:0] port2_din,
    output reg [DATA_WIDTH-1:0] port2_dout,
    
    // Port 3 interface (e.g., Motion Analysis Engine)
    input wire port3_en,
    input wire port3_we,
    input wire [ADDR_WIDTH-1:0] port3_addr,
    input wire [DATA_WIDTH-1:0] port3_din,
    output reg [DATA_WIDTH-1:0] port3_dout
);

    // Memory bank selection parameters
    localparam NUM_BANKS = 192;  // For 3MB with 16KB per bank (192 * 16KB = 3MB)
    localparam BANK_ADDR_WIDTH = 14;  // 2^14 = 16K addresses per bank
    
    // Memory arbitration state machine
    reg [1:0] current_port;
    reg [1:0] next_port;
    
    // Bank selection logic
    wire [7:0] port1_bank = port1_addr[ADDR_WIDTH-1:BANK_ADDR_WIDTH];
    wire [7:0] port2_bank = port2_addr[ADDR_WIDTH-1:BANK_ADDR_WIDTH];
    wire [7:0] port3_bank = port3_addr[ADDR_WIDTH-1:BANK_ADDR_WIDTH];
    
    wire [BANK_ADDR_WIDTH-1:0] port1_bank_addr = port1_addr[BANK_ADDR_WIDTH-1:0];
    wire [BANK_ADDR_WIDTH-1:0] port2_bank_addr = port2_addr[BANK_ADDR_WIDTH-1:0];
    wire [BANK_ADDR_WIDTH-1:0] port3_bank_addr = port3_addr[BANK_ADDR_WIDTH-1:0];
    
    // Memory banks (using Xilinx dual-port BRAMs)
    // We'll instantiate one for demonstration, but in practice, you'll generate multiple banks
    
    // Bank signals
    reg [NUM_BANKS-1:0] bank_we_a;
    reg [NUM_BANKS-1:0] bank_we_b;
    reg [NUM_BANKS-1:0] bank_en_a;
    reg [NUM_BANKS-1:0] bank_en_b;
    reg [BANK_ADDR_WIDTH-1:0] bank_addr_a [NUM_BANKS-1:0];
    reg [BANK_ADDR_WIDTH-1:0] bank_addr_b [NUM_BANKS-1:0];
    reg [DATA_WIDTH-1:0] bank_din_a [NUM_BANKS-1:0];
    reg [DATA_WIDTH-1:0] bank_din_b [NUM_BANKS-1:0];
    wire [DATA_WIDTH-1:0] bank_dout_a [NUM_BANKS-1:0];
    wire [DATA_WIDTH-1:0] bank_dout_b [NUM_BANKS-1:0];
    
    // Bank instantiation (simplified - would be generated in actual implementation)
    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : memory_banks
            // Using True Dual Port RAM IP
            blk_mem_gen_0 bank (
                .clka(clk),
                .ena(bank_en_a[i]),
                .wea(bank_we_a[i]),
                .addra(bank_addr_a[i]),
                .dina(bank_din_a[i]),
                .douta(bank_dout_a[i]),
                
                .clkb(clk),
                .enb(bank_en_b[i]),
                .web(bank_we_b[i]),
                .addrb(bank_addr_b[i]),
                .dinb(bank_din_b[i]),
                .doutb(bank_dout_b[i])
            );
        end
    endgenerate
    
    // Memory arbitration logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_port <= 2'b00;
        end else begin
            current_port <= next_port;
        end
    end
    
    // Round-robin arbitration
    always @(*) begin
        case (current_port)
            2'b00: next_port = (port2_en) ? 2'b01 : ((port3_en) ? 2'b10 : ((port1_en) ? 2'b00 : 2'b00));
            2'b01: next_port = (port3_en) ? 2'b10 : ((port1_en) ? 2'b00 : ((port2_en) ? 2'b01 : 2'b01));
            2'b10: next_port = (port1_en) ? 2'b00 : ((port2_en) ? 2'b01 : ((port3_en) ? 2'b10 : 2'b10));
            default: next_port = 2'b00;
        endcase
    end
    
    // Bank control logic (simplified - would be more complex in actual implementation)
    integer j;
    always @(*) begin
        // Default values
        for (j = 0; j < NUM_BANKS; j = j + 1) begin
            bank_en_a[j] = 1'b0;
            bank_en_b[j] = 1'b0;
            bank_we_a[j] = 1'b0;
            bank_we_b[j] = 1'b0;
            bank_addr_a[j] = {BANK_ADDR_WIDTH{1'b0}};
            bank_addr_b[j] = {BANK_ADDR_WIDTH{1'b0}};
            bank_din_a[j] = {DATA_WIDTH{1'b0}};
            bank_din_b[j] = {DATA_WIDTH{1'b0}};
        end
        
        // Port 1 access
        if (port1_en) begin
            bank_en_a[port1_bank] = 1'b1;
            bank_we_a[port1_bank] = port1_we;
            bank_addr_a[port1_bank] = port1_bank_addr;
            bank_din_a[port1_bank] = port1_din;
        end
        
        // Port 2 access (if no conflict with port 1)
        if (port2_en && !(port1_en && port1_bank == port2_bank)) begin
            bank_en_a[port2_bank] = 1'b1;
            bank_we_a[port2_bank] = port2_we;
            bank_addr_a[port2_bank] = port2_bank_addr;
            bank_din_a[port2_bank] = port2_din;
        end else if (port2_en) begin
            // If conflict, use port B
            bank_en_b[port2_bank] = 1'b1;
            bank_we_b[port2_bank] = port2_we;
            bank_addr_b[port2_bank] = port2_bank_addr;
            bank_din_b[port2_bank] = port2_din;
        end
        
        // Port 3 access (handle conflicts)
        if (port3_en) begin
            if (!(port1_en && port1_bank == port3_bank) && 
                !(port2_en && port2_bank == port3_bank)) begin
                // No conflicts
                bank_en_a[port3_bank] = 1'b1;
                bank_we_a[port3_bank] = port3_we;
                bank_addr_a[port3_bank] = port3_bank_addr;
                bank_din_a[port3_bank] = port3_din;
            end else if (!(port2_en && port2_bank == port3_bank && 
                          bank_en_b[port2_bank])) begin
                // Use port B if available
                bank_en_b[port3_bank] = 1'b1;
                bank_we_b[port3_bank] = port3_we;
                bank_addr_b[port3_bank] = port3_bank_addr;
                bank_din_b[port3_bank] = port3_din;
            end
            // If all ports busy, will need to wait (handled by state machine)
        end
    end
    
    // Output data logic
    always @(posedge clk) begin
        if (port1_en && !port1_we)
            port1_dout <= bank_dout_a[port1_bank];
            
        if (port2_en && !port2_we) begin
            if (port1_en && port1_bank == port2_bank)
                port2_dout <= bank_dout_b[port2_bank];
            else
                port2_dout <= bank_dout_a[port2_bank];
        end
        
        if (port3_en && !port3_we) begin
            if ((port1_en && port1_bank == port3_bank) || 
                (port2_en && port2_bank == port3_bank && bank_en_b[port2_bank]))
                port3_dout <= bank_dout_b[port3_bank];
            else
                port3_dout <= bank_dout_a[port3_bank];
        end
    end

endmodule