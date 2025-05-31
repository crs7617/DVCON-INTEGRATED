module MEMORY_CONTROLLER #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 22,  // ~3MB addressing (2^22 = 4M addresses)
    parameter NUM_CLIENTS = 3   // Vision, Audio, Motion processing units
)(
    input wire clk,
    input wire rst_n,
    
    // Client interface signals (from processing units)
    input wire [NUM_CLIENTS-1:0] client_req,         // Request from each client
    input wire [NUM_CLIENTS-1:0] client_rw,          // 0: Read, 1: Write
    input wire [NUM_CLIENTS-1:0][ADDR_WIDTH-1:0] client_addr,
    input wire [NUM_CLIENTS-1:0][DATA_WIDTH-1:0] client_wdata,
    output reg [NUM_CLIENTS-1:0] client_ack,         // Acknowledge to clients
    output reg [NUM_CLIENTS-1:0][DATA_WIDTH-1:0] client_rdata,
    
    // DMA Engine interface
    input wire dma_req,
    input wire dma_rw,
    input wire [ADDR_WIDTH-1:0] dma_addr,
    input wire [DATA_WIDTH-1:0] dma_wdata,
    output reg dma_ack,
    output reg [DATA_WIDTH-1:0] dma_rdata,
    
    // SRAM Buffer interface
    output reg port1_en,
    output reg port1_we,
    output reg [ADDR_WIDTH-1:0] port1_addr,
    output reg [DATA_WIDTH-1:0] port1_din,
    input wire [DATA_WIDTH-1:0] port1_dout,
    
    output reg port2_en,
    output reg port2_we,
    output reg [ADDR_WIDTH-1:0] port2_addr,
    output reg [DATA_WIDTH-1:0] port2_din,
    input wire [DATA_WIDTH-1:0] port2_dout,
    
    output reg port3_en,
    output reg port3_we,
    output reg [ADDR_WIDTH-1:0] port3_addr,
    output reg [DATA_WIDTH-1:0] port3_din,
    input wire [DATA_WIDTH-1:0] port3_dout,
    
    // Configuration and status
    input wire [1:0] priority_mode,          // 0: Round-robin, 1: Fixed, 2: Weighted
    input wire [NUM_CLIENTS*2-1:0] client_priority, // Priority levels (packed 2-bit per client)
    output reg [NUM_CLIENTS-1:0] client_active,     // Currently active client
    output wire [2:0] memory_utilization      // Memory utilization indicator (0-7)
);

    // State machine states
    localparam IDLE = 3'b000;
    localparam GRANT_ACCESS = 3'b001;
    localparam WAIT_SRAM = 3'b010;
    localparam RETURN_DATA = 3'b011;
    localparam DMA_ACCESS = 3'b100;
    localparam DMA_WAIT = 3'b101;
    
    // Register signals
    reg [2:0] state, next_state;
    reg [1:0] current_client, next_client;
    reg [1:0] port_map [0:NUM_CLIENTS-1];  // Maps clients to SRAM ports
    reg [NUM_CLIENTS-1:0] pending_req;
    reg [1:0] dma_port;                    // Port assigned to DMA
    
    // Request tracking counters
    reg [7:0] req_count [0:NUM_CLIENTS-1];
    reg [7:0] access_count [0:NUM_CLIENTS-1];
    reg [15:0] memory_access_total;
    
    // Loop variables - declared at the module level, not inside loops
    integer i, j;

    // Memory utilization calculation
    assign memory_utilization = (memory_access_total > 16'd700) ? 3'b111 :
                               (memory_access_total > 16'd600) ? 3'b110 :
                               (memory_access_total > 16'd500) ? 3'b101 :
                               (memory_access_total > 16'd400) ? 3'b100 :
                               (memory_access_total > 16'd300) ? 3'b011 :
                               (memory_access_total > 16'd200) ? 3'b010 :
                               (memory_access_total > 16'd100) ? 3'b001 : 3'b000;
    
    // Initialize port mapping
    initial begin
        for (i = 0; i < NUM_CLIENTS; i = i + 1) begin
            port_map[i] = i;  // Default: Client i uses port i+1
            req_count[i] = 8'd0;
            access_count[i] = 8'd0;
        end
        dma_port = 2'd0;      // Default: DMA uses port 1 when needed
    end
    
    // State machine: Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            current_client <= 2'd0;
            pending_req <= {NUM_CLIENTS{1'b0}};
            memory_access_total <= 16'd0;
            
            for (i = 0; i < NUM_CLIENTS; i = i + 1) begin
                req_count[i] <= 8'd0;
                access_count[i] <= 8'd0;
            end
            
            client_ack <= {NUM_CLIENTS{1'b0}};
            dma_ack <= 1'b0;
            client_active <= {NUM_CLIENTS{1'b0}};
        end
        else begin
            state <= next_state;
            current_client <= next_client;
            
            // Track pending requests
            for (i = 0; i < NUM_CLIENTS; i = i + 1) begin
                if (client_req[i] && !client_ack[i])
                    pending_req[i] <= 1'b1;
                else if (client_ack[i])
                    pending_req[i] <= 1'b0;
                    
                // Count requests for statistics
                if (client_req[i] && !pending_req[i])
                    req_count[i] <= req_count[i] + 8'd1;
                    
                // Count accesses
                if (client_ack[i])
                    access_count[i] <= access_count[i] + 8'd1;
            end
            
            // Update active client indicator
            client_active <= {NUM_CLIENTS{1'b0}};
            if (state == GRANT_ACCESS || state == WAIT_SRAM || state == RETURN_DATA)
                client_active[current_client] <= 1'b1;
                
            // Track total memory accesses (rolling window of ~1000 cycles)
            if (memory_access_total > 16'd900) 
                memory_access_total <= memory_access_total - 16'd10;
                
            if (|client_ack || dma_ack)
                memory_access_total <= memory_access_total + 16'd1;
        end
    end
    
    // Client selection logic based on priority mode
    function [1:0] select_next_client;
        input [NUM_CLIENTS-1:0] pending;
        input [1:0] current;
        input [1:0] mode;
        input [NUM_CLIENTS*2-1:0] priorities;  // Packed priorities
        
        reg [1:0] selected;
        reg [1:0] client_pri [0:NUM_CLIENTS-1];  // Local unpacked priorities
        integer j;  // Using the j variable declared at module level
        
        begin
            // Unpack priorities - this is compliant with standard Verilog
            for (j = 0; j < NUM_CLIENTS; j = j + 1) begin
                client_pri[j] = priorities[j*2 +: 2];
            end
            
            selected = 2'd0;
            
            case (mode)
                // Round-robin
                2'd0: begin
                    case (current)
                        2'd0: selected = (pending[1]) ? 2'd1 : ((pending[2]) ? 2'd2 : ((pending[0]) ? 2'd0 : 2'd0));
                        2'd1: selected = (pending[2]) ? 2'd2 : ((pending[0]) ? 2'd0 : ((pending[1]) ? 2'd1 : 2'd1));
                        2'd2: selected = (pending[0]) ? 2'd0 : ((pending[1]) ? 2'd1 : ((pending[2]) ? 2'd2 : 2'd2));
                        default: selected = 2'd0;
                    endcase
                end
                
                // Fixed priority (0 highest, 3 lowest)
                2'd1: begin
                    if (pending[0] && client_pri[0] == 2'd0) selected = 2'd0;
                    else if (pending[1] && client_pri[1] == 2'd0) selected = 2'd1;
                    else if (pending[2] && client_pri[2] == 2'd0) selected = 2'd2;
                    else if (pending[0] && client_pri[0] == 2'd1) selected = 2'd0;
                    else if (pending[1] && client_pri[1] == 2'd1) selected = 2'd1;
                    else if (pending[2] && client_pri[2] == 2'd1) selected = 2'd2;
                    else if (pending[0]) selected = 2'd0;
                    else if (pending[1]) selected = 2'd1;
                    else if (pending[2]) selected = 2'd2;
                    else selected = current;
                end
                
                // Weighted priority
                2'd2: begin
                    // Use access_count to track fairness
                    if (pending[0] && (access_count[0] <= access_count[1]) && 
                        (access_count[0] <= access_count[2])) selected = 2'd0;
                    else if (pending[1] && (access_count[1] <= access_count[0]) && 
                            (access_count[1] <= access_count[2])) selected = 2'd1;
                    else if (pending[2] && (access_count[2] <= access_count[0]) && 
                            (access_count[2] <= access_count[1])) selected = 2'd2;
                    else if (pending[0]) selected = 2'd0;
                    else if (pending[1]) selected = 2'd1;
                    else if (pending[2]) selected = 2'd2;
                    else selected = current;
                end
                
                default: selected = current;
            endcase
            
            select_next_client = selected;
        end
    endfunction

    // State machine: Combinational logic
    always @(*) begin
        // Default values
        next_state = state;
        next_client = current_client;
        client_ack = {NUM_CLIENTS{1'b0}};
        dma_ack = 1'b0;
        
        // Default SRAM port signals
        port1_en = 1'b0;
        port1_we = 1'b0;
        port1_addr = {ADDR_WIDTH{1'b0}};
        port1_din = {DATA_WIDTH{1'b0}};
        
        port2_en = 1'b0;
        port2_we = 1'b0;
        port2_addr = {ADDR_WIDTH{1'b0}};
        port2_din = {DATA_WIDTH{1'b0}};
        
        port3_en = 1'b0;
        port3_we = 1'b0;
        port3_addr = {ADDR_WIDTH{1'b0}};
        port3_din = {DATA_WIDTH{1'b0}};
        
        // Clear client data by default
        for (i = 0; i < NUM_CLIENTS; i = i + 1) begin
            client_rdata[i] = {DATA_WIDTH{1'b0}};
        end
        dma_rdata = {DATA_WIDTH{1'b0}};
        
        case (state)
            IDLE: begin
                // Check for DMA request (highest priority)
                if (dma_req) begin
                    next_state = DMA_ACCESS;
                end
                // Otherwise check for client requests
                else if (|pending_req) begin
                    next_client = select_next_client(pending_req, current_client, priority_mode, client_priority);
                    next_state = GRANT_ACCESS;
                end
            end
            
            GRANT_ACCESS: begin
                next_state = WAIT_SRAM;
                
                // Set up SRAM port based on mapping
                case (port_map[current_client])
                    2'd0: begin
                        port1_en = 1'b1;
                        port1_we = client_rw[current_client];
                        port1_addr = client_addr[current_client];
                        port1_din = client_wdata[current_client];
                    end
                    2'd1: begin
                        port2_en = 1'b1;
                        port2_we = client_rw[current_client];
                        port2_addr = client_addr[current_client];
                        port2_din = client_wdata[current_client];
                    end
                    2'd2: begin
                        port3_en = 1'b1;
                        port3_we = client_rw[current_client];
                        port3_addr = client_addr[current_client];
                        port3_din = client_wdata[current_client];
                    end
                endcase
            end
            
            WAIT_SRAM: begin
                // Wait for SRAM to complete operation
                next_state = RETURN_DATA;
            end
            
            RETURN_DATA: begin
                // Send acknowledgment and data to client
                client_ack[current_client] = 1'b1;
                
                // Route data based on port mapping
                case (port_map[current_client])
                    2'd0: client_rdata[current_client] = port1_dout;
                    2'd1: client_rdata[current_client] = port2_dout;
                    2'd2: client_rdata[current_client] = port3_dout;
                endcase
                
                next_state = IDLE;
            end
            
            DMA_ACCESS: begin
                next_state = DMA_WAIT;
                
                // Set up SRAM port for DMA
                case (dma_port)
                    2'd0: begin
                        port1_en = 1'b1;
                        port1_we = dma_rw;
                        port1_addr = dma_addr;
                        port1_din = dma_wdata;
                    end
                    2'd1: begin
                        port2_en = 1'b1;
                        port2_we = dma_rw;
                        port2_addr = dma_addr;
                        port2_din = dma_wdata;
                    end
                    2'd2: begin
                        port3_en = 1'b1;
                        port3_we = dma_rw;
                        port3_addr = dma_addr;
                        port3_din = dma_wdata;
                    end
                endcase
            end
            
            DMA_WAIT: begin
                dma_ack = 1'b1;
                
                // Route data based on port mapping
                case (dma_port)
                    2'd0: dma_rdata = port1_dout;
                    2'd1: dma_rdata = port2_dout;
                    2'd2: dma_rdata = port3_dout;
                endcase
                
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end

endmodule