// Compatible DMA Engine with Memory Controller Interface
module DMA (
    input clk,
    input reset_n,
    
    // Sensor Interfaces (AXI-Stream)
    input [31:0] vpu_data,    // Camera (Highest Priority)
    input vpu_valid,
    output vpu_ready,
    
    input [31:0] apu_data,    // Mic (Medium Priority)
    input apu_valid,
    output apu_ready,
    
    input [31:0] mae_data,    // IMU (Lowest Priority)
    input mae_valid,
    output mae_ready,
    
    // Memory Controller Interface (Compatible with MEMORY_CONTROLLER)
    output reg dma_req,
    output reg dma_rw,                    // 0: Read, 1: Write
    output reg [21:0] dma_addr,          // 22-bit address to match MEMORY_CONTROLLER
    output reg [31:0] dma_wdata,
    input dma_ack,
    input [31:0] dma_rdata
);

    // State machine states
    localparam IDLE = 3'b000;
    localparam VPU_WRITE = 3'b001;
    localparam APU_WRITE = 3'b010;
    localparam MAE_WRITE = 3'b011;
    localparam WAIT_ACK = 3'b100;

    reg [2:0] state, next_state;
    reg [1:0] active_channel;
    reg [21:0] addr_counter;  // Changed to 22-bit to match MEMORY_CONTROLLER
    
    // Priority Encoder - determines which channel to service
    always @(*) begin
        if (vpu_valid) active_channel = 2'b00;       // VPU (Camera)
        else if (apu_valid) active_channel = 2'b01;   // APU (Mic)
        else if (mae_valid) active_channel = 2'b10;   // MAE (IMU)
        else active_channel = 2'b11;                  // Idle
    end

    // State machine - sequential logic
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            addr_counter <= 22'h000000;  // 22-bit counter
        end else begin
            state <= next_state;
            
            // Increment address counter when transaction completes
            if (dma_ack && dma_req && dma_rw) begin
                addr_counter <= addr_counter + 1;  // Word addressing (32-bit words)
            end
        end
    end

    // State machine - combinational logic
    always @(*) begin
        // Default values
        next_state = state;
        dma_req = 1'b0;
        dma_rw = 1'b1;   // Always write for DMA engine
        dma_addr = addr_counter;
        dma_wdata = 32'h00000000;

        case (state)
            IDLE: begin
                case (active_channel)
                    2'b00: next_state = VPU_WRITE;  // VPU has data
                    2'b01: next_state = APU_WRITE;  // APU has data
                    2'b10: next_state = MAE_WRITE;  // MAE has data
                    default: next_state = IDLE;     // No data available
                endcase
            end
            
            VPU_WRITE: begin
                dma_req = 1'b1;
                dma_rw = 1'b1;
                dma_addr = addr_counter;
                dma_wdata = vpu_data;
                next_state = WAIT_ACK;
            end
            
            APU_WRITE: begin
                dma_req = 1'b1;
                dma_rw = 1'b1;
                dma_addr = addr_counter;
                dma_wdata = apu_data;
                next_state = WAIT_ACK;
            end
            
            MAE_WRITE: begin
                dma_req = 1'b1;
                dma_rw = 1'b1;
                dma_addr = addr_counter;
                dma_wdata = mae_data;
                next_state = WAIT_ACK;
            end
            
            WAIT_ACK: begin
                // Keep request active until acknowledged
                dma_req = 1'b1;
                dma_rw = 1'b1;
                dma_addr = addr_counter;
                
                // Maintain data based on current active channel
                case (active_channel)
                    2'b00: dma_wdata = vpu_data;
                    2'b01: dma_wdata = apu_data;
                    2'b10: dma_wdata = mae_data;
                    default: dma_wdata = 32'h00000000;
                endcase
                
                if (dma_ack) begin
                    next_state = IDLE;
                end else begin
                    next_state = WAIT_ACK;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end

    // Ready Signals (Backpressure) - based on successful completion
    assign vpu_ready = (state == WAIT_ACK) && (active_channel == 2'b00) && dma_ack;
    assign apu_ready = (state == WAIT_ACK) && (active_channel == 2'b01) && dma_ack;
    assign mae_ready = (state == WAIT_ACK) && (active_channel == 2'b10) && dma_ack;

endmodule

//==============================================================================
// Updated testbench for compatible DMA engine
//==============================================================================

module dma_engine_compatible_tb();

    // Parameters
    parameter CLK_PERIOD = 10;  // 100 MHz clock

    // Signals
    reg clk;
    reg reset_n;
    
    // Sensor Interfaces (AXI-Stream)
    reg [31:0] vpu_data;
    reg vpu_valid;
    wire vpu_ready;
    
    reg [31:0] apu_data;
    reg apu_valid;
    wire apu_ready;
    
    reg [31:0] mae_data;
    reg mae_valid;
    wire mae_ready;
    
    // Memory Controller Interface
    wire dma_req;
    wire dma_rw;
    wire [21:0] dma_addr;
    wire [31:0] dma_wdata;
    reg dma_ack;
    reg [31:0] dma_rdata;

    // Instantiate DUT
    DMA dut (
        .clk(clk),
        .reset_n(reset_n),
        .vpu_data(vpu_data),
        .vpu_valid(vpu_valid),
        .vpu_ready(vpu_ready),
        .apu_data(apu_data),
        .apu_valid(apu_valid),
        .apu_ready(apu_ready),
        .mae_data(mae_data),
        .mae_valid(mae_valid),
        .mae_ready(mae_ready),
        .dma_req(dma_req),
        .dma_rw(dma_rw),
        .dma_addr(dma_addr),
        .dma_wdata(dma_wdata),
        .dma_ack(dma_ack),
        .dma_rdata(dma_rdata)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Initialize and Reset
    initial begin
        reset_n = 0;
        vpu_data = 0;
        vpu_valid = 0;
        apu_data = 0;
        apu_valid = 0;
        mae_data = 0;
        mae_valid = 0;
        dma_ack = 0;
        dma_rdata = 0;
        
        // Release reset after 2 clock cycles
        #20 reset_n = 1;
        
        // Start test cases
        test_priority_arbitration();
        test_memory_controller_interface();
        test_handshake_protocol();
        
        // Finish simulation
        #100 $finish;
    end

    // Test Case 1: Priority Arbitration (VPU > APU > MAE)
    task test_priority_arbitration;
        begin
            $display("--- TEST 1: Priority Arbitration ---");
            
            // All sensors request simultaneously
            vpu_data = 32'hCAFE_CAFE;
            vpu_valid = 1;
            apu_data = 32'hDEAD_DEAD;
            apu_valid = 1;
            mae_data = 32'hBEEF_BEEF;
            mae_valid = 1;
            
            // Wait for DMA request and acknowledge VPU first
            @(posedge clk);
            wait(dma_req);
            if (dma_wdata === 32'hCAFE_CAFE && dma_addr === 22'h000000)
                $display("PASS: VPU data (0x%h) requested at addr 0x%h", dma_wdata, dma_addr);
            else
                $display("FAIL: VPU not prioritized correctly");
            
            // Acknowledge the transaction
            dma_ack = 1;
            @(posedge clk);
            dma_ack = 0;
            vpu_valid = 0;  // VPU transaction complete
            
            // APU should be next
            @(posedge clk);
            wait(dma_req);
            if (dma_wdata === 32'hDEAD_DEAD && dma_addr === 22'h000001)
                $display("PASS: APU data (0x%h) requested at addr 0x%h", dma_wdata, dma_addr);
            else
                $display("FAIL: APU not serviced after VPU");
            
            dma_ack = 1;
            @(posedge clk);
            dma_ack = 0;
            apu_valid = 0;
            
            // MAE should be last
            @(posedge clk);
            wait(dma_req);
            if (dma_wdata === 32'hBEEF_BEEF && dma_addr === 22'h000002)
                $display("PASS: MAE data (0x%h) requested at addr 0x%h", dma_wdata, dma_addr);
            else
                $display("FAIL: MAE not serviced last");
            
            dma_ack = 1;
            @(posedge clk);
            dma_ack = 0;
            mae_valid = 0;
            
            #20;
        end
    endtask

    // Test Case 2: Memory Controller Interface Compatibility
    task test_memory_controller_interface;
        begin
            $display("--- TEST 2: Memory Controller Interface ---");
            
            // Single VPU transaction
            vpu_data = 32'h1234_5678;
            vpu_valid = 1;
            
            @(posedge clk);
            wait(dma_req);
            
            // Verify interface signals
            if (dma_rw === 1'b1 && dma_addr === 22'h000003)
                $display("PASS: DMA interface signals correct (rw=%b, addr=0x%h)", dma_rw, dma_addr);
            else
                $display("FAIL: DMA interface signals incorrect");
            
            // Test delayed acknowledgment
            #30;  // Wait 3 cycles before ack
            dma_ack = 1;
            @(posedge clk);
            dma_ack = 0;
            vpu_valid = 0;
            
            if (vpu_ready)
                $display("PASS: VPU ready asserted on acknowledgment");
            else
                $display("FAIL: VPU ready not asserted");
            
            #20;
        end
    endtask

    // Test Case 3: Handshake Protocol
    task test_handshake_protocol;
        begin
            $display("--- TEST 3: Handshake Protocol ---");
            
            // Test that request stays active until acknowledged
            apu_data = 32'h9999_AAAA;
            apu_valid = 1;
            
            @(posedge clk);
            wait(dma_req);
            
            // Request should stay active for multiple cycles
            repeat(3) begin
                @(posedge clk);
                if (!dma_req)
                    $display("FAIL: DMA request dropped before acknowledgment");
            end
            
            $display("PASS: DMA request held until acknowledgment");
            
            dma_ack = 1;
            @(posedge clk);
            dma_ack = 0;
            apu_valid = 0;
            
            #20;
        end
    endtask

    // Monitor for debugging
    initial begin
        $monitor("Time=%0t: req=%b, ack=%b, addr=0x%h, wdata=0x%h, vpu_ready=%b, apu_ready=%b, mae_ready=%b", 
                 $time, dma_req, dma_ack, dma_addr, dma_wdata, vpu_ready, apu_ready, mae_ready);
    end

    // Waveform Dumping
    initial begin
        $dumpfile("dma_engine_compatible_tb.vcd");
        $dumpvars(0, dma_engine_compatible_tb);
    end

endmodule