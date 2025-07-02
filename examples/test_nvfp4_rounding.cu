// Compilation: nvcc -arch=sm_90 -std=c++17 test_nvfp4_rounding.cu -o test_nvfp4_rounding
// For Godbolt: Add flags: -arch=sm_90 -std=c++17
// Note: Requires SM 9.0+ for FP4 E2M1 intrinsics

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

// FP4 E2M1 lookup table
const float fp4_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Simple test kernel to verify PTX instruction works
__global__ void test_ptx_instruction() {
    // Test with known values
    float test_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned int results[4];

    for (int i = 0; i < 4; i++) {
        asm volatile (
            "{\n\t"
            "  .reg .b8 fp4_byte;\n\t"
            "  cvt.rn.satfinite.e2m1x2.f32 fp4_byte, %1, %2;\n\t"
            "  cvt.u32.u8 %0, fp4_byte;\n\t"
            "}\n\t"
            : "=r"(results[i])
            : "f"(test_vals[i]), "f"(0.0f)
        );
    }

    // Print results (only thread 0)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("PTX Instruction Test:\n");
        for (int i = 0; i < 4; i++) {
            uint8_t byte = results[i] & 0xFF;
            uint8_t low = byte & 0xF;
            uint8_t high = (byte >> 4) & 0xF;
            printf("  %.1f → byte: 0x%02x (low: 0x%x, high: 0x%x)\n",
                   test_vals[i], byte, low, high);
        }
        printf("\n");
    }
}

// Test kernel - correct PTX usage based on NVIDIA docs
__global__ void test_fp4_conversion_kernel(
    float* inputs,
    uint8_t* fp4_outputs,
    uint8_t* raw_bytes,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float val1 = inputs[idx];
    float val2 = 0.0f;  // dummy second value

    // The cvt instruction outputs a single byte containing two FP4 values
    // We need to use inline PTX with proper register declarations
    unsigned int result = 0;

    asm volatile (
        "{\n\t"
        "  .reg .b8 fp4_byte;\n\t"
        "  cvt.rn.satfinite.e2m1x2.f32 fp4_byte, %1, %2;\n\t"
        "  cvt.u32.u8 %0, fp4_byte;\n\t"  // Convert byte to 32-bit for output
        "}\n\t"
        : "=r"(result)
        : "f"(val1), "f"(val2)
    );

    // Extract the byte
    uint8_t byte_result = result & 0xFF;
    raw_bytes[idx] = byte_result;

    // The e2m1x2 format packs two FP4 values:
    // First float → ??? (need to determine which nibble)
    // Second float → ??? (need to determine which nibble)

    // For now, store the full byte and we'll analyze it on the host
    fp4_outputs[idx] = byte_result;
}

int main() {
    std::cout << "Testing CUDA FP4 E2M1 Rounding Behavior\n";
    std::cout << "=======================================\n\n";

    // First, run a simple test to verify PTX instruction works
    test_ptx_instruction<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Print all possible E2M1 values
    std::cout << "All possible FP4 E2M1 values:\n";
    std::cout << "-----------------------------\n";
    std::cout << "FP4   Binary   Value\n";
    std::cout << "---   ------   -----\n";

    for (uint8_t i = 0; i <= 15; i++) {
        std::cout << "0x" << std::hex << (int)i << std::dec
                  << "   0b" << ((i>>3)&1) << ((i>>2)&1) << ((i>>1)&1) << (i&1)
                  << "   " << std::setw(5) << fp4_e2m1_lut[i] << "\n";
    }
    std::cout << "\n";

    // Test values - focusing on tie cases
    std::vector<float> test_values = {
        // Perfect tie cases
        0.75f,    // Exactly between 0.5 and 1.0
        1.25f,    // Exactly between 1.0 and 1.5
        1.75f,    // Exactly between 1.5 and 2.0
        2.5f,     // Exactly between 2.0 and 3.0
        3.5f,     // Exactly between 3.0 and 4.0
        5.0f,     // Exactly between 4.0 and 6.0
        // Negative ties
        -0.75f, -1.25f, -1.75f, -2.5f, -3.5f, -5.0f,
        // Non-ties for comparison
        0.6f, 1.1f, 2.2f, 2.8f
    };

    // Allocate memory
    float* d_inputs;
    uint8_t* d_outputs;
    uint8_t* d_raw_bytes;
    cudaMalloc(&d_inputs, test_values.size() * sizeof(float));
    cudaMalloc(&d_outputs, test_values.size() * sizeof(uint8_t));
    cudaMalloc(&d_raw_bytes, test_values.size() * sizeof(uint8_t));

    // Copy inputs
    cudaMemcpy(d_inputs, test_values.data(),
               test_values.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // Run kernel
    int threads = 256;
    int blocks = (test_values.size() + threads - 1) / threads;
    test_fp4_conversion_kernel<<<blocks, threads>>>(d_inputs, d_outputs, d_raw_bytes, test_values.size());
    cudaDeviceSynchronize();

    // Get results
    std::vector<uint8_t> h_outputs(test_values.size());
    std::vector<uint8_t> h_raw_bytes(test_values.size());
    cudaMemcpy(h_outputs.data(), d_outputs,
               h_outputs.size() * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_raw_bytes.data(), d_raw_bytes,
               h_raw_bytes.size() * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    // Debug: Show raw bytes and figure out nibble ordering
    std::cout << "\nDebug: Raw bytes from PTX instruction:\n";
    std::cout << "--------------------------------------\n";

    for (size_t i = 0; i < 6 && i < test_values.size(); i++) {
        uint8_t raw = h_raw_bytes[i];
        uint8_t low = raw & 0xF;
        uint8_t high = (raw >> 4) & 0xF;

        std::cout << "Input: " << std::setw(6) << test_values[i]
                  << " → Raw byte: 0x" << std::hex << std::setw(2) << std::setfill('0') << (int)raw << std::dec;

        float decoded_low = fp4_e2m1_lut[low];
        float decoded_high = fp4_e2m1_lut[high];
        float error_low = std::abs(test_values[i] - decoded_low);
        float error_high = std::abs(test_values[i] - decoded_high);

        std::cout << "\n  Low nibble:  0x" << std::hex << (int)low << std::dec
                  << " = " << std::setw(5) << decoded_low
                  << " (error: " << error_low << ")";
        if (error_low < error_high) std::cout << " ← likely match";

        std::cout << "\n  High nibble: 0x" << std::hex << (int)high << std::dec
                  << " = " << std::setw(5) << decoded_high
                  << " (error: " << error_high << ")";
        if (error_high < error_low) std::cout << " ← likely match";

        std::cout << std::setfill(' ') << "\n\n";
    }

    // Determine which nibble to use based on errors
    std::cout << "Determining nibble assignment...\n";
    float total_error_low = 0, total_error_high = 0;
    int count_low_better = 0, count_high_better = 0;

    for (size_t i = 0; i < test_values.size(); i++) {
        uint8_t raw = h_raw_bytes[i];
        uint8_t low = raw & 0xF;
        uint8_t high = (raw >> 4) & 0xF;

        float error_low = std::abs(test_values[i] - fp4_e2m1_lut[low]);
        float error_high = std::abs(test_values[i] - fp4_e2m1_lut[high]);

        total_error_low += error_low;
        total_error_high += error_high;

        if (error_low < error_high) count_low_better++;
        else if (error_high < error_low) count_high_better++;
    }

    bool use_low_nibble = (total_error_low <= total_error_high);
    std::cout << "Total error using low nibble: " << total_error_low << "\n";
    std::cout << "Total error using high nibble: " << total_error_high << "\n";
    std::cout << "Decision: Use " << (use_low_nibble ? "LOW" : "HIGH") << " nibble for first float\n\n";

    // Update outputs based on decision
    for (size_t i = 0; i < h_outputs.size(); i++) {
        if (use_low_nibble) {
            h_outputs[i] = h_raw_bytes[i] & 0xF;
        } else {
            h_outputs[i] = (h_raw_bytes[i] >> 4) & 0xF;
        }
    }

    // Analyze results
    std::cout << "Conversion Results:\n";
    std::cout << "==================\n";
    std::cout << std::setw(8) << "Input"
              << std::setw(8) << "FP4"
              << std::setw(10) << "Decoded"
              << std::setw(10) << "Error"
              << std::setw(20) << "Comment\n";

    for (size_t i = 0; i < test_values.size(); i++) {
        float input = test_values[i];
        uint8_t fp4 = h_outputs[i];
        float decoded = fp4_e2m1_lut[fp4];
        float error = std::abs(input - decoded);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(8) << input
                  << std::setw(8) << "0x" << std::hex << (int)fp4 << std::dec
                  << std::setw(10) << decoded
                  << std::setw(10) << error;

        // Check for tie cases
        bool is_tie = false;
        float option1 = 0, option2 = 0;
        for (int j = 0; j < 16; j++) {
            for (int k = j+1; k < 16; k++) {
                if (std::abs(std::abs(input - fp4_e2m1_lut[j]) -
                            std::abs(input - fp4_e2m1_lut[k])) < 1e-6f &&
                    std::abs(input - fp4_e2m1_lut[j]) < 0.51f) {
                    is_tie = true;
                    option1 = fp4_e2m1_lut[j];
                    option2 = fp4_e2m1_lut[k];
                    if (option1 > option2) std::swap(option1, option2);
                }
            }
        }

        if (is_tie) {
            std::cout << " (tie: " << option1 << " vs " << option2 << ")";
        }
        std::cout << "\n";
    }

    // Detailed tie analysis
    std::cout << "\n\nTie-Breaking Analysis:\n";
    std::cout << "=====================\n";
    std::cout << "For round-to-nearest-even, ties should go to the value with even mantissa (m=0)\n\n";

    std::cout << std::setw(8) << "Input"
              << std::setw(15) << "Options"
              << std::setw(10) << "Result"
              << std::setw(10) << "Mantissa"
              << std::setw(20) << "Round-to-even?\n";

    // Analyze specific tie cases
    std::vector<float> tie_values = {0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    std::vector<std::pair<float, float>> tie_options = {
        {0.5f, 1.0f}, {1.0f, 1.5f}, {1.5f, 2.0f},
        {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 6.0f}
    };

    for (size_t i = 0; i < tie_values.size(); i++) {
        // Find the result in our test
        for (size_t j = 0; j < test_values.size(); j++) {
            if (std::abs(test_values[j] - tie_values[i]) < 1e-6f) {
                uint8_t fp4 = h_outputs[j];
                float decoded = fp4_e2m1_lut[fp4];
                int mantissa = fp4 & 1;

                std::cout << std::setw(8) << tie_values[i]
                          << std::setw(15) << tie_options[i].first
                          << " vs " << tie_options[i].second
                          << std::setw(10) << decoded
                          << std::setw(10) << "m=" << mantissa
                          << std::setw(20) << (mantissa == 0 ? "Yes ✓" : "No ✗")
                          << "\n";
                break;
            }
        }
    }

    // Special focus on 2.5
    std::cout << "\n\n2.5 Detailed Analysis:\n";
    std::cout << "=====================\n";
    for (size_t i = 0; i < test_values.size(); i++) {
        if (std::abs(test_values[i] - 2.5f) < 1e-6f) {
            uint8_t fp4 = h_outputs[i];
            float decoded = fp4_e2m1_lut[fp4];

            std::cout << "Input: 2.5\n";
            std::cout << "FP4 encoding: 0x" << std::hex << (int)fp4 << std::dec
                      << " (0b" << ((fp4>>3)&1) << ((fp4>>2)&1)
                      << ((fp4>>1)&1) << (fp4&1) << ")\n";
            std::cout << "Decoded value: " << decoded << "\n";
            std::cout << "Mantissa bit: " << (fp4 & 1) << "\n\n";

            std::cout << "Options were:\n";
            std::cout << "  2.0 (0x4, m=0) - even mantissa\n";
            std::cout << "  3.0 (0x5, m=1) - odd mantissa\n\n";

            if (fp4 == 0x4) {
                std::cout << "✓ Correctly rounded to even (2.0)\n";
            } else if (fp4 == 0x5) {
                std::cout << "✗ Rounded to odd (3.0) - NOT round-to-nearest-even\n";
            } else {
                std::cout << "! Unexpected result\n";
            }
            break;
        }
    }

    // Cleanup
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_raw_bytes);

    return 0;
}
