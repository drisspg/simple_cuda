#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// PTX version is automatically set by nvcc for sm_100a target

__device__ __nv_bfloat16 stochastic_round_f32_to_bf16(float input, uint32_t random_bits) {
    uint32_t packed_result;
    
    // Use the correct .rs syntax for bf16x2 - converts two f32 to packed bf16x2
    // We'll use the same input twice to get one result
    asm("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" 
        : "=r"(packed_result) 
        : "f"(input), "f"(input), "r"(random_bits));
    
    // Extract the lower 16 bits (second conversion result)
    return __ushort_as_bfloat16(packed_result & 0xFFFF);
}

__global__ void stochastic_demo(float* inputs, __nv_bfloat16* outputs, uint32_t* random_bits, int n, int trials) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * trials) return;
    
    int input_idx = idx / trials;
    int trial_idx = idx % trials;
    
    // Use different random bits for each trial
    uint32_t trial_random = random_bits[idx];
    outputs[idx] = stochastic_round_f32_to_bf16(inputs[input_idx], trial_random);
}

int main() {
    const int N = 4; // Fewer values to keep output manageable
    const int TRIALS = 5; // Show 5 trials per value
    const float test_values[N] = {1.7f, 2.3f, 3.14159f, 0.1f};
    
    float *d_inputs;
    __nv_bfloat16 *d_outputs;
    uint32_t *d_random_bits;
    
    const int total_size = N * TRIALS;
    
    cudaMalloc(&d_inputs, N * sizeof(float));
    cudaMalloc(&d_outputs, total_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_random_bits, total_size * sizeof(uint32_t));
    
    // Generate different random bits for each trial
    uint32_t *random_bits = new uint32_t[total_size];
    for (int i = 0; i < total_size; i++) {
        random_bits[i] = rand();
    }
    
    cudaMemcpy(d_inputs, test_values, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_random_bits, random_bits, total_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    stochastic_demo<<<1, total_size>>>(d_inputs, d_outputs, d_random_bits, N, TRIALS);
    
    __nv_bfloat16 *outputs = new __nv_bfloat16[total_size];
    cudaMemcpy(outputs, d_outputs, total_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    printf("Stochastic Rounding Results - Multiple Trials:\n");
    printf("Notice how the same input can produce different outputs!\n\n");
    
    for (int i = 0; i < N; i++) {
        printf("Input: %.6f\n", test_values[i]);
        for (int t = 0; t < TRIALS; t++) {
            int idx = i * TRIALS + t;
            printf("  Trial %d: %.6f [rbits: 0x%08X]\n", 
                   t + 1, __bfloat162float(outputs[idx]), random_bits[idx]);
        }
        printf("\n");
    }
    
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_random_bits);
    delete[] random_bits;
    delete[] outputs;

    return 0;
}