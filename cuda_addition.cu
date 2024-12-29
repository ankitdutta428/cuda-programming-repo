#This CUDA program enables to add two arrays in a parallel computing process

/*%%cuda*/  #Remove the comment of cuda on this line before running it on Colab

#include <iostream>
#include <cuda_runtime.h>

#define N (1 << 20) // 1 million elements
#define THREADS_PER_BLOCK 256

// CUDA kernel to perform vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Host vectors
    float *h_A, *h_B, *h_C;

    // Allocate memory on the host
    size_t bytes = N * sizeof(float);
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    // Initialize input vectors with large values
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Device vectors
    float *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "All results are correct!" << std::endl;
    }

    // Print the elements of the result vector
    for (int i = 0; i < 10000; i++) { // Print first 10 elements for brevity
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
