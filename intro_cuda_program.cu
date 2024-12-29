/*%%cuda*/       #Comment out the first line
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h> // Include for printf

__global__ void hellokernel() {
    printf("Hello from block: %d, thread: %d\n", blockIdx.x, threadIdx.x);
}

int main(){
    hellokernel<<<4,4>>>();
    cudaDeviceSynchronize();
}