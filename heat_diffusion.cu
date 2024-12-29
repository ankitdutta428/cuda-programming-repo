/*%%cuda*/  #Comment out the first line 
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void heatDiffusionKernel(float *grid, float *newGrid, int N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1 && z > 0 && z < N - 1) {
        int idx = x + y * N + z * N * N;
        newGrid[idx] = 0.2 * (grid[idx] + grid[idx - 1] + grid[idx + 1] +
                               grid[idx - N] + grid[idx + N] +
                               grid[idx - N * N] + grid[idx + N * N]);
    }
}

class HeatDiffusion {
public:
    HeatDiffusion(int gridSize) : N(gridSize), bytes(N * N * N * sizeof(float)) {
        h_grid.resize(N * N * N, 0.0);
        h_newGrid.resize(N * N * N, 0.0);

        for (int i = 0; i < N * N * N; i++) {
            h_grid[i] = (i % 100 == 0) ? 100.0 : 0.0; // Heat source
        }

        cudaMalloc(&d_grid, bytes);
        cudaMalloc(&d_newGrid, bytes);
        cudaMemcpy(d_grid, h_grid.data(), bytes, cudaMemcpyHostToDevice);
    }

    ~HeatDiffusion() {
        cudaFree(d_grid);
        cudaFree(d_newGrid);
    }

    void simulate(int timeSteps) {
        dim3 threads(8, 8, 8);
        dim3 blocks((N + 7) / 8, (N + 7) / 8, (N + 7) / 8);

        for (int t = 0; t < timeSteps; t++) {
            heatDiffusionKernel<<<blocks, threads>>>(d_grid, d_newGrid, N);
            cudaMemcpy(d_grid, d_newGrid, bytes, cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(h_newGrid.data(), d_newGrid, bytes, cudaMemcpyDeviceToHost);
    }

    float getCenterHeat() const {
        return h_newGrid[N / 2 * N * N + N / 2 * N + N / 2];
    }

private:
    int N;
    size_t bytes;
    std::vector<float> h_grid, h_newGrid;
    float *d_grid, *d_newGrid;
};

int main() {
    int gridSize = 32;
    int timeSteps = 100;

    HeatDiffusion simulation(gridSize);

    simulation.simulate(timeSteps);

    std::cout << "Final heat at center: " << simulation.getCenterHeat() << std::endl;

    return 0;
}

