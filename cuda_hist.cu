#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

__global__ void histogram_gpu(unsigned char *data, int *hist, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        atomicAdd(&hist[data[i]], 1);
    }
}

int main()
{
    std::vector<int> sizes = {100000, 500000, 1000000, 1500000};
    std::srand(static_cast<unsigned>(std::time(0)));

    for (int N : sizes)
    {
        std::vector<unsigned char> h_data(N);
        std::vector<int> h_hist(256, 0);

        for (int i = 0; i < N; i++)
            h_data[i] = std::rand() % 256;

        unsigned char *d_data;
        int *d_hist;
        cudaMalloc(&d_data, N * sizeof(unsigned char));
        cudaMalloc(&d_hist, 256 * sizeof(int));

        cudaMemcpy(d_data, h_data.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hist, h_hist.data(), 256 * sizeof(int), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        histogram_gpu<<<blocks, threads>>>(d_data, d_hist, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaMemcpy(h_hist.data(), d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

        long long sum = 0;
        for (int i = 0; i < 256; i++) sum += h_hist[i];

        std::cout << "-------------------------------------\n";
        std::cout << "Tamaño de datos: " << N << "\n";
        std::cout << "Suma total de frecuencias: " << sum << " (debería ser " << N << ")\n";
        std::cout << "Tiempo de ejecución del kernel: " << milliseconds << " ms\n";

        cudaFree(d_data);
        cudaFree(d_hist);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
