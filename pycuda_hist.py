import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code = """
__global__ void histogram_gpu(unsigned char *data, int *hist, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        atomicAdd(&hist[data[i]], 1);
    }
}
"""

mod = SourceModule(kernel_code)
histogram_gpu = mod.get_function("histogram_gpu")

threads_per_block = 256
sizes = [100_000, 500_000, 1_000_000, 1_500_000]

for N in sizes:
    data = np.random.randint(0, 256, size=N, dtype=np.uint8)
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    hist = np.zeros(256, dtype=np.int32)
    hist_gpu = cuda.mem_alloc(hist.nbytes)
    cuda.memcpy_htod(hist_gpu, hist)

    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    start = cuda.Event()
    end = cuda.Event()

    start.record()
    histogram_gpu(data_gpu, hist_gpu, np.int32(N),
                  block=(threads_per_block, 1, 1),
                  grid=(blocks_per_grid, 1, 1))
    end.record()
    end.synchronize()

    elapsed_time = start.time_till(end)
    cuda.memcpy_dtoh(hist, hist_gpu)

    print("-------------------------------------")
    print(f"Tamaño de datos: {N}")
    print(f"Suma total de frecuencias = {hist.sum()} (debería ser {N})")
    print(f"Tiempo de ejecución del kernel: {elapsed_time:.4f} ms")

    data_gpu.free()
    hist_gpu.free()
