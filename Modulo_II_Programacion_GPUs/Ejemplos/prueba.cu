#include <iostream>
#include <cuda_runtime.h>

__global__ void incvec(float *a, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    
    a[i] = a[i] + 1.0f;
}

int main(int argc, const char** argv) {
    float h_a[] = {0, 1, 2, 3, 4, 5};
    float *d_a;
    size_t n = 6;//sizeof(h_a) / sizeof(float);
    cudaError_t err = cudaMalloc((void **)(&d_a), n * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cout << "err: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cout << "err: " << cudaGetErrorString(err) << std::endl;
    }
    incvec<<<1, 256>>>(d_a, n);
    err = cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cout << "err: " << cudaGetErrorString(err) << std::endl;
    }

    for (size_t i = 0; i < n; i++)
    {
        std::cout << h_a[i] << std::endl;
    }
    
    return 0;
}