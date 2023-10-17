/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * @file Ejemplo.cu
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Incrementar los elementos de un vector
 * @version 0.1
 * 
 * Compilar con: nvcc -o Ejemplo Ejemplo.cu
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Kernel definition: 
__global__ void incVec(float *A, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    A[i]++;
  }
}

int main() { 
  cudaError_t err;
  int N = 5000000;
  float *h_A = (float *)malloc(N * sizeof(float)); // h_A = host_A
  float *d_A;                                      // d_A = device_A                   
  err = cudaMalloc((void **)&d_A, N * sizeof(float));
  if (err != cudaSuccess) {
    std::string error = cudaGetErrorString(err); // devuelve una cadena de error de la última llamada a una función de la API CUDA
    std::cout << "Error reservando memoria para d_A: " << error << std::endl;
    exit(0);
  } else {
    std::cout << "Reserva de memoria para d_A correcta" << std::endl;
  }

  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0f;
  }

  err = cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  incVec<<<gridSize, blockSize>>>(d_A, N);

  cudaMemcpy(h_A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  free(h_A);

  return 0;
}