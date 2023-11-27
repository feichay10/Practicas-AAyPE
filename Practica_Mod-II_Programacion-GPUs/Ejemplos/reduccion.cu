/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * @file Ejemplo.cu
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Sumar los elementos de un vector de tipo float por patron de paralelismo Reduccion
 * @version 0.1
 * 
 * Compilar y ejecutar con: nvcc reduccion.cu -o reduccion && ./reduccion
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NH 1024 // Numero de hilos

__device__ float d_sum = 0.0f; // Variable global en memoria de dispositivo

__global__ void sumElementVector(float *vector, int numElem) {
  int tidx = threadIdx.x;                            // Indice local del hilo
  int i = blockDim.x * blockIdx.x + threadIdx.x;     // Indice global del hilo
  __shared__ float s_alocal[NH];                     // Memoria compartida
  s_alocal[tidx] = (i < numElem) ? vector[i] : 0.0f; // Cargar elementos en memoria compartida
  __syncthreads();                                   // Sincronizar hilos

  // Reduccion
  for (int thread = blockDim.x / 2; thread > 0; thread >>= 1) { // int thread = NH / 2 tambien vale
    if (tidx < thread) {
      s_alocal[tidx] += s_alocal[tidx + thread];
    }
    __syncthreads();
  }
  // End Reduccion

  // Sumar elemento 0 atomicamente en d_sum
  if (tidx == 0) {
    atomicAdd(&d_sum, s_alocal[0]);
  }
}

int main() {
  int numElem = 1000000;
  float h_suma = 0.0f;
  float *h_A = (float *)malloc(numElem * sizeof(float));
  float *d_A;
  float h_sumaGPU = 0.0f;
  cudaMalloc((void **)&d_A, numElem * sizeof(float));
  
  for (int i = 0; i < numElem; i++) {
    h_A[i] = h_A[i] = (float)rand()/(float)RAND_MAX;
    h_suma += h_A[i];
  }
  
  cudaMemcpy(d_A, h_A, numElem * sizeof(float), cudaMemcpyHostToDevice);
  sumElementVector<<<(numElem + NH - 1) / NH, NH>>>(d_A, numElem);
  cudaMemcpyFromSymbol(&h_sumaGPU, d_sum, sizeof(float), 0, cudaMemcpyDeviceToHost);
  std::cout << "Suma CPU: " << h_suma << std::endl;
  std::cout << "Suma GPU: " << h_sumaGPU << std::endl;
  std::cout << "Error: " << fabs(d_sum - h_sumaGPU) << std::endl;
  cudaFree(d_A);
  free(h_A);
  return 0;
}