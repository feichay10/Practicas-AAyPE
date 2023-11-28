/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * @file Ejemplo.cu
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Sumar los elementos de un vector de tipo float
 * @version 0.1
 * 
 * Compilar y ejecutar con: nvcc sumElementVector.cu -o sumElementVector && ./sumElementVector
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float d_sum; // Variable global en el device, se inicializa a 0

__global__ void sumElementVector(float *d_A, int numElem) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElem) {
    atomicAdd(&d_sum, d_A[i]);
  }
}

// fabs(d_suma - h_suma ) < 1e-2

// Hacerlo con patron de paralelismo: reducción
int main(int argc, char **argv) {
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
  cudaMemcpy(d_A, h_A, numElem * sizeof(float), cudaMemcpyHostToDevice);              // Copiar datos de host a device
  sumElementVector<<<(numElem + 255) / 256, 256>>>(d_A, numElem);                     // Lanzar el kernel
  cudaMemcpyFromSymbol(&h_sumaGPU, d_sum, sizeof(float), 0, cudaMemcpyDeviceToHost);  // Copiar datos de device a host
  
  std::cout << "Suma en CPU: " << h_suma << std::endl;
  std::cout << "Suma en GPU: " << h_sumaGPU << std::endl;
  std::cout << "Error: " << fabs(d_sum - h_suma) << std::endl;
  
  // Liberar memoria
  cudaFree(d_A);
  free(h_A);
  return 0;
}

