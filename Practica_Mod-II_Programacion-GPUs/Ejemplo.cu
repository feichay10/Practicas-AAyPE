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

// Función para obtener el id del SM
__device__ int get_smid(void) {
  int ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

// Kernel definition: 
__global__ void incVec(float *A, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    // printf("Thread %d ejecutando el kernel\n", i);
    // printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
    A[i]++;
  }

  // if (i == 0)
    printf("Thread %d ejecutando el kernel en el SM %d\n", i, get_smid());

}

int main() { 
  cudaError_t err;      // código de error devuelto por las funciones CUDA 
  int N = 500000;         // número de elementos del vector
  int blockSize = 512;  // cantidad de hilos por bloque, se puede cambiar a 512, 1024, ...
  int gridSize = (N + blockSize - 1) / blockSize;
  float *h_A = (float *)malloc(N * sizeof(float));    // h_A = host_A
  float *d_A;                                         // d_A = device_A                   
  err = cudaMalloc((void **)&d_A, N * sizeof(float)); // Reservar memoria en la GPU
  // medir el tiempo:
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventSynchronize(stop);
  float elapsedTime;

  // Comprobar si se ha reservado memoria correctamente
  if (err != cudaSuccess) {
    std::string error = cudaGetErrorString(err); // devuelve una cadena de error de la última llamada a una función de la API CUDA
    std::cout << "Error reservando memoria para d_A: " << error << std::endl;
    exit(0);
  } else {
    std::cout << "Reserva de memoria para d_A correcta" << std::endl;
  }

  // Inicializar el vector en la CPU
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0f;
  }

  err = cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice); // Copiar el vector de la CPU a la GPU
  
  cudaEventRecord(start, 0); // Iniciar el cronómetro
  incVec<<<gridSize, blockSize>>>(d_A, N); // Lanzar el kernel
  cudaEventRecord(stop, 0); // Parar el cronómetro

  cudaMemcpy(h_A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost); // Copiar el resultado de la GPU a la CPU

  cudaFree(d_A); // Liberar memoria de la GPU
  free(h_A);     // Liberar memoria de la CPU

  cudaEventElapsedTime(&elapsedTime, start, stop); // Calcular el tiempo transcurrido
  std::cout << "Tiempo de ejecución: " << elapsedTime << " ms" << std::endl;

  return 0;
}