/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * @file histograma.cu
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Version 1 histograma en CUDA, crear tantos hilos como elementos de V
 * para que cada uno se encargue de ir al elemento que le corresponda en V e
 * incremente la caja correcta en el vector histograma H (posiblemente de forma
 * atómica).
 * @version 0.1
 *
 * Compilar y ejecutar con: nvcc histograma.cu -o histograma
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cuda.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

#define N 500000000  // Numero de elementos en el vector V
#define M 8          // Numero de cajas en el histograma

#define REPETITIONS 10000  // Repeticion de pruevas para calculo de media, max y min
#define SCALA 50            // Datos calculados en cada hilo

__device__ int vector_V[N];  // Vector de datos de entrada
__device__ int vector_H[M];  // Vector del histograma, "cajas"

/**
 * @brief Funcion para la comprobación de errores de CUDA
 *
 */
static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

/**
 * @brief Kernel para la inicializacion del vector de datos de entrada
 * 
 */
__global__ void kernel_init_vector(int random, curandState *states, int threadsPerBlock, int blocksPerGrid) {
  int iterations = SCALA;
  if (blocksPerGrid - 1 == blockIdx.x && threadIdx.x == threadsPerBlock - 1) {
    iterations = iterations + (N % SCALA);
  }
  unsigned id_x = blockIdx.x * blockDim.x + threadIdx.x;
  curandState *state = states + id_x;

  curand_init(random, id_x, 0, state);
  for (int i = 0; i < iterations; i++) {
    if (id_x * SCALA + i < N) {
      vector_V[id_x * SCALA + i] = (int)((curand_uniform(state) * 1000)) % M;
    }
  }
}

/**
 * @brief Kernel para la inicializacion del vector del histograma
 * 
 */
__global__ void kernel_init_histogram() {
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (id_x < M) {
    vector_H[id_x] = 0;
  }
}

/**
 * @brief Kernel para el calculo del histograma
 * 
 * @param threadsPerBlock 
 * @param blocksPerGrid 
 * @return __global__ 
 */
__global__ void histogram(int threadsPerBlock, int blocksPerGrid) {
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  int iterations = SCALA;
  if (blocksPerGrid - 1 == blockIdx.x && threadIdx.x == threadsPerBlock - 1) {
    iterations = iterations + (N % SCALA);
  }
  for (int i = 0; i < iterations; i++) {
    if (id_x * SCALA + i < N) {
      atomicAdd(&vector_H[vector_V[id_x * SCALA + i]], 1);
    }
  }
}

int main() {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  int random = time(NULL);

  curandState *devStates;
  CUDA_CHECK_RETURN(cudaMalloc(&devStates, threadsPerBlock * blocksPerGrid * sizeof(curandState)));

  kernel_init_vector<<<blocksPerGrid, threadsPerBlock>>>(random, devStates, threadsPerBlock, blocksPerGrid);
  kernel_init_histogram<<<(M + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>();

  cudaDeviceSynchronize();

  double min_time = DBL_MAX;
  double max_time = 0;
  double mean_time = 0;

  for (int i = 0; i < REPETITIONS; i++) {
    clock_t start = clock();
    histogram<<<blocksPerGrid, threadsPerBlock>>>(threadsPerBlock, blocksPerGrid);
    cudaDeviceSynchronize();
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    mean_time += time;
    if (time < min_time) {
      min_time = time;
    }
    if (time > max_time) {
      max_time = time;
    }
  }

  mean_time /= REPETITIONS;

  std::cout << "Min time: " << min_time << std::endl;
  std::cout << "Max time: " << max_time << std::endl;
  std::cout << "Mean time: " << mean_time << std::endl;

  return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess) {
    return;
  } 

  std::cerr << statement<< " returned " << cudaGetErrorString(err) << "("<<err<< ") at "<< file << ":" << line << std::endl;
	exit (EXIT_FAILURE);
}