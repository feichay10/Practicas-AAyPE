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

#define REPETITIONS 20  // Repeticion de pruevas para calculo de media, max y min
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
  // Valores para la inicializacion del vector de datos de entrada
  int random = time(NULL);
  static curandState *states;

  int h_vector_H[M];  // Vector del histograma en el host
  int threadsPerBlock = 1024; // Hilos por bloque
  int blocksPerGrid = ((N / SCALA) + threadsPerBlock - 1) / threadsPerBlock;
  float t_duration[REPETITIONS]; // Vector para guardar los tiempos de ejecucion

  // Variables para el calculo de tiempo
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < REPETITIONS; i++) {
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&states, blocksPerGrid * threadsPerBlock * sizeof(curandState)));
    kernel_init_vector<<<blocksPerGrid, threadsPerBlock>>>(random, states, threadsPerBlock, blocksPerGrid);
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    kernel_init_histogram<<<1, M>>>();
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    histogram<<<blocksPerGrid, threadsPerBlock>>>(threadsPerBlock, blocksPerGrid);

    CUDA_CHECK_RETURN(cudaGetLastError());

    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_vector_H, vector_H, M * sizeof(int)));
    int acum = 0;

    std::cout << "\nHistograma:" << std::endl;
    for (int j = 0; j < M; j++) {
      std::cout << "\tH[" << j << "]: " << h_vector_H[j] << std::endl;
      acum += h_vector_H[i];
    }
    std::cout << "Total: " << acum << std::endl;

    CUDA_CHECK_RETURN(cudaFree(states));
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&t_duration[i], start, stop));
  }
  
  float t_max = 0, t_min = FLT_MAX, mean = 0;
  for (int i = 0; i < REPETITIONS; i++) {
    mean += t_duration[i];
    if (t_duration[i] > t_max) {
      t_max = t_duration[i];
    }
    if (t_duration[i] < t_min) {
      t_min = t_duration[i];
    }
  }

  std::cout << "\n\nSe han realizado " << REPETITIONS << " pruebas" << std::endl;
  std::cout << "Obteniendo un tiempo medio de: " << mean / REPETITIONS << " ms" << std::endl;
  std::cout << "Con un tiempo maximo de: " << t_max << " ms" << std::endl;
  std::cout << "Con un tiempo minimo de: " << t_min << " ms" << std::endl;

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