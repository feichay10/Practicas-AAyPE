/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * @file histograma.cu
 * @brief Version 1 histograma en CUDA: Como implementación base (que podremos 
 * mejorar en tiempo o no) se pide crear tantos hilos como elementos de V para 
 * que cada uno se encargue de ir al elemento que le corresponda en V e 
 * incremente la caja correcta en el vector histograma H (posiblemente de forma 
 * atómica).
 * 
 * @version 0.1
 *
 * Compilar y ejecutar con: nvcc histograma.cu -o histograma
 *
 * @date 2023
 *
 */

#include <cuda.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <fstream>

// Function to check the return value of the CUDA runtime API call and exit
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

#define N 8000000         // Numero de elementos en el vector V
#define M 8                 // Numero de elementos o cajas en el histograma (tamaño del histograma)
#define REPETITIONS 10000      // Numero de repeticiones para el calculo de la media, max y min

#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

// Variables globales en el device
__device__ int vector_V[N]; // Vector V (vector) de un numero elevado N de elementos enteros aleatorios
__device__ int vector_H[M]; // Vector H (Histograma) que tiene M elementos que representan "cajas"

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
  if (err == cudaSuccess) {
    return;
  } 

  std::cerr << statement << " returned " << cudaGetErrorString(err) << " (" << err << ") at " << file << ":" << line << std::endl;
  exit (EXIT_FAILURE);
}

/**
 * @brief Function to start the timer
 * 
 * @param start 
 * @param stop 
 */
void startTimer(cudaEvent_t *start, cudaEvent_t *stop) {
  CUDA_CHECK_RETURN(cudaEventCreate(start));
  CUDA_CHECK_RETURN(cudaEventCreate(stop));
  CUDA_CHECK_RETURN(cudaEventRecord(*start));
}

/**
 * @brief Function to stop the timer and report the elapsed time
 * 
 * @param start 
 * @param stop 
 * @return float 
 */
float stopAndPrintTimer(cudaEvent_t *start, cudaEvent_t *stop) {
  CUDA_CHECK_RETURN(cudaEventRecord(*stop));
  CUDA_CHECK_RETURN(cudaEventSynchronize(*stop));

  float milliseconds = 0;
  CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, *start, *stop));

  CUDA_CHECK_RETURN(cudaEventDestroy(*start));
  CUDA_CHECK_RETURN(cudaEventDestroy(*stop));

  return milliseconds;
}

// ====================================================================================================

/**
 * @brief Kernel para la inicialización del vector V
 * 
 */
__global__ void initVectorV(int random, curandState *state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    curand_init(random, i, 0, &state[i]);
    vector_V[i] = curand(&state[i]) % 1000;
  }
}

/**
 * @brief Kernel para la inicialización del vector H
 * 
 */
__global__ void initVectorH() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) {
    vector_H[i] = 0;
  }
}

/**
 * @brief Kernel para el calculo del histograma con atomicAdd
 * 
 */
__global__ void histogram() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    atomicAdd(&vector_H[vector_V[i] % M], 1);
  }
}

int main() {
  srand((unsigned)time(NULL)); // Inicializar la semilla para los numeros aleatorios
  int random = rand();         // Generar un numero aleatorio
  curandState *devStates;      // Puntero al estado de curand

  int k = 1;

  // Reservar memoria en el device para el estado de curand
  CUDA_CHECK_RETURN(cudaMalloc((void **)&devStates, N * sizeof(curandState)));

  int host_vector_H[M];           // Vector H en el host
  float elapsedTime[REPETITIONS]; // Array para almacenar los tiempos de ejecución

  std::cout << "Numero de elementos en el vector V: " << N << std::endl;
  std::cout << "Numero de elementos o cajas en el histograma: " << M << std::endl;

  for (int i = 0; i < REPETITIONS; i++) {
    // Inicializar el vector V
    initVectorV<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(random, devStates);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Inicializar el vector H
    initVectorH<<<1, M>>>();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Para calcular el tiempo de ejecución
    cudaEvent_t start, stop;
    startTimer(&start, &stop);

    // Calcular el histograma
    histogram<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Detener el temporizador y almacenar el tiempo
    elapsedTime[i] = stopAndPrintTimer(&start, &stop);

    // Copiar el vector H al host
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(host_vector_H, vector_H, M * sizeof(int)));

    int acum = 0;
    std::cout << "\nVector H " << k++ << ": \t| ";
    for (int j = 0; j < M; j++) {
      std::cout << host_vector_H[j] << " ";
      acum += host_vector_H[j];
    }
    std::cout << "| \nTotal: " << acum << std::endl;
    std::cout << "Tiempo de ejecución: " << elapsedTime[i] << " ms" << std::endl;
  }

  // Liberar memoria
  CUDA_CHECK_RETURN(cudaFree(devStates));

  // Calcular la media, maximo y minimo de los tiempos de ejecución
  float max = 0;
  float min = FLT_MAX;
  float mean = 0;
  for (int i = 0; i < REPETITIONS; i++) {
    mean += elapsedTime[i];
    if (elapsedTime[i] > max) {
      max = elapsedTime[i];
    }
    if (elapsedTime[i] < min) {
      min = elapsedTime[i];
    }
  }

  std::cout << "\n\n==================== RESULTADOS Histograma 1 ====================\n";
  std::cout << "Numero de elementos en el vector V: " << N << std::endl;
  std::cout << "\nSe ha hecho " << REPETITIONS << " repeticiones" << std::endl;
  std::cout << "Tiempo medio:  " << mean / REPETITIONS << " ms" << std::endl;
  std::cout << "Tiempo maximo: " << max << " ms" << std::endl;
  std::cout << "Tiempo minimo: " << min << " ms" << std::endl;

  return 0;
}