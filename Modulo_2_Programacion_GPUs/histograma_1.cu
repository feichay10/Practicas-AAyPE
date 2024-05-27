/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * @file histograma_1.cu
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

#define N 500000000  // Numero de valores de entrada
#define M 8          // Tamaño del histograma

// Repeticon de pruevas para calculo de media, max y min
#define REPETICONES 10000
#define SCALA 50  // Datos calculados en cada hilo

__device__ int vector_V[N];  // Vector de datos de entrada
__device__ int vector_H[M];  // Vector del histograma

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
__global__ void inicializacion_vector(int random, curandState *states,
                                      int threadsPerBlock, int blocksPerGrid) {
  int iteraciones = SCALA;
  if (blocksPerGrid - 1 == blockIdx.x && threadIdx.x == threadsPerBlock - 1) {
    iteraciones = iteraciones + (N % SCALA);
  }
  unsigned id_x = blockIdx.x * blockDim.x + threadIdx.x;
  curandState *state = states + id_x;

  curand_init(random, id_x, 0, state);
  for (int i = 0; i < iteraciones; i++) {
    if (id_x * SCALA + i < N) {
      vector_V[id_x * SCALA + i] = (int)((curand_uniform(state) * 1000)) % M;
    }
  }
}

/**
 * @brief Kernel para la inicializacion del vector del histograma
 *
 */
__global__ void inicializacion_histograma() {
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (id_x < M) {
    vector_H[id_x] = 0;
  }
}

__global__ void histograma(int threadsPerBlock, int blocksPerGrid) {
  int iteraciones = SCALA;
  if (blocksPerGrid - 1 == blockIdx.x && threadIdx.x == threadsPerBlock - 1) {
    iteraciones = iteraciones + (N % SCALA);
  }
  unsigned id_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < iteraciones; i++) {
    if (id_x * SCALA + i < N) {
      int mod = vector_V[id_x * SCALA + i] % M;
      atomicAdd(&vector_H[mod], 1);
    }
  }
}

int main() {}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}