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

#define N 500000000    // Numero de elementos en el vector V
#define M 8            // Numero de elementos o cajas en el histograma
#define REPETITIONS 20 // Numero de repeticiones para el calculo de la media


/**
 * @brief Funcion para la comprobación de errores de CUDA
 *
 */
static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

/**
 * @brief Kernel para el calculo del histograma
 * 
 */
__global__ void kernel(int *vector, int *histogram) {
  int id_x = threadIdx.x + blockIdx.x * blockDim.x;
  int pos_histogram = 0;
  if (id_x < N) {
    pos_histogram = vector[id_x] % M; // ValorElementoV mod M
    atomicAdd(&histogram[pos_histogram], 1);
  }
}

/**
 * @brief Kernel para la inicializacion del histograma
 * 
 * @param threadsPerBlock 
 * @param blocksPerGrid 
 * @return __global__ 
 */
__global__ void histogram(int *histogram) {
  int id_x = threadIdx.x + blockIdx.x * blockDim.x;
  if (id_x < N) {
    if (id_x == 0) {
      for (int i = 0; i < M; i++) {
        histogram[i] = 0;
      }
    }
  }
}


int main() {
  // Realizar un histograma de un vector V de un número elevado N de elementos
  // enteros aleatorios. El histograma consiste en un vector H que tiene M
  // elementos que representan "cajas".

  // En cada caja se cuenta el número de veces que ha aparecido un elemento del
  // vector V con el valor adecuado para asignarlo a esa caja (normalmente cada
  // caja representa un rango o intervalo de valores). En nuestro caso, para
  // simplificar la asignación del elemento de V a su caja correspondiente del
  // histograma, vamos a realizar la operación ValorElementoV Módulo M, que nos
  // da directamente el índice de la caja del histograma a la que pertenecerá
  // ese elemento y cuyo contenido deberemos incrementar. Se sugiere como N un
  // valor del orden de millones de elementos y como M, 8 cajas.

  // Como implementación base (que podremos mejorar en tiempo o no) se pide
  // crear tantos hilos como elementos de V para que cada uno se encargue de ir
  // al elemento que le corresponda en V e incremente la caja correcta en el
  // vector histograma H (posiblemente de forma atómica).

  // Inicializacion de los vectores
  int *h_vector = (int *)malloc(N * sizeof(int));
  int *h_histogram = (int *)malloc(M * sizeof(int));

  // Inicializacion de los vectores con valores aleatorios
  for (int i = 0; i < N; i++) {
    h_vector[i] = (int)rand() / (int)RAND_MAX;
  }

  // Inicializacion de los vectores en la GPU
  int *d_vector;
  int *d_histogram;
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_vector, N * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_histogram, M * sizeof(int)));

  // Bloques e hilos
  int threadPerBlock = 256;
  int blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;



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