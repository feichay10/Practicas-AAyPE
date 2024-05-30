/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * @file histograma.cu
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Version 1 histograma en CUDA: Realizar un histograma de un vector V de
 * un número elevado N de elementos enteros aleatorios. El histograma consiste 
 * en un vector H que tiene M elementos que representan "cajas". En cada caja se 
 * cuenta el número de veces que ha aparecido un elemento del vector V con el 
 * valor adecuado para asignarlo a esa caja (normalmente cada caja representa un 
 * rango o intervalo de valores). En nuestro caso, para simplificar la asignación
 * del elemento de V a su caja correspondiente del histograma, vamos a realizar 
 * la operación ValorElementoV Módulo M, que nos da directamente el índice de la 
 * caja del histograma a la que pertenecerá ese elemento y cuyo contenido deberemos 
 * incrementar. Se sugiere como N un valor del orden de millones de elementos y 
 * como M, 8 cajas.
 * 
 * Como implementación base (que podremos mejorar en tiempo o no) se pide crear 
 * tantos hilos como elementos de V para que cada uno se encargue de ir
 * al elemento que le corresponda en V e incremente la caja correcta en el
 * vector histograma H (posiblemente de forma atómica).
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

#define N 500000000         // Numero de elementos en el vector V
#define M 8                 // Numero de elementos o cajas en el histograma (tamaño del histograma)
#define REPETITIONS 20      // Numero de repeticiones para el calculo de la media, max y min

__device__ int vector_V[N]; // Vector V de un numero elevado N de elementos enteros aleatorios
__device__ int vector_H[M]; // Vector H que tiene M elementos que representan "cajas"

// Function to start the timer
void startTimer(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventCreate(start);
    cudaEventCreate(stop);
    cudaEventRecord(*start);
}

// Function to stop the timer and report the elapsed time
void stopAndPrintTimer(cudaEvent_t *start, cudaEvent_t *stop) {
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, *start, *stop);

    printf("Time spent in kernel: %f ms\n", milliseconds);

    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
}

// ==========================================================================

/**
 * @brief Funcion para la comprobación de errores de CUDA
 *
 */
static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

/**
 * @brief Kernel para la inicialización del vector V
 * 
 */
__global__ void initVectorV(int random, curandState *state) {
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  curandState *localState = state + id_x;
  curand_init(random, id_x, 0, localState);

  for (int i = 0; i < N; i++) {
    if (id_x == i) {
      vector_V[i] = curand(localState);
    }
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
 * @brief Kernel para el calculo del histograma
 * 
 */
__global__ void histogram(int threadsPerBlock, int blocksPerGrid) {
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < N; i++) {
    if (id_x == i) {
      int mod = vector_V[id_x * i] % M;
      atomicAdd(&vector_H[mod], 1);
    }
  }
}



int main() {
  srand(time(NULL)); // Inicializar la semilla para los numeros aleatorios
  int random = rand();    // Generar un numero aleatorio
  static curandState *devStates = NULL;
  int threadsPerBlock = 1024;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // int host_vector_V[N];
  int host_vector_H[M];

  float elapsedTime[REPETITIONS]; // Array para almacenar los tiempos de ejecución

  // Para calcular el tiempo de ejecución
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Numero de elementos en el vector V: " << N << std::endl;

  for (int i = 0; i < REPETITIONS; i++) {
    CUDA_CHECK_RETURN(cudaMalloc((void **)&devStates, N * sizeof(curandState) * threadsPerBlock * blocksPerGrid));
    // Inicializar el vector V
    initVectorV<<<blocksPerGrid, threadsPerBlock>>>(random, devStates);

    // Inicializar el vector H
    initVectorH<<<blocksPerGrid, threadsPerBlock>>>();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Esperar a que el kernel termine

    // Lanzar el kernel
    // startTimer(&start, &stop);
    histogram<<<blocksPerGrid, threadsPerBlock>>>(threadsPerBlock, blocksPerGrid);
    // stopAndPrintTimer(&start, &stop);

    // Copiar el vector H al host
    cudaMemcpyFromSymbol(host_vector_H, vector_H, M * sizeof(int));

    int acum = 0;
    for (int i = 0; i < M; i++) {
      std::cout << "H[" << i << "] = " << host_vector_H[i] << std::endl;
      acum += host_vector_H[i];
    }

    std::cout << "Acumulado: " << acum << std::endl;

    cudaFree(devStates);
  }

  float t_max = 0, t_min = FLT_MAX, t_mean = 0;
  for (int i = 0; i < REPETITIONS; i++) {
    t_mean += elapsedTime[i];
    if (elapsedTime[i] > t_max) {
      t_max = elapsedTime[i];
    }
    if (elapsedTime[i] < t_min) {
      t_min = elapsedTime[i];
    }
  }

  std::cout << "Se ha hecho: " << REPETITIONS << " repeticiones" << std::endl;
  std::cout << "Con una media de: " << t_mean / REPETITIONS << " ms" << std::endl;
  std::cout << "Con un maximo de: " << t_max << " ms" << std::endl;
  std::cout << "Con un minimo de: " << t_min << " ms" << std::endl;

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