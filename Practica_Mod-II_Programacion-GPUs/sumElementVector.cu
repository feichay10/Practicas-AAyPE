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

#define N 1000 // Numero de elementos del vector

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__device__ float suma;
__device__ long long int ti[N];
__device__ long long int tf[N];

__global__ void sumVec(float *A, int tam_vec) {
  long long int t_ini = clock64();
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // threadIdx.x == 0
  if (i == 0) {
    suma = 0.0;
  }

  if (i < tam_vec) {
    suma = suma + A[i];
  }
  tf[i] = clock64();
  ti[i] = t_ini;
}

// atomicAdd(&suma, A[i]) {
//   float old = *address, assumed;
//   do {
//     assumed = old;
//     old = atomicCAS(address, assumed, val + assumed);
//   } while (assumed != old);
// }

// 1. Ubicar h_A
// 2. Inicializar h_A
//    h_A[i] = rand() / (float)RAND_MAX
// 3. Copiar h_A a d_A
// 4. Encontrar suma de h_A
// 5. Invocar kernel
// 6. Obtener 
//     1. suma
//     2. ti[A]
//     3. tf[A]
// 7. Imprimir suma tf[i]

int main() {
  float *h_A;
  float *d_A;
  float suma = 0.0;
  long long int *h_ti;
  long long int *h_tf;
  long long int *d_ti;
  long long int *d_tf;

  h_A = (float *)malloc(N * sizeof(float));
  h_ti = (long long int *)malloc(N * sizeof(long long int));
  h_tf = (long long int *)malloc(N * sizeof(long long int));

  for (int i = 0; i < N; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
  }

  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, N * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_ti, N * sizeof(long long int)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tf, N * sizeof(long long int)));

  CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

  sumVec<<<1, N>>>(d_A, N);

  CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&suma, suma, sizeof(float), 0, cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_ti, ti, N * sizeof(long long int), 0, cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_tf, tf, N * sizeof(long long int), 0, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    std::cout << "ti[" << i << "] = " << h_ti[i] << std::endl;
    std::cout << "tf[" << i << "] = " << h_tf[i] << std::endl;
  } 

  std::cout << "Suma = " << suma << std::endl;

  free(h_A);
  free(h_ti);
  free(h_tf);
  cudaFree(d_A);
  cudaFree(d_ti);

  return 0;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}