/*
 ============================================================================
 Name        : sum_vec_template.cu
 Ejemplo de suma de elementos de un vector con operaciones atómicas, medida
 de tiempos con clock64() y uso de variables ubicadas directamente en memoria
 global de la GPU. No hay garantías de corrección si se inicializa la suma así
 y hay más de un bloque de hilos por la sincronización entre bloques
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


//Num. de elementos a sumar, no debería exceder el numero de hilos por bloque para sincronizar la puesta a cero

#define N (4097000)
#define HILOSPORBLOQUE (16)

__device__ float d_suma; //suma ubicada directamente en la GPU, se puede inicializar aqui con cudaMemcpyToSymbol
__device__ long long int d_ti[N];
__device__ long long int d_tf[N];

long long int h_ti[N];
long long int h_tf[N];

void findMinMaxTimes(){
    long long int t, tmin = LLONG_MAX, tmax = 0;
    int itmin, itmax;
    // Copia los tiempos en variables accesibles por el host
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_ti, d_ti, N*sizeof(long long int)));
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_tf, d_tf, N*sizeof(long long int)));
    for (int i = 0; i < N; i++) {
    	t = h_tf[i] - h_ti[i];
        if (h_tf[i] < h_ti[i]){
        	printf("Error en tiempos %u\n", i);
        }
        if (t < tmin){
          	tmin = t;
          	itmin = i;
        }
        if (t > tmax){
            tmax = t;
            itmax=i;
    	}
    }
    printf("El tiempo máximo es %lld ciclos en hilo %d y el mínimo %lld ciclos en hilo %d\n\n",tmax, itmax, tmin, itmin);
}


/**
 * CUDA Kernel Device code
 *
 * Computes the sum of elements of vector A 
 */

// Versión con suma atómica
__global__ void
sumVec_v1(const float *A, int numElements) {

	long long int tini = clock64();

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
    	atomicAdd(&d_suma,A[i]);
        d_tf[i] = clock64();
        d_ti[i] = tini;
    }
}

// Versión con reducción dentro del bloque con espacio local
__global__ void
sumVec_v2(const float *A, int numElements) {

	long long int tini = clock64();

    // índice hilo en el vector
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // índice del hilo en el bloque
    int tid = threadIdx.x;

    // Copiar trozo del vector a memoria compartida
    __shared__ float s_data[HILOSPORBLOQUE];
    s_data[tid] = (i < numElements) ? A[i] : 0.0f;

    // Espera a acabar copia
    __syncthreads();

    // Iteraciones de reducción s es el numero de hilos que van a trabajar
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Sincronizar en cada iteración para asegurar que todas las sumas estén completas
        __syncthreads();
    }

    // El hilo cero escribe el resultado
    if (tid == 0) {
        atomicAdd(&d_suma, s_data[0]);
    }
    d_tf[i] = clock64();
    d_ti[i] = tini;
}

// Version de sincronización de hilos en un grid
__global__ void
sumVec_v3(const float *A, int numElements)  {
    cg::grid_group grid = cg::this_grid();
    int i = grid.thread_rank();
    float d_suma = 0.0;

    if (i < numElements) {
        if (i == 0) {
            d_suma = 0.0;
        }
        grid.sync();
        if (i < numElements) {
            atomicAdd(&d_suma, A[i]);
        }
    }
}

/**
 * Host main routine
 */

int main(void) {

    // Vector length to be used, and compute its size
    const int numElements = N;
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    //float min_A = 1.0;
    //int min_i;
    for (int i = 0; i < numElements; i++) {
        //h_A[i] = rand()/(float)RAND_MAX; //inicializa a valores en [0.0, 1.0]
    	h_A[i] = (float)i;
        //printf("A[%u] = %f\n", i, h_A[i]);
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size));

    // Copy the host input vector A in host memory to the device input vector in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Obtiene un puntero usable desde kernel, aunque en este ejemplo no se usa. Atención al typecast!
    //float* d_sum;
    //CUDA_CHECK_RETURN(cudaGetSymbolAddress((void **)&d_sum, d_suma));

    // Suma inicializada a cero, se debe copiar a la variable en device antes de cada ejecución
    float sumini = 0.0f;
    float sumatotal;

    // Launch the sumVec CUDA Kernels
    int threadsPerBlock = HILOSPORBLOQUE;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Suma atomica
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_suma, &sumini, sizeof(float)));
    printf("Kernel con N hilos y suma atómica\n");
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    sumVec_v1<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    CUDA_CHECK_RETURN(cudaGetLastError());
    // Copia el resultado en sumatotal
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&sumatotal, d_suma, sizeof(float)));
    printf("Suma = %f\n", sumatotal);
    findMinMaxTimes();

    // Suma con reducción en bloque
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_suma, &sumini, sizeof(float)));
    printf("Kernel con reducción en bloque\n");
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    sumVec_v2<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    CUDA_CHECK_RETURN(cudaGetLastError());
    // Copia el resultado en sumatotal
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&sumatotal, d_suma, sizeof(float)));
    printf("Suma = %f\n", sumatotal);
    findMinMaxTimes();

    // Suma con sincronización de hilos en grid
    // Suma con sincronización de hilos en grid
    printf("Kernel con sincronización de hilos en grid\n");
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    sumVec_v3<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    CUDA_CHECK_RETURN(cudaGetLastError());
    // Copia el resultado en sumatotal
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&sumatotal, d_suma, sizeof(float)));
    printf("Suma = %f\n", sumatotal);
    findMinMaxTimes();


    
    // Verificar que el resultado es correcto, en general se acumula mucho error en tipo float y en N grande puede ser mucho,
    // especialmente para valores sumados pequeños (pudiendo no ser conmutativa la suma, por eso se suman en dos órdenes diferentes)
    float sumcheck1 = 0.0;
    float sumcheck2 = 0.0;
    for (int i = 0; i < numElements; i++) {
    	sumcheck1 += h_A[i];
    	sumcheck2 += h_A[numElements - i - 1];
    }
    printf("Sumas de comprobación %f y %f\n", sumcheck1, sumcheck2);
    //printf("Suma teórica aproximada de muchos números aleatorios entre 0 y 1: %f\n", (double)N * 0.5);
    printf("Suma teórica de 0 a %d números naturales: %lld\n", N-1, (long long int)N*(N-1)/2);

    /*
    if ( (fabs(sumcheck1-sumatotal) > 1e-1) || (fabs(sumcheck2-sumatotal) > 1e-1) ){
        fprintf(stderr, "Result verification failed, da %f y los checks son %lf y %lf, errores de %lf y %lf\n", sumatotal, sumcheck1, sumcheck2, sumatotal - sumcheck1, sumatotal - sumcheck2);
        exit(EXIT_FAILURE);
    } else {
    	printf("Test PASSED\n");
    }
    */
    
    // Free device global memory
    CUDA_CHECK_RETURN(cudaFree(d_A));

    // Free host memory
    free(h_A);

    printf("Done\n");
    return EXIT_SUCCESS;
}

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
