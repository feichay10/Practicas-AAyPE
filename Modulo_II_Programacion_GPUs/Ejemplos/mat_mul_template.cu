/*
 ============================================================================
 Name        : matmul.cu
 Description : CUDA compute matrix multip
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

#define N (200)
#define M (300)
#define I (100)

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes product matrix
 */

__global__ void matMul(float *d_A, float *d_B, float *d_C,
		               unsigned d_fil, unsigned d_inner, unsigned d_col,
		               size_t pitchA, size_t pitchB, size_t pitchC)
{   
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; // Fila
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x; // Columnas

    // Cada hilo calcula un elemento de la matriz d_C
    if((row < d_fil) && (col < d_col)) {                         
        float *row_A = (float *)((char *)d_A + row * pitchA);
        float *row_C = (float *)((char *)d_C + row * pitchC);
        
        row_C[col] = 0.0;
        for (size_t i = 0; i < d_inner; i++){
            float *row_B = (float *)((char *)d_B + i * pitchB);
            row_C[col] += row_A[i] * row_B[col];
        }       
    }
}

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


//Copia resultado de GPU al host y compara con resultado obtenido en host
int cmpMat(float *d_C, size_t pitchC, float *h_C, float *h_R, float tol)
{
    CUDA_CHECK_RETURN(cudaMemcpy2D (h_C, M * sizeof(float), d_C, pitchC, M * sizeof(float), N, cudaMemcpyDeviceToHost ));

    bool Ok = true;
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < M; j++){
            if (fabs(h_C[(i * M) + j] - h_R[(i * M) + j]) > tol){
                fprintf(stderr, "Result verification failed at element (%d, %d)! GPU=%f, CPU=%f\n", i, j, h_C[(i * M) + j], h_R[(i * M) + j]);
                Ok = false;
	        }
  	    }
    }
    return Ok;
}

int main(void)
{
    cudaEvent_t start, stop;

    // Print the dimensions to be used
    printf("[Matrix multiplication of (%dx%d) X (%dx%d) matrices]\n", N, I, I, M);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(N*I*sizeof(float));
    // Allocate the host input vector B
    float *h_B = (float *)malloc(I*M*sizeof(float));
    // Allocate the host output vector C
    float *h_C = (float *)malloc(N*M*sizeof(float));
    // Allocate result verification matrix R
    float *h_R = (float *)malloc(N*M*sizeof(float));
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_R == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host matrix A
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < I; j++){
    		h_A[i*I+j] = rand()/(float)RAND_MAX;
    	}
    }
    
    // Initialize the host matrix B
    for (int i = 0; i < I; i++){
        for (int j = 0; j < M; j++){
        	h_B[i*M+j] = rand()/(float)RAND_MAX;
        }
    }

    //Compute result matrix
    for (int i = 0; i < N; i++){
    	for (int j = 0; j < M; j++){
           float acc = 0.0;
           for (int k = 0; k < I; k++){
    		acc = acc + (h_A[(i * I) + k] * h_B[(k * M) + j]);
           }
           h_R[(i * M) + j] = acc;
    	}
    }


    // Allocate the device input matrix A
    float *d_A = NULL;
    size_t pitchA;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_A, &pitchA, I * sizeof(float), N));
    printf("[Pitch of A is %d bytes, Width is %d elements, Width in bytes is %d and Height is %d]\n", pitchA, I, I*sizeof(float), N);

    // Allocate the device input matrix B
    float *d_B = NULL;
    size_t pitchB;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_B, &pitchB, M * sizeof(float), I));
    printf("[Pitch of B is %d bytes, Width is %d elements, Width in bytes is %d and Height is %d]\n", pitchB, M, M*sizeof(float), I);

    // Allocate the device output matrix C
    float *d_C = NULL;
    size_t pitchC;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_C, &pitchC, M * sizeof(float), N));
    printf("[Pitch of C is %d bytes, Width is %d elements, Width in bytes is %d and Height is %d]\n", pitchC, M, M*sizeof(float), N);


    // Copy the host input matrices A and B in host memory to the device input matrices in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_A, pitchA, h_A, I * sizeof(float), I * sizeof(float), N, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_B, pitchB, h_B, M * sizeof(float), M * sizeof(float), I, cudaMemcpyHostToDevice ));

    // Launch the Matrix product CUDA Kernel
    dim3 threadsPerBlock1(32, 32); // RELLENAR ADECUADAMENTE
    dim3 blocksPerGrid1((M + threadsPerBlock1.x - 1) / threadsPerBlock1.x, (N + threadsPerBlock1.y - 1) / threadsPerBlock1.y); // RELLENAR ADECUADAMENTE
    printf("CUDA kernel launch with (%d, %d) blocks of (%d, %d) threads\n", blocksPerGrid1.x, blocksPerGrid1.y, threadsPerBlock1.x, threadsPerBlock1.y);
    startTimer(&start, &stop);
    matMul<<<blocksPerGrid1, threadsPerBlock1>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)){
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }
    
   
    // Free device global memory
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_R);

    printf("Done\n");
    return EXIT_SUCCESS;

}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}
