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
#include <cuda_runtime.h>

#define N (200)
#define M (300)
#define I (500)
#define BSIZE (32)

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernels that compute product matrix
 */


/*
dim3 threadsPerBlock1(16,16);
dim3 blocksPerGrid1( (M + threadsPerBlock1.x - 1) / threadsPerBlock1.x , (N + threadsPerBlock1.y - 1) / threadsPerBlock1.y );
MxN hilos distribuidos en bloques de (p x q) hilos, cada hilo encargado de calcular un elemento de la matriz C
*/

__global__ void matMul1(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //NxM hilos distribuidos en bloques de (n x n) hilos
	unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned id_y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ( (id_x < d_col) && (id_y < d_fil) ){
        float acc = 0.0;
        //inicializa los punteros a las filas de A y C
        float *rowa = (float *)((char *)d_A + id_y * pitchA);
		float *rowc = (float *)((char *)d_C + id_y * pitchC);
		for (int k = 0; k < d_inner; k++){
			float *rowb = (float *)((char *)d_B + k * pitchB);
			acc = acc + (rowa[k] * rowb[id_x]);
		}
		rowc[id_x] = acc;
	}
}

/*
dim3 threadsPerBlock1x(16,16);
dim3 blocksPerGrid1x( (M + threadsPerBlock1x.x - 1) / threadsPerBlock1x.x , (N + threadsPerBlock1x.y - 1) / threadsPerBlock1x.y );
MxN hilos distribuidos en bloques de (p x q) hilos, con punteros, cada uno encargado de un elemento de la matriz C
*/

__global__ void matMul1x(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //MxN hilos distribuidos en bloques de (m x n) hilos, con punteros
	unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned id_y = blockIdx.y*blockDim.y + threadIdx.y;

	if ( (id_x < d_col) && (id_y < d_fil) ) {
		float acc = 0.0;
		float *elemc = (float *)((char *)d_C + id_y * pitchC) + id_x;
		float *elema = (float *)((char *)d_A + id_y * pitchA); //Base de fila de A
		float *elemb = d_B + id_x; //Primer elemento de la columna de B
		for (int k = 0; k < d_inner; k++) {
			acc = acc + (*elema++) * (*elemb);
			elemb = (float *)((char *)elemb + pitchB); //Avanza el puntero por la columna de B
		}
		*elemc = acc;
	}
}

/*
    int threadsPerBlock2 = I;
    dim3 blocksPerGrid2(N, M);
    se lanzan NxM bloques de I hilos, cada hilo hace su producto de dos elementos y
    lo suma de forma atómica a una variable compartida, el hilo 0 la escribe en C
*/
 
__global__ void matMul2(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
	//NxM bloques de I hilos
	unsigned i = blockIdx.x;
	unsigned j = blockIdx.y;
	unsigned k = threadIdx.x;

	__shared__ float acc;

	if (k == 0){
		acc = 0.0;
	}
	__syncthreads();
	float el_a = *((float *)((char *)d_A + i * pitchA) + k);
	float el_b = *((float *)((char *)d_B + k * pitchB) + j);
	atomicAdd(&acc, el_a * el_b);
	__syncthreads();
	if (k == 0){
	  *((float *)((char *)d_C + i * pitchC) + j) = acc;
	}
}


/*   int threadsPerBlock3 = I;
    dim3 blocksPerGrid3(N, M);
    Se lanzan NxM bloques de I hilos, cada uno suma su producto de dos elementos 
    en el elemento de la matriz C de forma atómica
*/

__global__ void matMul3(float *d_A, float *d_B, float *d_C,
		    unsigned d_fil, unsigned d_inner, unsigned d_col,
		    size_t pitchA, size_t pitchB, size_t pitchC)
{
	//NxM bloques de I hilos
	unsigned i = blockIdx.x;
	unsigned j = blockIdx.y;
	unsigned k = threadIdx.x;

	float *el_c = (float *)((char *)d_C + i * pitchC) + j; //puntero a elemento a escribir
	if (k == 0){
		  *el_c = 0.0;
	}
	__syncthreads(); //espera a que se haya inicializado
	float el_a = *((float *)((char *)d_A + i * pitchA) + k);
	float el_b = *((float *)((char *)d_B + k * pitchB) + j);
	atomicAdd(el_c, el_a * el_b); //Todos los hilos escriben secuencialmente
}

/*
    int threadsPerBlock4 = I;
    int blocksPerGrid4 = M;
    Se lanzan M bloques de I hilos (un bloque por columna de B), colaboran cada uno leyendo un elemento de la columna de B,
    y van multiplicando el elemento por el correspondiente de A sec. y suman atomic en var compartida
*/

__global__ void matMul4(float *d_A, float *d_B, float *d_C,
    unsigned d_fil, unsigned d_inner, unsigned d_col,
    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //M bloques de I hilos (columnas de B)
    unsigned j = blockIdx.x;
    unsigned k = threadIdx.x;
    __shared__ float acc;

    float el_b = *((float *)((char *)d_B + k * pitchB) + j); //Obtiene el elemento de B asociado a este hilo
    for (int i = 0; i < N; i++) {
        if (k == 0){
            acc = 0.0;
        }
        __syncthreads(); //espera a que se haya inicializado
        float el_a = *((float *)((char *)d_A + i * pitchA) + k);
        float *rowc = (float *)((char *)d_C + i * pitchC);
        atomicAdd(&acc, el_a * el_b); //Todos los hilos escriben ordenadamente
        __syncthreads(); //espera finalizar
        if (k == 1){
            rowc[j] = acc;
        }
    }
}


/*
    int threadsPerBlock4b = I;
    int blocksPerGrid4b = M;
    Se lanzan M bloques de I hilos (un bloque por columna de B), colaboran cada uno leyendo un elemento de la columna de B,
    y van multiplicando el elemento por el correspondiente de A sec. y suman atomic elem final de C
*/

__global__ void matMul4b(float *d_A, float *d_B, float *d_C,
    unsigned d_fil, unsigned d_inner, unsigned d_col,
    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //M bloques de I hilos (columnas de B)
    unsigned j = blockIdx.x;
    unsigned k = threadIdx.x;

    float el_b = *((float *)((char *)d_B + k * pitchB) + j); //Obtiene el elemento de B asociado a este hilo
    for (int i = 0; i < N; i++) {
        float *rowc = (float *)((char *)d_C + i * pitchC);
        if (k == 0){
            rowc[j] = 0.0;
        }
        __syncthreads(); //espera a que se haya inicializado
        float el_a = *((float *)((char *)d_A + i * pitchA) + k); 
        atomicAdd(&rowc[j], el_a * el_b); //Todos los hilos escriben ordenadamente
    }
}



/*
    int threadsPerBlock5 = I;
    int blocksPerGrid5 = N;
    Se lanzan N bloques de I hilos (un bloque por fila de A), colaboran cada uno leyendo un elemento de la fila de A,
    y van multiplicando el elemento por el correspondiente de B sec. y suman atomic en var compartida
*/

__global__ void matMul5(float *d_A, float *d_B, float *d_C,
    unsigned d_fil, unsigned d_inner, unsigned d_col,
    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //N bloques de I hilos (filas de A)
    unsigned i = blockIdx.x;
    unsigned k = threadIdx.x;
    __shared__ float acc;

    float el_a = *((float *)((char *)d_A + i * pitchA) + k); //Obtiene el elemento de A asociado a este hilo
    float *rowc = (float *)((char *)d_C + i * pitchC); //fila de C
    for (int j = 0; j < M; j++) { //Iteramos por las M columnas de B
        if (k == 0){
            acc = 0.0;
        }
        __syncthreads(); //espera a que se haya inicializado
        float el_b = *((float *)((char *)d_B + k * pitchB) + j);
        atomicAdd(&acc, el_a * el_b); //Todos los hilos escriben secuencialmente
        __syncthreads(); //espera finalizar
        if (k == 0){
            rowc[j] = acc;
        }
    }
}


/*
    int threadsPerBlock5b = I;
    int blocksPerGrid5b = N;
    Se lanzan N bloques de I hilos, colaboran cada uno leyendo un elemento de la fila de A, 
    y van multiplicando el elemento por el correspondiente de B sec. y suman atomic en elemento de C corresp.
*/

__global__ void matMul5b(float *d_A, float *d_B, float *d_C,
    unsigned d_fil, unsigned d_inner, unsigned d_col,
    size_t pitchA, size_t pitchB, size_t pitchC)
{
    //N bloques de I hilos (filas de A)
    unsigned i = blockIdx.x;
    unsigned k = threadIdx.x;
    //__shared__ float acc;

    float el_a = *((float *)((char *)d_A + i * pitchA) + k); //Obtiene el elemento de A asociado a este hilo
    float *rowc = (float *)((char *)d_C + i * pitchC); //fila de C
    float *rowb = (float *)((char *)d_B + k * pitchB); //fila de B
    for (int j = 0; j < M; j++) { //Iteramos por las M columnas de B
        if (k == 0){
            rowc[j] = 0.0;
        }
        __syncthreads(); //espera a que se haya inicializado
        atomicAdd(&rowc[j], el_a * rowb[j]); //Todos los hilos escriben secuencialmente
    }
}

/*
    dim3 threadsPerBlock6(BSIZE, BSIZE);
    dim3 blocksPerGrid6( (M + threadsPerBlock6.x - 1) / threadsPerBlock6.x ,
     (N + threadsPerBlock6.y - 1) / threadsPerBlock6.y );
 
    Se lanzan M x N bloques de (BSIZE x BSIZE), cada bloque es responsable del calculo de submatriz C de
    tamaño (BSIZE x BSIZE) multiplicando cada elemento de una macrofila de A compuesta por I/BSIZE submatrices
    por cada elemento de una macrocolumna de B compuesta por I/BSIZE submatrices. Cada uno de estos productos de 
    submatrices se acumula en la submatriz de C. Cada una de las dos submatrices a multiplicar se copia a memoria 
    compartida y cada hilo del bloque realiza el cálculo parcial de un elemento de la submatriz C y lo suma en ella
*/
__global__ void matMul6(float *d_A, float *d_B, float *d_C,
    unsigned d_fil, unsigned d_inner, unsigned d_col,
    size_t pitchA, size_t pitchB, size_t pitchC)
{
    __shared__ float As[BSIZE][BSIZE];
    __shared__ float Bs[BSIZE][BSIZE];
    // Fila y columna en bloques Csub
    int bRow = blockIdx.y;
    int bCol = blockIdx.x;
    // Fila y columna dentro de Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    int colC = bCol * BSIZE + col;
    int rowC = bRow * BSIZE + row;
    float *elemC = (float *)((char *)d_C + rowC * pitchC) + colC;
    if (rowC < d_fil && colC < d_col) {
    // Each thread zeroes its element of Csub
        *elemC = 0.0;
    } else {
        elemC = NULL;
    }
    __syncthreads();
    // Iterar por las submatrices of A and B
    // que multiplicadas y sumadas dan Csub
    int blockInner = (d_inner + BSIZE - 1) / BSIZE;
    for (int m = 0; m < blockInner; m++) {
        // Obtener Asub de A y Bsub de B copiandolas en locales
        int colA = m * BSIZE + col;
        int rowB = m * BSIZE + row;
        As[row][col] = (colA < I) ? *((float *)((char *)d_A + (bRow * BSIZE + row) * pitchA) + colA) : 0.0;
        Bs[row][col] = (rowB < I) ? *((float *)((char *)d_B + rowB * pitchB) + bCol * BSIZE + col) : 0.0;
        __syncthreads(); //Esperar a acabar la copia local de Asub y Bsub
        // Multiplicar Asub and Bsub
        for (int e = 0; e < BSIZE; e++)
            if (elemC != NULL)
                *elemC += As[row][e] * Bs[e][col];    
        __syncthreads(); //acabar la acumulación en Csub del producto antes de pasar al siguiente
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
    // Print the matrices dimensions
    printf("[Matrix multiplication of (%dx%d) X (%dx%d) matrices]\n\n", N, I, I, M);

    // Allocate the host input matrix A
    float *h_A = (float *)malloc(N*I*sizeof(float));
    // Allocate the host input matrix B
    float *h_B = (float *)malloc(I*M*sizeof(float));
    // Allocate the host output matrix C
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
    //printf("[Pitch of A is %d, Width is %d, Width in bytes is %d and height is %d]\n", pitchA, I, I*sizeof(float), N);


    // Allocate the device input matrix B
    float *d_B = NULL;
    size_t pitchB;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_B, &pitchB, M * sizeof(float), I));

    // Allocate the device output matrix C
    float *d_C = NULL;
    size_t pitchC;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_C, &pitchC, M * sizeof(float), N));

    // Copy the host input matrices A and B in host memory to the device input matrices in
    // device memory
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_A, pitchA, h_A, I * sizeof(float), I * sizeof(float), N, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy2D ( d_B, pitchB, h_B, M * sizeof(float), M * sizeof(float), I, cudaMemcpyHostToDevice ));


    // Launch the Matrix product CUDA Kernels
    dim3 threadsPerBlock1(32,32);
    dim3 blocksPerGrid1( (M + threadsPerBlock1.x - 1) / threadsPerBlock1.x , (N + threadsPerBlock1.y - 1) / threadsPerBlock1.y );
    printf("CUDA kernel 1 launch with (%d, %d) blocks of (%d, %d) threads\n", blocksPerGrid1.x, blocksPerGrid1.y, threadsPerBlock1.x, threadsPerBlock1.y);
    startTimer(&start, &stop);
    matMul1<<<blocksPerGrid1, threadsPerBlock1>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    dim3 threadsPerBlock1x(64,8);
    dim3 blocksPerGrid1x( (M + threadsPerBlock1x.x - 1) / threadsPerBlock1x.x , (N + threadsPerBlock1x.y - 1) / threadsPerBlock1x.y );
    printf("CUDA kernel 1x launch with (%d, %d) blocks of (%d, %d) threads\n", blocksPerGrid1x.x, blocksPerGrid1x.y, threadsPerBlock1x.x, threadsPerBlock1x.y);
    startTimer(&start, &stop);
    matMul1x<<<blocksPerGrid1x, threadsPerBlock1x>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    int threadsPerBlock2 = I;
    dim3 blocksPerGrid2(N, M);
    printf("CUDA kernel 2 launch with (%d, %d) blocks of %d threads\n", blocksPerGrid2.x, blocksPerGrid2.y, threadsPerBlock2);
    startTimer(&start, &stop);
    matMul2<<<blocksPerGrid2, threadsPerBlock2>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    int threadsPerBlock3 = I;
    dim3 blocksPerGrid3(N, M);
    printf("CUDA kernel 3 launch with (%d, %d) blocks of %d threads\n", blocksPerGrid3.x, blocksPerGrid3.y, threadsPerBlock3);
    startTimer(&start, &stop);
    matMul3<<<blocksPerGrid3, threadsPerBlock3>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    int threadsPerBlock4 = I;
    int blocksPerGrid4 = M;
    printf("CUDA kernel 4 launch with %d blocks of %d threads\n", blocksPerGrid4, threadsPerBlock4);
    startTimer(&start, &stop);
    matMul4<<<blocksPerGrid4, threadsPerBlock4>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    int threadsPerBlock4b = I;
    int blocksPerGrid4b = M;
    printf("CUDA kernel 4b launch with %d blocks of %d threads\n", blocksPerGrid4b, threadsPerBlock4b);
    startTimer(&start, &stop);
    matMul4b<<<blocksPerGrid4b, threadsPerBlock4b>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    int threadsPerBlock5 = I;
    int blocksPerGrid5 = N;
    printf("CUDA kernel 5 launch with %d blocks of %d threads\n", blocksPerGrid5, threadsPerBlock5);
    startTimer(&start, &stop);
    matMul5<<<blocksPerGrid5, threadsPerBlock5>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    int threadsPerBlock5b = I;
    int blocksPerGrid5b = N;
    printf("CUDA kernel 5b launch with %d blocks of %d threads\n", blocksPerGrid5b, threadsPerBlock5b);
    startTimer(&start, &stop);
    matMul5b<<<blocksPerGrid5b, threadsPerBlock5b>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

    printf("\n");

    dim3 threadsPerBlock6(BSIZE, BSIZE);
    dim3 blocksPerGrid6( (M + threadsPerBlock6.x - 1) / threadsPerBlock6.x ,
     (N + threadsPerBlock6.y - 1) / threadsPerBlock6.y );
    printf("CUDA kernel 6 launch with (%d, %d) blocks of (%d, %d) threads\n", blocksPerGrid6.x, blocksPerGrid6.y, threadsPerBlock6.x, threadsPerBlock6.y);
    startTimer(&start, &stop);
    matMul6<<<blocksPerGrid6, threadsPerBlock6>>>(d_A, d_B, d_C, N, I, M, pitchA, pitchB, pitchC);
    CUDA_CHECK_RETURN(cudaGetLastError());
    stopAndPrintTimer(&start, &stop);
    if (cmpMat(d_C, pitchC, h_C, h_R, 1.0e-3)) printf("Test PASSED\n");

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
