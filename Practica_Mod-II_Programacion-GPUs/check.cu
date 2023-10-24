/*
 ============================================================================
 Name        : check.cu
 Ejemplo de testeo de errores
 ============================================================================
 */

#include <cuda_runtime.h>
#include <iostream>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


/**
 * Host main routine
 */
int main(void) {
	// Allocate the device input vector A
	float *d_A = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size));
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
