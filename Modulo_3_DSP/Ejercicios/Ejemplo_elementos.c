/*
 * ejercicio_pract.c
 *
 *      Author: Sergio Díaz González
 */


#include <cstdio>
#include "Ejemplo_elementos.h"
#include <c6x.h> // defines _itoll, TSCH, TSCL

#define Size 30

void example(){
	short x[] = {2, 3, 2, 3, 4, 1, 0, 2, 9, 10,
				  5, 4, 3, 2, 3, 4, 4, 10, 13, 20,
				  0, 1, 2, 4, 2, 1, 5, 2, 5, 6};
	short y[] = {1, 2, 3, 4, 5, 6, 7, 8 ,9, 10,
				  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
				  21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

	short arr1[Size];
	printf("\n\n\nINICIO PRUEBA\n\n\n");

	// In the variable declaration portion of the code:
	uint64_t start_time, end_time, overhead, cyclecount;

	// In the initialization portion of the code:
	start_time = _itoll(TSCH, TSCL);
	end_time = _itoll(TSCH, TSCL);
	overhead = end_time-start_time; //Calculating the overhead of the method.
	// Code to be profiled

	// Suma de vectores no optimizada
	printf("\nSuma no optimizada\n");
	start_time = _itoll(TSCH, TSCL);
	vecsum(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma no optimizada cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);
	/*
	// Suma de vectores restric
	printf("\nSuma rescrict\n");
	set_zeros(arr1, Size);
	start_time = _itoll(TSCH, TSCL);
	vecsum_restrict(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma rescrict cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);

	// Suma de vectores restric y const
	printf("\nSuma rescrict\n");
	set_zeros(arr1, Size);
	start_time = _itoll(TSCH, TSCL);
	vecsum_const_restrict(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma rescrict y const cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);
	*/
	/*

	// Suma de vectores add
	printf("\nSuma add2\n");
	set_zeros(arr1, Size);
	start_time = _itoll(TSCH, TSCL);
	vecsum_add(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma add cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);

	// Suma de vectores must
	printf("\nSuma must\n");
	set_zeros(arr1, Size);
	start_time = _itoll(TSCH, TSCL);
	vecsum_must(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma must cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);
	*/

	/*
	// Suma de vectores must desenrollado
	printf("\nSuma must desenrollado\n");
	set_zeros(arr1, Size);
	start_time = _itoll(TSCH, TSCL);
	vecsum_must_desenrollo(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma must cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);

	// Suma de vectores add_alig
	printf("\nSuma add allig\n");
	set_zeros(arr1, Size);
	start_time = _itoll(TSCH, TSCL);
	vecsum_add_alig(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma add alig cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);

	// Suma de vectores n_assert
	printf("\nSuma n_assert \n");
	set_zeros(arr1, Size);
	start_time = _itoll(TSCH, TSCL);
	vecsum_nassert(arr1, x, y, Size);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma n_assert cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr1, Size);

	// ALINEAR PARA VECTOR IMPAR
	short x2[] = {2, 3, 2, 3, 4};
	short y2[] = {1, 2, 3, 4, 5};

	short arr2[5];

	printf("\nVector add_allig IMPAR\n");
	start_time = _itoll(TSCH, TSCL);
	vecsum_add_alig_impar(arr2, x2, y2, 5);
	end_time = _itoll(TSCH, TSCL);
	cyclecount = end_time-start_time-overhead;
	printf("Suma add_alig IMPAR cyclecount: %lld CPU cycles\n", cyclecount);
	showArray_example(arr2, 5);
	*/
}



void showArray_example(short * const restrict arr, short N){
	int i =0;
	for (i = 0; i < N; i++) {
			printf("i: %d, arr[i]:%d\n",i, arr[i]);
	}
}

void set_zeros(short * restrict arr, short N){
	int i =0;
	for (i = 0; i < N; i++) {
			arr[i]=0;
	}
}

void vecsum(short * sum, short * in1, short * in2, short N){
	int i;
	for (i = 0; i < N; i++)
		sum[i] = in1[i] + in2[i];
}
void vecsum_restrict(short * restrict sum, short * restrict in1, short * restrict in2, short N){
	int i;
	for (i = 0; i < N; i++)
		sum[i] = in1[i] + in2[i];
}
void vecsum_const_restrict(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N){
	int i;
	for (i = 0; i < N; i++)
		sum[i] = in1[i] + in2[i];
}

void vecsum_add(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N) {
	int i;
	for (i = 0; i < N; i++)
		sum[i] = _add2(in1[i], in2[i]);
 }

void vecsum_must(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N) {
	int i;
	#pragma MUST_ITERATE (30, 30)
	for (i = 0; i < N; i++)
		sum[i] = in1[i] + in2[i];
}

void vecsum_must_desenrollo(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N) {
	int i;
	#pragma MUST_ITERATE (15, 15)
	for (i = 0; i < N; i+=2){
		sum[i] = in1[i] + in2[i];
		sum[i+1] = in1[i+1] + in2[i+1];
	}
}

void vecsum_add_alig(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N) {
	int i;
	#pragma MUST_ITERATE (15, 15)
	for (i = 0; i < N; i+=2){
		_amem4(&sum[i]) = _add2(_amem4_const(&in1[i]), _amem4_const(&in2[i]));
	}
}

void vecsum_add_alig_impar(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N) {
	int i;

	#pragma MUST_ITERATE (15, 15)
	for (i = 0; i < N; i+=2){
		_amem4(&sum[i]) = _add2(_amem4_const(&in1[i]), _amem4_const(&in2[i]));
	}
	if (N & 0x1){
		sum[N-1] = in1[N-1] + in2[N-1];
	}

}

void vecsum_nassert(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N){
	int i;
	_nassert(((int)sum & 0x3) == 0);
	_nassert(((int)in1 & 0x3) == 0);
	_nassert(((int)in2 & 0x3) == 0);
	#pragma MUST_ITERATE (30, 30);
	for (i = 0; i < N; i++){
		sum[i] = in1[i] + in2[i];
	}
}
