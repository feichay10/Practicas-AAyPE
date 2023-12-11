/*
 * ejercicio_pract.c
 *
 *      Author: Sergio D�az Gonz�lez
 */

#include <cstdio>
#include "ejercicio 3.h"

void ejercicio_pract(){
	int x[] = {1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
				  1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
				  0, 1, 0, 1, 0, 0, 1, 0, 1, 1};
	int y[] = {8, 9, 12, 3, 4, 1, 6, 3, 4, 4,
				  9, 12, 3, 4, 1, 6, 3, 4, 4, 2,
				  6, 3, 4, 4, 2, 23, 4, 5, 1, 9};
	int z[] = {19, 29, 2, 6, 14, 10, 26, 5, 7, 2,
			   10, 26, 5, 4, 5, 1, 9, 43, 4, 2,
				2, 3, 1, 5, 4, 4, 3, 2, 1, 9};

	int arr1[Size], arr2[Size], arr3[Size], arrMax[Size];
	int sum, sca;
	sum = sumVector(y);
	maxVector(arrMax, z, x);
	vectorDotVector(arr1, z, y);
	sca = scalarProd(x, y);
}

void vectorDotVector(int * vDot, int *in1, int *in2){
	int i;
	for (i = 0; i < Size; i++) {
		vDot[i] = in1[i] * in2[i];
   }
}

void maxVector(int *maxV, int *in1, int *in2){
	int i;
	for (i = 0; i < Size; i++) {
		if(in1[i] > in2[i]){
			maxV[i] = in1[i];
		}else{
			maxV[i] = in2[i];
		}
	}
}

int sumVector(int *in1){
	int i;
	int out;
	for (i = 0; i < Size; i++) {
		out += in1[i];
	}
	return out;
}

int scalarProd(int *in1, int *in2){
	int i;
	int out;
	for (i = 0; i < Size; i++) {
		out += in1[i] * in2[i];
	}
	return out;
}

void showArray_ejercicio3(short * const restrict arr, short N){
	int i =0;
	for (i = 0; i < N; i++) {
			printf("i: %d, arr[i]:%d\n",i, arr[i]);
	}
}


