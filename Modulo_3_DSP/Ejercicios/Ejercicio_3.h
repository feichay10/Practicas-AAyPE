/*
 * ejercicio_pract.h
 *
 *      Author: Sergio D�az Gonz�lez
 */

#define Size 30


#ifndef EJERCICIO_3
#define EJERCICIO_3

void ejercicio_pract();

void vectorDotVector(int * vDot, int *in1, int *in2);

void vectorSumVector(int * vSum, int *in1, int *in2);

void maxVector(int *maxV, int *in1, int *in2);

int sumVector(int *in1);

int scalarProd(int *in1, int *in2);

void showArray_ejercicio3(short * const restrict arr, short N);


#endif /* EJERCICIO_3 */
