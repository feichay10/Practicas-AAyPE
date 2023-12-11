/*
 * ejercicio_pract.h
 *
 *      Author: Sergio D�az Gonz�lez
 */

#define Size 30


#ifndef EJEMPLO_ELEMENTOS
#define EJEMPLO_ELEMENTOS

void example();
//void vectorDotVector(short * restrict vDot, short * restrict in1, short * restrict in2);

//void vectorSumVector(short * restrict vSum, short * restrict in1, short * restrict in2);

void vecsum(short * sum, short * in1, short * in2, short N);
void vecsum_restrict(short * restrict sum, short * restrict in1, short * restrict in2, short N);
void vecsum_const_restrict(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N);
void vecsum_add(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N);
void vecsum_must(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N);
void vecsum_must_desenrollo(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N);
void vecsum_add_alig(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N);
void vecsum_add_alig_impar(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N);
void vecsum_nassert(short * restrict sum,  short * const restrict in1,  short * const restrict in2, short N);

void showArray_example(short * const restrict arr, short N);
void set_zeros(short * restrict arr, short N);
#endif /* EJEMPLO_ELEMENTOS */