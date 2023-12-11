/*
 * practica.h
 *
 */

#define Size1 50
#define Size2 200

#ifndef PRACTICA
#define PRACTICA

void practica();
int funcion_1(int val);
int funcion_2(int val);
void vDotV(int * vDot, int *in1, int *in2);
void minV(int *maxV, int *in1, int *in2);
int sumVelements(int *in1, int siz);

void showArray_practica(int * const restrict arr, int N);

#endif /* PRACTICA */