/*
 * practica.c
 *
 */

#include <stdlib.h>

#include <stdio.h>

#include "practica.h"

int funcion_1(int val) { return val * 4; }

int funcion_2(int val) {
  int res;
  if (val > 10) {
    res = val + 5;
  } else {
    res = val + 20;
  }
  return res;
}

int sumVelements(int *in1, int siz) {
  int i;
  int sum;
  for (i = 0; i < siz; i++) {
    sum += in1[i];
  }
  return sum;
}

void minV(int *minV, int *in1, int *in2) {
  int i;
  for (i = 0; i < Size2; i++) {
    if (in1[i] < in2[i]) {
      minV[i] = in1[i];
    } else {
      minV[i] = in2[i];
    }
  }
}

void vDotV(int *vDot, int *in1, int *in2) {
  int i;
  for (i = 0; i < Size2; i++) {
    vDot[i] = in1[i] * in2[i];
  }
  printf("ERHSklfhas�lfhsdf");
}

void showArray_practica(int *const restrict arr, int N) {
  int i = 0;
  for (i = 0; i < N; i++) {
    printf("i: %d, arr[i]:%d\n", i, arr[i]);
  }
}

void practica() {
  // Assign memory
  int *arr1 = malloc(Size1 * sizeof(int));
  int *arr2 = malloc(Size2 * sizeof(int));
  int *arr3 = malloc(Size2 * sizeof(int));
  int *c = malloc(Size2 * sizeof(int));
  int *d = malloc(Size2 * sizeof(int));

  int sum1, sum2, sum3;

  int i;
  i = 0;

  for (i; i < 350; i++) {
    if (i < 50) {
      int a, b;
      a = funcion_2(i);
      b = funcion_2(3);
      arr1[i] = (b * a) + i;
    } else {
      int a;
      a = funcion_1(i - 50);
      d[i - 50] = (i - 25 * 2) + a;
      d[i - 50] = funcion_2(d[i - 50]);
      c[i - 50] = i - 5;
    }
  }

  minV(arr2, d, c);
  vDotV(arr3, arr2, c);
  sum1 = sumVelements(arr1, 25);
  sum2 = sumVelements(arr2, 25);
  sum3 = sumVelements(arr3, 25);

  printf("sum1: %d, sum2: %d, sum3: %d\n", sum1, sum2, sum3);

  // Free memory
  free(arr1);
  free(arr2);
  free(arr3);
  free(c);
  free(d);
}