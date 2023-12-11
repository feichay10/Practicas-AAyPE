/*
 * ejercicio2.c
 *
 */


#include <cstdio>
#include "Ejercicio_2.h"

int funcion_bucle1(int val){
	return val*3;
}

int funcion_bucle2(int val){
	int res;
	if(val > 10){
		res = val*2;
	}else{
		res = val*5;
	}
	return res;
}

void ejecicio_b(){
	int arr1[500], arr2[500];
	int i;
	i=0;
	for (i; i < 500; i++) {
		if(i<100){
			int a, b;
			a = funcion_bucle2(i);
			b = funcion_bucle2(3);
			arr1[i]=(b*a) + i;
		}else{
			int a;
			int d[400];
			a = funcion_bucle1(i);
			d[i] = i * 4;
			d[i] = funcion_bucle2(d[i]);
		}
	}
}

void ejecicio_a(){
	int i;
	i=0;
	for (i = 0; i < 100; i++) {
		int a, b, x;
		a=4;
		if(a<=5){
			b=funcion_bucle1(i);
			x += 2*b;
		}else{
			b=funcion_bucle1(i);
			x += 3*b;
		}
	}
}