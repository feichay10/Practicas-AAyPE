/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * Filtro Fir: version 3
 * @file version_3.c
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Version 3: Desenrrollado manual y la optimización bucles,
 * condicionales, tipos y funciones. Implica desenrrollado del prologo y epilodo
 * del filtro.
 *
 * @version 0.1
 * @date 2024-01-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define COEF 25  // Número de coeficientes del filtro
#define N 7000   // Número de datos de entrada

#define BLOCK 10  // Número de BLOCKs en los que se divide el bucle

// Número de repeticiones para el cálculo de la media de tiempo y ciclos
#define REPETICIONES 1000

float* inicializacion_coeficientes() {
  float* vector_coeficientes = (float*)malloc(COEF * sizeof(float));
  FILE* fich_coef = fopen("../data/Coeficientes.csv", "r");
  if (fich_coef == NULL) {
    printf("Error al abrir el archivo\n");
    exit(1);
  }

  for (int i = 0; i < COEF; i++) {
    if (fscanf(fich_coef, "%f", &vector_coeficientes[i]) == EOF) {
      break;
    }
  }
  fclose(fich_coef);

  return vector_coeficientes;
}

float* inicializacion_vector_in() {
  float* array_data = (float*)malloc(N * sizeof(float));
  FILE* file_data = fopen("../data/musica4.csv", "r");
  if (file_data == NULL) {
    printf("Error al abrir el archivo\n");
    exit(1);
  }

  for (int i = 0; i < N; i++) {
    if (fscanf(file_data, "%f", &array_data[i]) == EOF) {
      break;
    }
  }
  fclose(file_data);

  return array_data;
}

void firfilter(const float* restrict vector_coef, const float* restrict vector_data, float* restrict result) {
  for (int i = 0; i < COEF; i++) {
    int indice = 0;
    int iteraciones = COEF / BLOCK;
    int resto = COEF % BLOCK;

    while (iteraciones-- > 0) {
      result[i] += vector_coef[0 + indice] * vector_data[i - 0 + indice];
      result[i] += vector_coef[1 + indice] * vector_data[i - 1 + indice];
      result[i] += vector_coef[2 + indice] * vector_data[i - 2 + indice];
      result[i] += vector_coef[3 + indice] * vector_data[i - 3 + indice];
      result[i] += vector_coef[4 + indice] * vector_data[i - 4 + indice];
      result[i] += vector_coef[5 + indice] * vector_data[i - 5 + indice];
      result[i] += vector_coef[6 + indice] * vector_data[i - 6 + indice];
      result[i] += vector_coef[7 + indice] * vector_data[i - 7 + indice];
      result[i] += vector_coef[8 + indice] * vector_data[i - 8 + indice];
      result[i] += vector_coef[9 + indice] * vector_data[i - 9 + indice];
      indice += BLOCK;
    }

    switch (resto) {
      case 1: result[i] += vector_coef[1 + indice] * vector_data[i - 1 + indice];
      case 2: result[i] += vector_coef[2 + indice] * vector_data[i - 2 + indice];
      case 3: result[i] += vector_coef[3 + indice] * vector_data[i - 3 + indice];
      case 4: result[i] += vector_coef[4 + indice] * vector_data[i - 4 + indice];
      case 5: result[i] += vector_coef[5 + indice] * vector_data[i - 5 + indice];
      case 6: result[i] += vector_coef[6 + indice] * vector_data[i - 6 + indice];
      case 7: result[i] += vector_coef[7 + indice] * vector_data[i - 7 + indice];
      case 8: result[i] += vector_coef[8 + indice] * vector_data[i - 8 + indice];
      case 9: result[i] += vector_coef[9 + indice] * vector_data[i - 9 + indice];
      default: break;
    }
  }
}

uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

int main() {
  float* restrict vector_in = inicializacion_vector_in();
  float* restrict vector_coef = inicializacion_coeficientes();
  float* restrict result = (float*)calloc(N + COEF - 1, sizeof(float));
  int i;

  // Variables para el cálculo del tiempo de ejecución y ciclos
  clock_t start, end;
  double elapsed = 0;
  uint64_t start_cycle, end_cycle;

  // Variables para el cálculo de la media de tiempo y ciclos
  float mean_time[REPETICIONES];
  uint64_t mean_cycles[REPETICIONES];

  // Aplicación del filtro FIR
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    start = clock();
    firfilter(vector_coef, vector_in, result);
    end = clock();
    end_cycle = rdtsc();

    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    mean_time[i] = elapsed;
    mean_cycles[i] = end_cycle - start_cycle;
  }

  // Calculo de la media
  double mean_time_result = 0;
  uint64_t mean_cycles_result = 0;

  for (int i = 0; i < REPETICIONES; i++) {
    mean_time_result += mean_time[i];
    mean_cycles_result += mean_cycles[i];
  }

  mean_time_result /= REPETICIONES;
  mean_cycles_result /= REPETICIONES;

  printf("============================================\n");
  printf("\t\tResultados:");
  printf("\nTiempo de ejecución: %f ms\n", mean_time_result);
  printf("Ciclos de reloj: %" PRIu64 "\n", mean_cycles_result);
  printf("============================================\n");

  free(vector_coef);
  free(vector_in);
  free(result);

  return 0;
}
