/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * Filtro Fir: version 1
 * @file version1.c
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Version 2: Desenrrollar el bucle interior del filtro FIR
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
#include <sys/time.h>
#include <time.h>

#define COEF 25  // Número de coeficientes del filtro
#define N 7000     // Número de datos de entrada

#define BLOCK 10 // Número de BLOCKs en los que se divide el bucle

// Número de repeticiones para el cálculo de la media de tiempo y ciclos
#define REPETICIONES 1000

float* inicializacion_coeficientes() {
  float* vector_coeficientes = (float*)malloc(COEF * sizeof(float));
  int i = 0;
  FILE* fich_coef = fopen("../data/Coeficientes.csv", "r");
  if (fich_coef == NULL) {
    printf("Error al abrir el archivo\n");
    exit(1);
  }

  while (fscanf(fich_coef, "%f", &vector_coeficientes[i]) != EOF && i < COEF) {
    i++;
  }
  fclose(fich_coef);

  return vector_coeficientes;
}

float* inicializacion_vector_in() {
  float* array_data = (float*)malloc(N * sizeof(float));

  int i = 0;
  FILE* file_data = fopen("../data/musica4.csv", "r");
  if (file_data == NULL) {
    printf("Error al abrir el archivo\n");
    exit(1);
  }

  while (fscanf(file_data, "%f", &array_data[i]) != EOF && i < N) {
    i++;
  }

  fclose(file_data);

  return array_data;
}

void firfilter(float* vector_coef, float* vector_data, float* result) {
  int i = 0;
  for (i = 0; i < COEF; i++) {
    int indice = 0;
    int iteraciones = (COEF / BLOCK);
    int resto = (COEF % BLOCK);
    while (iteraciones-- > 0) {
      result[i] += vector_data[indice + 10 + i] * vector_coef[indice + 10 - 1];
      result[i] += vector_data[indice + 9 + i] * vector_coef[indice + 9 - 1];
      result[i] += vector_data[indice + 8 + i] * vector_coef[indice + 8 - 1];
      result[i] += vector_data[indice + 7 + i] * vector_coef[indice + 7 - 1];
      result[i] += vector_data[indice + 6 + i] * vector_coef[indice + 6 - 1];
      result[i] += vector_data[indice + 5 + i] * vector_coef[indice + 5 - 1];
      result[i] += vector_data[indice + 4 + i] * vector_coef[indice + 4 - 1];
      result[i] += vector_data[indice + 3 + i] * vector_coef[indice + 3 - 1];
      result[i] += vector_data[indice + 2 + i] * vector_coef[indice + 2 - 1];
      result[i] += vector_data[indice + 1 + i] * vector_coef[indice + 1 - 1];
      indice += BLOCK;
    }
    switch (resto) {
      case 9:
        result[i] += vector_data[indice + 9 + i] * vector_coef[indice + 9 - 1];
      case 8:
        result[i] += vector_data[indice + 8 + i] * vector_coef[indice + 8 - 1];
      case 7:
        result[i] += vector_data[indice + 7 + i] * vector_coef[indice + 7 - 1];
      case 6:
        result[i] += vector_data[indice + 6 + i] * vector_coef[indice + 6 - 1];
      case 5:
        result[i] += vector_data[indice + 5 + i] * vector_coef[indice + 5 - 1];
      case 4:
        result[i] += vector_data[indice + 4 + i] * vector_coef[indice + 4 - 1];
      case 3:
        result[i] += vector_data[indice + 3 + i] * vector_coef[indice + 3 - 1];
      case 2:
        result[i] += vector_data[indice + 2 + i] * vector_coef[indice + 2 - 1];
      case 1:
        result[i] += vector_data[indice + 1 + i] * vector_coef[indice + 1 - 1];
      case 0:;
    }
  }
}

uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

int main() {
  float* vector_in = inicializacion_vector_in();
  float* vector_coef = inicializacion_coeficientes();
  float* result = (float*)calloc(N + COEF - 1, sizeof(float));
  int i;

  // Variables para el cálculo del tiempo de ejecución y ciclos
  struct timespec start, end;
  double elapsed = 0;
  uint64_t start_cycle, end_cycle;

  // Variables para el cálculo de la media de tiempo y ciclos
  float mean_time[REPETICIONES];
  uint64_t mean_cycles[REPETICIONES];

  printf("============================================\n");
  printf("\t\tCOEFICIENTES:\n");
  for (i = 0; i < COEF; i++) {
    printf("%f\n", vector_coef[i]);
  }
  printf("============================================\n\n\n");
  printf("============================================\n");
  printf("\t\tMusica4:\n");
  for (i = 0; i < N + COEF - 1; i++) {
    printf("%f\n", vector_in[i]);
  }
  printf("============================================\n\n\n");

  // APLICACION DEL FIR FILTER
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &start);
    firfilter(vector_coef, vector_in, result);
    clock_gettime(CLOCK_MONOTONIC, &end);
    end_cycle = rdtsc();

    elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
    mean_time[i] = elapsed;
    mean_cycles[i] = end_cycle - start_cycle;
  }

  // Calculo de la media
  double mean_time_result = 0;
  uint64_t mean_cycles_result = 0;

  for (i = 0; i < REPETICIONES; i++) {
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