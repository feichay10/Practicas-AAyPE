/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * Filtro Fir: version 4
 * @file version_4.c
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Version 4: Utilizar pragmas. Pudiendo llegar a sustituir el desenrrollado
 * manual hecho por los pragmas.
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

// Número de repeticiones para el cálculo de la media de tiempo y ciclos
#define REPETICIONES 1000

/**
 * @brief Inicialización de los coeficientes del filtro FIR
 * 
 * @return float* 
 */
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

/**
 * @brief Inicialización de los datos de entrada
 * 
 * @return float* 
 */
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


/**
 * @brief Aplicación del filtro FIR version 4 con desenrrollado hecho por pragmas
 * 
 * @param vector_coef 
 * @param vector_data 
 * @param result 
 */
void firfilter(float* vector_coef, float* vector_data, float* result) {
  int i, j;

  #pragma MUST_ITERATE(1000)
  for (i = 0; i < N + COEF - 1; i++) {
    result[i] = 0;
    #pragma unroll(3)
    for (j = 0; j < COEF; j++) {
      result[i] += vector_coef[i + j] * vector_data[i];
    }
  }
}

/**
 * @brief Función para calcular los ciclos de reloj
 * 
 * @return uint64_t 
 */
uint64_t rdtsc(){
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

int main() {
  #pragma DATA_ALIGN(8)
  float* restrict vector_in = inicializacion_vector_in();
  float* restrict vector_coef = inicializacion_coeficientes();
  float* restrict result = (float*)calloc(N + COEF - 1, sizeof(float));

  struct timespec start, end;
  double elapsed = 0;
  uint64_t start_cycle, end_cycle;

  float mean_time[REPETICIONES];
  uint64_t mean_cycles[REPETICIONES];

  for (int i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &start);
    firfilter(vector_coef, vector_in, result);
    clock_gettime(CLOCK_MONOTONIC, &end);
    end_cycle = rdtsc();

    elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
    mean_time[i] = elapsed;
    mean_cycles[i] = end_cycle - start_cycle;
  }

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