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

// Número de REPETITIONS para el cálculo de la media de tiempo y ciclos
#define REPETITIONS 100

/**
 * @brief Inicialización de los coeficientes del filtro FIR
 * 
 * @return float* 
 */
float* init_coefficients() {
  float* array_coeff = (float*)malloc(COEF * sizeof(float));
  int i = 0;
  FILE* file_coeff = fopen("../data/Coeficientes.csv", "r");
  if (file_coeff == NULL) {
    printf("Error al abrir el archivo\n");
    exit(1);
  }

  while (fscanf(file_coeff, "%f", &array_coeff[i]) != EOF && i < COEF) {
    i++;
  }
  fclose(file_coeff);

  return array_coeff;
}

/**
 * @brief Inicialización de los datos de entrada
 * 
 * @return float* 
 */
float* init_data() {
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
 * @brief Aplicación del filtro FIR version 4 utilizando pragmas
 * 
 * @param vector_coef 
 * @param vector_data 
 * @param result 
 */
void firfilter(const float* restrict const vector_coef, const float* restrict const vector_data, float* restrict const result) {
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    result[i] = 0;
    #pragma unroll 25
    for (int j = 0; j < COEF; j++) {
      result[i] += vector_coef[j] * vector_data[i - j];
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
  float* vector_in = init_data();
  float* vector_coef = init_coefficients();
  float* result = (float*)malloc((N + COEF - 1) * sizeof(float));
  int i;

  // Variables para el cálculo del tiempo de ejecución y ciclos
  clock_t start, end;
  double elapsed = 0;
  uint64_t start_cycle, end_cycle;

  // Variables para el cálculo de la media de tiempo y ciclos
  float mean_time[REPETITIONS];
  uint64_t mean_cycles[REPETITIONS];

  // Aplicación del filtro FIR
  for (i = 0; i < REPETITIONS; i++) {
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

  for (int i = 0; i < REPETITIONS; i++) {
    mean_time_result += mean_time[i];
    mean_cycles_result += mean_cycles[i];
  }

  mean_time_result /= REPETITIONS;
  mean_cycles_result /= REPETITIONS;

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