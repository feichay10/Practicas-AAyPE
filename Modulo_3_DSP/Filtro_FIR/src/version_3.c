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

// Implementación del filtro FIR desenrrollado manualmente y optimizado
void firfilter(const float* restrict const vector_coef, const float* restrict const vector_data, float* restrict const result) {
  int i;
  for (i = 0; i < N; i++) {
    result[i] = 0;

    // Desenrollado manual del bucle interno
    if (i >= 24) result[i] += vector_coef[24] * vector_data[i - 24];
    if (i >= 23) result[i] += vector_coef[23] * vector_data[i - 23];
    if (i >= 22) result[i] += vector_coef[22] * vector_data[i - 22];
    if (i >= 21) result[i] += vector_coef[21] * vector_data[i - 21];
    if (i >= 20) result[i] += vector_coef[20] * vector_data[i - 20];
    if (i >= 19) result[i] += vector_coef[19] * vector_data[i - 19];
    if (i >= 18) result[i] += vector_coef[18] * vector_data[i - 18];
    if (i >= 17) result[i] += vector_coef[17] * vector_data[i - 17];
    if (i >= 16) result[i] += vector_coef[16] * vector_data[i - 16];
    if (i >= 15) result[i] += vector_coef[15] * vector_data[i - 15];
    if (i >= 14) result[i] += vector_coef[14] * vector_data[i - 14];
    if (i >= 13) result[i] += vector_coef[13] * vector_data[i - 13];
    if (i >= 12) result[i] += vector_coef[12] * vector_data[i - 12];
    if (i >= 11) result[i] += vector_coef[11] * vector_data[i - 11];
    if (i >= 10) result[i] += vector_coef[10] * vector_data[i - 10];
    if (i >= 9) result[i] += vector_coef[9] * vector_data[i - 9];
    if (i >= 8) result[i] += vector_coef[8] * vector_data[i - 8];
    if (i >= 7) result[i] += vector_coef[7] * vector_data[i - 7];
    if (i >= 6) result[i] += vector_coef[6] * vector_data[i - 6];
    if (i >= 5) result[i] += vector_coef[5] * vector_data[i - 5];
    if (i >= 4) result[i] += vector_coef[4] * vector_data[i - 4];
    if (i >= 3) result[i] += vector_coef[3] * vector_data[i - 3];
    if (i >= 2) result[i] += vector_coef[2] * vector_data[i - 2];
    if (i >= 1) result[i] += vector_coef[1] * vector_data[i - 1];
    result[i] += vector_coef[0] * vector_data[i - 0];
  }
}

uint64_t rdtsc() {
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
