/**
 *
 * Universidad de La Laguna
 * Escuela Superior de Ingeniería y Tecnología
 * Grado en Ingeniería Informática
 * Asignatura: Arquitecturas Avanzadas y de Propósito Específico
 * Curso: 4º
 * Filtro Fir: version 2
 * @file version_2.c
 * @author Cheuk Kelly Ng Pante (alu0101364544@ull.edu.es)
 * @brief Version 2: Debe incluir el uso de keywords, como const y restrict en
 * las variables que consideres.
 * 
 * @version 0.1
 * @date 2024-01-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define COEF 5
#define N 5

float* inicializacion_coeficientes() {
  float* restrict vector_coeficientes = (float*)malloc(COEF * sizeof(float));
  int i = 0;
  FILE* restrict fich_coef = fopen("../data/Coeficientes.csv", "r");
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
  float* restrict array_data = (float*)malloc(N * sizeof(float));

  int i = 0;
  FILE* restrict file_data = fopen("../data/musica4.csv", "r");
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

float* firfilter(float* restrict vector_coef, float* restrict vector_data) {
  float* restrict result = (float*)malloc(N * sizeof(float));
  int i, j;
  for (i = 0; i < N + COEF - 1; i++) {
    result[i] = 0;
    for (j = 0; j < COEF; j++)
      result[i] += vector_coef[i + j] * vector_data[i];
  }

  return result;
}

int main() {
  float* restrict vector_in = inicializacion_vector_in();
  float* restrict vector_coef = inicializacion_coeficientes();
  float* restrict result;
  int i;
  clock_t start, end;

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
  start = clock();
  result = firfilter(vector_coef, vector_in);
  end = clock();

  printf("============================================\n");
  printf("\t\tVECTOR SALIDA:\n");
  for (i = 0; i < N + COEF - 1; i++) {
    printf("%f\n", result[i]);
  }
  printf("============================================\n");

  double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
  printf("\nTiempo de ejecución: %f\n", time_taken);

  free((float*)vector_coef);
  free((float*)vector_in);
  free((float*)result);

  return 0;
}