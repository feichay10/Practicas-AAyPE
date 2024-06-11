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
 * @brief Primera versión del código se corresponde con la practica.c sin
 * modificar, y a la segunda implementación del filtro FIR
 * @version 0.1
 * @date 2024-01-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_COEF 5
#define N_DATAS 5

float* init_coef();
float* init_vector();
float* fir_filter(float* vector_coef, float* vector_datas);

int main(void) {}

float* init_coef() {
  float* vector_coef = (float*)malloc(N_COEF * sizeof(float));

  int i = 0;
  FILE* file_coef = fopen("../data/Coeficientes.csv", "r");
  if (file_coef == NULL) {
    printf("Error al abrir el fichero de coeficientes\n");
    exit(1);
  }
  while (fscanf(file_coef, "%f", &vector_coef[i]) != EOF && i < N_COEF) {
    i++;
  }
  fclose(file_coef);

  return vector_coef;
}

float* init_vector() {
  float* vector_datas = (float*)malloc(N_DATAS * sizeof(float));

  int i = 0;
  FILE* file_datas = fopen("../data/musica4.csv", "r");
  if (file_datas == NULL) {
    printf("Error al abrir el fichero de datos\n");
    exit(1);
  }
  while (fscanf(file_datas, "%f", &vector_datas[i]) != EOF && i < N_DATAS) {
    i++;
  }
  fclose(file_datas);

  return vector_datas;
}