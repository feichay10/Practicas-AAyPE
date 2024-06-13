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
 * @brief Implementar la función del Filtro FIR para que se pueda trabajar con un
 * vector de datos y un vector de coeficientes cuyo tamaño se pueda modificar
 * fácilmente. Ambos vectores son pasados a la función como argumentos.
 * Dicha función tiene ser invocada desde el main y retornar un vector.
 * 
 * El programa efectúa la lectura de los valores de los
 * coeficientes (archivo coeficientes.csv) y de los valores del archivo
 * musica4.csv. Son 21 coeficientes. Del archivo musica4.csv no hace falta
 * obtener todos los valores, porque hablamos de un fichero muy grande. Se
 * puede jugar con el tamaño de los datos en las pruebas. El mínimo será 2000.
 * Además, los vectores hay que pasárselos como punteros. En este caso no
 * retorna el vector, sino el vector resultante también es pasado como un
 * puntero a la función.
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
#include <sys/time.h>

#define COEF 5
#define N 5

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

float* firfilter(float* vector_coef, float* vector_data) {
  float* result = (float*)malloc(N * sizeof(float));
  int i, j;
  for (i = 0; i < N + COEF - 1; i++) {
    result[i] = 0;
    for (j = 0; j < COEF; j++)
      result[i] += vector_coef[i + j] * vector_data[i];
  }

  return result;
}

int main() {
  float* vector_in = inicializacion_vector_in();
  float* vector_coef = inicializacion_coeficientes();
  float* result;
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
  printf("\t\tSalida:\n");
  for (i = 0; i < N + COEF - 1; i++) {
    printf("%f\n", result[i]);
  }
  printf("============================================\n");

  // Tiempo de ejecución en milisegundos
  double time_taken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
  printf("\nTiempo de ejecución: %f ms\n", time_taken);

  free((float*)vector_coef);
  free((float*)vector_in);
  free((float*)result);

  return 0;
}