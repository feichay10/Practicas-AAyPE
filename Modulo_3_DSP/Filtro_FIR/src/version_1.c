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
#include <stdint.h>

#define COEF 21
#define N 2000

float* init_coeficients() {
  float* array_coef = (float*)malloc(COEF * sizeof(float));
  int i = 0;
  FILE* file_coef = fopen("../data/coeficientes.csv", "r");
  if (file_coef == NULL) {
    printf("Error al abrir el archivo\n");
    exit(1);
  }

  const size_t line_size = 3000;
  char* line = malloc(line_size);
  fgets(line, line_size, file_coef);
  char delim[] = ",";
  char* token;
  for (token = strtok(line, delim); token != NULL; token = strtok(NULL, delim)) {
    if (i >= COEF) {
      break;
    } else {
      array_coef[i] = atof(token);
      printf("coef = %s\n", token);
      i++;
    }
  }

  fclose(file_coef);

  return array_coef;
}

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

float* fir_filter(float* data, float* coef) {
  float* result = (float*)malloc(N * sizeof(float));
  int i, j;
  for (i = 0; i < N; i++) {
    result[i] = 0;
    for (j = 0; j < COEF; j++) {
      if (i - j >= 0) {
        result[i] += data[i - j] * coef[j];
      }
    }
  }

  return result;
}

int main() {
  float* coef = init_coeficients();
  float* data = init_data();
  float* result;

  // Medir los ciclos de CPU
  clock_t start, end;
  start = clock();
  result = fir_filter(data, coef);
  end = clock();

  double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
  printf("Tiempo de ejecución: %f\n", time_taken);

  free(coef);
  free(data);
  free(result);

  return 0;
}