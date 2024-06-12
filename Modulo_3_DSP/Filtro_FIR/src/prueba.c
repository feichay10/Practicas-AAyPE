#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define COEF 21
#define N 5


float* inicializacion_coeficientes() {
  float* vector_coeficientes = (float*)malloc(COEF * sizeof(float));
  int i = 0;
  FILE* fich_coef = fopen("../data/Coeficientes.csv", "r");
  if (file_coef == NULL) {
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

float* firfilter(float* vector_coef, float* vector_in) {
  float* vector_salida = (float*)malloc((N + COEF - 1) * sizeof(float));
  int i;
  int j;
  for (i = 0; i < N + COEF - 1; i++) {
    vector_salida[i] = 0;
    for (j = 0; j < COEF; j++)
      vector_salida[i] += vector_coef[i + j] * vector_in[i];
  }

  return vector_salida;
}

int main(void) {
  float* vector_in = inicializacion_vector_in();
  float* vector_coef = inicializacion_coeficientes();
  float* vector_out;
  int i;

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
  vector_out = firfilter(vector_coef, vector_in);

  printf("============================================\n");
  printf("\t\tVECTOR SALIDA:\n");
  for (i = 0; i < N + COEF - 1; i++) {
    printf("%f\n", vector_out[i]);
  }
  printf("============================================\n");

  return 0;
}