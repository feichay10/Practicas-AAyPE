#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>  // SSE intrinsics

#define COEF 25  // Número de coeficientes del filtro
#define N 7000   // Número de datos de entrada

#define BLOCK 10

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
 * @brief Función para calcular los ciclos de reloj
 * 
 * @return uint64_t 
 */
uint64_t rdtsc(){
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

/**
 * @brief Aplicación del filtro FIR version base
 * 
 * @param vector_coef 
 * @param vector_data 
 * @param result 
 */
float* firfilter_base(float* vector_coef, float* vector_data) {
  float* result = (float*)malloc((N + COEF - 1) * sizeof(float));
  int i, j;
  for (i = 0; i < N + COEF - 1; i++) {
    result[i] = 0;
    for (j = 0; j < COEF; j++) {
      if (i - j >= 0 && i - j < N) {
        result[i] += vector_coef[j] * vector_data[i - j];
      }
    }
  }

  return result;
}

/**
 * @brief Aplicación del filtro FIR version 1
 * 
 * @param vector_coef 
 * @param vector_data 
 * @param result 
 */
void firfilter_v1(float* vector_coef, float* vector_data, float* result) {
  int i, j;
  for (i = 0; i < N + COEF - 1; i++) {
    result[i] = 0;
    for (j = 0; j < COEF; j++) {
      result[i] += vector_coef[j] * vector_data[i - j];
    }
  }
}

/**
 * @brief Aplicación del filtro FIR version 2 con keywords
 * 
 * @param vector_coef 
 * @param vector_data 
 * @param result 
 */
void firfilter_v2(const float* restrict const vector_coef, const float* restrict const vector_data, float* restrict const result) {
  int i, j;
  for (i = 0; i < N + COEF - 1; i++) {
    result[i] = 0;
    for (j = 0; j < COEF; j++) {
      result[i] += vector_coef[j] * vector_data[i - j];
    }
  }
}

void firfilter_v3(const float* restrict const vector_coef, const float* restrict const vector_data, float* restrict const result) {
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

void firfilter_v4(const float* restrict const vector_coef, const float* restrict const vector_data, float* restrict const result) {
  int i, j;

  #pragma MUST_ITERATE(1000)
  // # pragma GCC ivdep
  for (i = 0; i < N + COEF - 1; i++) {
    result[i] = 0;
    #pragma unroll(10)
    for (j = 0; j < COEF; j++) {
      result[i] += vector_coef[j] * vector_data[i - j];
    }
  }
}

void firfilter_v5(float* restrict vector_coef, float* restrict vector_data, float* restrict result) {
  int i, j;
  __m128 coef, data, res;

  for (i = 0; i < N + COEF - 1; i++) {
    res = _mm_setzero_ps();
    for (j = 0; j < COEF; j += 4) {
      if (i - j >= 0) {
        coef = _mm_loadu_ps(&vector_coef[j]);
        data = _mm_loadu_ps(&vector_data[i - j]);
        res = _mm_add_ps(res, _mm_mul_ps(coef, data));
      }
    }
    float temp[4];
    _mm_storeu_ps(temp, res);
    result[i] = temp[0] + temp[1] + temp[2] + temp[3];
  }
}


int main() {
  float* vector_in = inicializacion_vector_in();
  float* vector_coef = inicializacion_coeficientes();
  int i;

  // Variables para calcular el tiempo de ejecución
  uint64_t start, end;
  double sum_time = 0;
  uint64_t sum_cycles = 0;

  // Variables para almacenar los resultados de cada versión
  float* result_base;
  float* result_v1 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v2 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v3 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v4 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v5 = (float*)calloc(N + COEF - 1, sizeof(float));

  // Variables para almacenar los tiempos de ejecución y ciclos de cada versión
  double time_base, time_v1, time_v2, time_v3, time_v4, time_v5;
  uint64_t cycles_base, cycles_v1, cycles_v2, cycles_v3, cycles_v4, cycles_v5;

  // Variables para almacenar los tiempos de ejecución y ciclos de cada versión
  double media_time_base, media_time_v1, media_time_v2, media_time_v3, media_time_v4, media_time_v5;
  uint64_t media_cycles_base, media_cycles_v1, media_cycles_v2, media_cycles_v3, media_cycles_v4, media_cycles_v5;

  // Aplicación del filtro FIR

  // Versión base
  for (i = 0; i < REPETICIONES; i++) {
    start = rdtsc();
    result_base = firfilter_base(vector_coef, vector_in);
    end = rdtsc();
    sum_time += (double)(end - start);
    sum_cycles += end - start;
  }

  media_time_base = sum_time / REPETICIONES;
  media_cycles_base = sum_cycles / REPETICIONES;

  for (i = 0; i < REPETICIONES; i++) {
    start = rdtsc();
    firfilter_v1(vector_coef, vector_in, result_v1);
    end = rdtsc();
    sum_time += (double)(end - start);
    sum_cycles += end - start;
  }

  media_time_v1 = sum_time / REPETICIONES;
  media_cycles_v1 = sum_cycles / REPETICIONES;

  for (i = 0; i < REPETICIONES; i++) {
    start = rdtsc();
    firfilter_v2(vector_coef, vector_in, result_v2);
    end = rdtsc();
    sum_time += (double)(end - start);
    sum_cycles += end - start;
  }

  media_time_v2 = sum_time / REPETICIONES;
  media_cycles_v2 = sum_cycles / REPETICIONES;

  for (i = 0; i < REPETICIONES; i++) {
    start = rdtsc();
    firfilter_v3(vector_coef, vector_in, result_v3);
    end = rdtsc();
    sum_time += (double)(end - start);
    sum_cycles += end - start;
  }

  media_time_v3 = sum_time / REPETICIONES;
  media_cycles_v3 = sum_cycles / REPETICIONES;

  for (i = 0; i < REPETICIONES; i++) {
    start = rdtsc();
    firfilter_v4(vector_coef, vector_in, result_v4);
    end = rdtsc();
    sum_time += (double)(end - start);
    sum_cycles += end - start;
  }

  media_time_v4 = sum_time / REPETICIONES;
  media_cycles_v4 = sum_cycles / REPETICIONES;

  for (i = 0; i < REPETICIONES; i++) {
    start = rdtsc();
    firfilter_v5(vector_coef, vector_in, result_v5);
    end = rdtsc();
    sum_time += (double)(end - start);
    sum_cycles += end - start;
  }

  media_time_v5 = sum_time / REPETICIONES;
  media_cycles_v5 = sum_cycles / REPETICIONES;

  printf("============================================\n");
  printf("\t\tResultados base:\n");
  printf("Tiempo de ejecución versión base: %f ms\n", media_time_base);
  printf("Ciclos de reloj versión base: %" PRIu64 "\n", media_cycles_base);
  printf("============================================\n");

  printf("\n============================================\n");
  printf("\t\tResultados versión 1:\n");
  printf("Tiempo de ejecución versión 1: %f ms\n", media_time_v1);
  printf("Ciclos de reloj versión 1: %" PRIu64 "\n", media_cycles_v1);
  printf("============================================\n");

  printf("\n============================================\n");
  printf("\t\tResultados versión 2:\n");
  printf("Tiempo de ejecución versión 2: %f ms\n", media_time_v2);
  printf("Ciclos de reloj versión 2: %" PRIu64 "\n", media_cycles_v2);
  printf("============================================\n");

  printf("\n============================================\n");
  printf("\t\tResultados versión 3:\n");
  printf("Tiempo de ejecución versión 3: %f ms\n", media_time_v3);
  printf("Ciclos de reloj versión 3: %" PRIu64 "\n", media_cycles_v3);
  printf("============================================\n");

  printf("\n============================================\n");
  printf("\t\tResultados versión 4:\n");
  printf("Tiempo de ejecución versión 4: %f ms\n", media_time_v4);
  printf("Ciclos de reloj versión 4: %" PRIu64 "\n", media_cycles_v4);
  printf("============================================\n");

  printf("\n============================================\n");
  printf("\t\tResultados versión 5:\n");
  printf("Tiempo de ejecución versión 5: %f ms\n", media_time_v5);
  printf("Ciclos de reloj versión 5: %" PRIu64 "\n", media_cycles_v5);
  printf("============================================\n");

  free(vector_coef);
  free(vector_in);
  free(result_base);
  free(result_v1);
  free(result_v2);
  free(result_v3);
  free(result_v4);
  free(result_v5);

  return 0;
}