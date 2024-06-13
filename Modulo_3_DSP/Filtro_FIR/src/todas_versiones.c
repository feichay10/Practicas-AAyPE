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

// Implementación del filtro FIR desenrrollado manualmente y optimizado
void firfilter_v3(const float* restrict const vector_coef, const float* restrict const vector_data, float* restrict const result) {
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
  float* result_base;
  float* result_v1 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v2 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v3 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v4 = (float*)calloc(N + COEF - 1, sizeof(float));
  float* result_v5 = (float*)calloc(N + COEF - 1, sizeof(float));
  int i;

  // Variables para calcular el tiempo de ejecución para cada versión
  clock_t start, end;
  double elapsed = 0;
  uint64_t start_cycle, end_cycle;

  // Variables para calcular la media de tiempo y ciclos
  float mean_time_base[REPETICIONES], mean_time_v1[REPETICIONES], mean_time_v2[REPETICIONES], mean_time_v3[REPETICIONES], mean_time_v4[REPETICIONES], mean_time_v5[REPETICIONES];
  uint64_t mean_cycles_base[REPETICIONES], mean_cycles_v1[REPETICIONES], mean_cycles_v2[REPETICIONES], mean_cycles_v3[REPETICIONES], mean_cycles_v4[REPETICIONES], mean_cycles_v5[REPETICIONES];

  // Aplicación del filtro FIR
  // Versión base
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    start = clock();
    result_base = firfilter_base(vector_coef, vector_in);
    end = clock();
    end_cycle = rdtsc();

    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    mean_time_base[i] = elapsed;
    mean_cycles_base[i] = end_cycle - start_cycle;
  }

  // Calculo de la media base
  double mean_time_base_result = 0;
  uint64_t mean_cycles_base_result = 0;

  for (i = 0; i < REPETICIONES; i++) {
    mean_time_base_result += mean_time_base[i];
    mean_cycles_base_result += mean_cycles_base[i];
  }

  mean_time_base_result /= REPETICIONES;
  mean_cycles_base_result /= REPETICIONES;

  printf("============================================\n");
  printf("\t\tResultados version base:");
  printf("\nTiempo de ejecución: %f ms\n", mean_time_base_result);
  printf("Ciclos de reloj: %" PRIu64 "\n", mean_cycles_base_result);
  printf("============================================\n");

  elapsed = 0;

  // Versión 1
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    start = clock();
    firfilter_v1(vector_coef, vector_in, result_v1);
    end = clock();
    end_cycle = rdtsc();

    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    mean_time_v1[i] = elapsed;
    mean_cycles_v1[i] = end_cycle - start_cycle;
  }

  // Calculo de la media v1
  double mean_time_v1_result = 0;
  uint64_t mean_cycles_v1_result = 0;

  for (i = 0; i < REPETICIONES; i++) {
    mean_time_v1_result += mean_time_v1[i];
    mean_cycles_v1_result += mean_cycles_v1[i];
  }

  mean_time_v1_result /= REPETICIONES;
  mean_cycles_v1_result /= REPETICIONES;

  printf("\n\n============================================\n");
  printf("\t\tResultados version 1:");
  printf("\nTiempo de ejecución: %f ms\n", mean_time_v1_result);
  printf("Ciclos de reloj: %" PRIu64 "\n", mean_cycles_v1_result);
  printf("============================================\n");

  elapsed = 0;

  // Versión 2
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    start = clock();
    firfilter_v2(vector_coef, vector_in, result_v2);
    end = clock();
    end_cycle = rdtsc();

    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    mean_time_v2[i] = elapsed;
    mean_cycles_v2[i] = end_cycle - start_cycle;
  }

  // Calculo de la media v2
  double mean_time_v2_result = 0;
  uint64_t mean_cycles_v2_result = 0;

  for (i = 0; i < REPETICIONES; i++) {
    mean_time_v2_result += mean_time_v2[i];
    mean_cycles_v2_result += mean_cycles_v2[i];
  }

  mean_time_v2_result /= REPETICIONES;
  mean_cycles_v2_result /= REPETICIONES;

  printf("\n\n============================================\n");
  printf("\t\tResultados version 2:");
  printf("\nTiempo de ejecución: %f ms\n", mean_time_v2_result);
  printf("Ciclos de reloj: %" PRIu64 "\n", mean_cycles_v2_result);
  printf("============================================\n");

  elapsed = 0;

  // Versión 3
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    start = clock();
    firfilter_v3(vector_coef, vector_in, result_v3);
    end = clock();
    end_cycle = rdtsc();

    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    mean_time_v3[i] = elapsed;
    mean_cycles_v3[i] = end_cycle - start_cycle;
  }

  // Calculo de la media v3
  double mean_time_v3_result = 0;
  uint64_t mean_cycles_v3_result = 0;

  for (i = 0; i < REPETICIONES; i++) {
    mean_time_v3_result += mean_time_v3[i];
    mean_cycles_v3_result += mean_cycles_v3[i];
  }

  mean_time_v3_result /= REPETICIONES;
  mean_cycles_v3_result /= REPETICIONES;

  printf("\n\n============================================\n");
  printf("\t\tResultados version 3:");
  printf("\nTiempo de ejecución: %f ms\n", mean_time_v3_result);
  printf("Ciclos de reloj: %" PRIu64 "\n", mean_cycles_v3_result);
  printf("============================================\n");

  elapsed = 0;

  // Versión 4
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    start = clock();
    firfilter_v4(vector_coef, vector_in, result_v4);
    end = clock();
    end_cycle = rdtsc();

    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    mean_time_v4[i] = elapsed;
    mean_cycles_v4[i] = end_cycle - start_cycle;
  }

  // Calculo de la media v4
  double mean_time_v4_result = 0;
  uint64_t mean_cycles_v4_result = 0;

  for (i = 0; i < REPETICIONES; i++) {
    mean_time_v4_result += mean_time_v4[i];
    mean_cycles_v4_result += mean_cycles_v4[i];
  }

  mean_time_v4_result /= REPETICIONES;
  mean_cycles_v4_result /= REPETICIONES;

  printf("\n\n============================================\n");
  printf("\t\tResultados version 4:");
  printf("\nTiempo de ejecución: %f ms\n", mean_time_v4_result);
  printf("Ciclos de reloj: %" PRIu64 "\n", mean_cycles_v4_result);
  printf("============================================\n");

  elapsed = 0;

  // Versión 5
  for (i = 0; i < REPETICIONES; i++) {
    start_cycle = rdtsc();
    start = clock();
    firfilter_v5(vector_coef, vector_in, result_v5);
    end = clock();
    end_cycle = rdtsc();

    elapsed = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    mean_time_v5[i] = elapsed;
    mean_cycles_v5[i] = end_cycle - start_cycle;
  }

  // Calculo de la media v5
  double mean_time_v5_result = 0;
  uint64_t mean_cycles_v5_result = 0;

  for (i = 0; i < REPETICIONES; i++) {
    mean_time_v5_result += mean_time_v5[i];
    mean_cycles_v5_result += mean_cycles_v5[i];
  }

  mean_time_v5_result /= REPETICIONES;
  mean_cycles_v5_result /= REPETICIONES;

  printf("\n\n============================================\n");
  printf("\t\tResultados version 5:");
  printf("\nTiempo de ejecución: %f ms\n", mean_time_v5_result);
  printf("Ciclos de reloj: %" PRIu64 "\n", mean_cycles_v5_result);
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