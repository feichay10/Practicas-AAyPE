import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Datos en formato de diccionario (convertidos a milisegundos)
data = {
    "Tamaño datos": [1000, 10000, 1000000, 2000000, 4000000, 8000000],
    "Tiempo Algoritmo 1 (ms)": [0.00646111 * 1000, 0.00833236 * 1000, 1.12617 * 1000, 0.831302 * 1000, 2.78837 * 1000, 4.11432 * 1000],
    "Tiempo Algoritmo 2 (ms)": [0.00651937 * 1000, 0.00693854 * 1000, 0.560478 * 1000, 0.0724803 * 1000, 1.83968 * 1000, 2.33973 * 1000]
}

def leer_datos_archivo(archivo):
    # Inicializar listas para almacenar los datos
    tamanios_datos = []
    tiempos_codigo_1 = []
    tiempos_codigo_2 = []

    # Leer el archivo línea por línea
    with open(archivo, 'r') as f:
        for linea in f:
            # Separar los valores de cada línea
            valores = linea.strip().split(';')
            tamanio = int(valores[0])
            tiempo1 = float(valores[1]) * 1000  # Convertir a milisegundos
            tiempo2 = float(valores[2]) * 1000  # Convertir a milisegundos

            # Añadir los valores a las listas
            tamanios_datos.append(tamanio)
            tiempos_codigo_1.append(tiempo1)
            tiempos_codigo_2.append(tiempo2)

    # Crear el diccionario con los datos
    data = {
        "Tamaño datos": tamanios_datos,
        "Tiempo Algoritmo 1 (ms)": tiempos_codigo_1,
        "Tiempo Algoritmo 2 (ms)": tiempos_codigo_2
    }

    return data

def create_plot(data):
    # Crear un DataFrame de pandas
    df = pd.DataFrame(data)

    # Generar la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(df["Tiempo Algoritmo 1 (ms)"], df["Tamaño datos"], label="Algoritmo 1", marker='o', color='blue')
    plt.plot(df["Tiempo Algoritmo 2 (ms)"], df["Tamaño datos"], label="Algoritmo 2", marker='o', color='red')

    # Añadir títulos y etiquetas
    plt.title("Comparación de Tiempos de Ejecución de Algoritmos")
    plt.xlabel("Tiempo de Ejecución (milisegundos)")
    plt.ylabel("Tamaño de Datos")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Escala logarítmica en el eje y para mejor visualización

    # Ajustar los ticks del eje X a intervalos de 0.2 ms dentro del rango de los datos
    max_time = max(max(df["Tiempo Algoritmo 1 (ms)"]), max(df["Tiempo Algoritmo 2 (ms)"]))
    plt.xticks(ticks=[i for i in range(0, int(max_time) + 1, 200)], labels=[f"{i/1000:.1f}" for i in range(0, int(max_time) + 1, 200)])

    # Mostrar la gráfica
    plt.show()


import argparse

def leer_datos_archivo(archivo):
    # Inicializar listas para almacenar los datos
    tamanios_datos = []
    tiempos_codigo_1 = []
    tiempos_codigo_2 = []

    # Leer el archivo línea por línea
    with open(archivo, 'r') as f:
        for linea in f:
            # Separar los valores de cada línea
            valores = linea.strip().split(';')
            tamanio = int(valores[0])
            tiempo1 = float(valores[1]) * 1000  # Convertir a milisegundos
            tiempo2 = float(valores[2]) * 1000  # Convertir a milisegundos

            # Añadir los valores a las listas
            tamanios_datos.append(tamanio)
            tiempos_codigo_1.append(tiempo1)
            tiempos_codigo_2.append(tiempo2)

    # Crear el diccionario con los datos
    data = {
        "Tamaño datos": tamanios_datos,
        "Tiempo Algoritmo 1 (ms)": tiempos_codigo_1,
        "Tiempo Algoritmo 2 (ms)": tiempos_codigo_2
    }

    return data

def main():
    # Crear el parser de argumentos
    parser = argparse.ArgumentParser(description='Leer datos de un archivo y procesarlos.')
    parser.add_argument('archivo', type=str, help='Ruta al archivo de datos')

    # Parsear los argumentos
    args = parser.parse_args()

    # Leer y procesar los datos del archivo
    data = leer_datos_archivo(args.archivo)

    # Imprimir el diccionario para verificar
    print(data)
    
    # Crear la gráfica
    create_plot(data)

if __name__ == "__main__":
    main()
