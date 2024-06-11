import argparse

def main(input_file1, input_file2, output_file, column_numbers):
    # Leer los datos de los archivos de entrada
    data1 = read_data(input_file1)
    data2 = read_data(input_file2)
    
    # Seleccionar las columnas especificadas
    selected_data1 = select_columns(data1, column_numbers)
    selected_data2 = select_columns(data2, column_numbers)
    
    # Escribir los datos seleccionados en el archivo de salida
    write_output(output_file, selected_data1, selected_data2)

def read_data(file_name):
    data = {}
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip().split(';')
            size = int(line[0])
            times = list(map(float, line[1:]))
            data[size] = times
    return data

def select_columns(data, column_numbers):
    selected_data = {}
    for size, times in data.items():
        selected_data[size] = [times[i] for i in column_numbers]
    return selected_data

def write_output(output_file, data1, data2):
    with open(output_file, 'w') as file:
        # file.write("Tamanio;Tiempo_Codigo_1;Tiempo_Codigo_2\n")
        for size in sorted(data1.keys()):
            times1 = data1[size]
            times2 = data2[size]
            file.write(f"{size};{';'.join(map(str, times1))};{';'.join(map(str, times2))}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para seleccionar columnas de datos de dos archivos CSV y escribirlos en otro archivo CSV.")
    parser.add_argument("input_file1", help="Ruta al primer archivo de entrada CSV")
    parser.add_argument("input_file2", help="Ruta al segundo archivo de entrada CSV")
    parser.add_argument("output_file", help="Ruta al archivo de salida CSV")
    parser.add_argument("column_numbers", nargs="+", type=int, help="Números de las columnas a seleccionar (0 para la primera columna)")
    args = parser.parse_args()

    main(args.input_file1, args.input_file2, args.output_file, args.column_numbers)
