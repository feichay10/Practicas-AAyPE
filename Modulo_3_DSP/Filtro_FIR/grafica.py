import matplotlib.pyplot as plt

# Datos de las versiones base
versiones_base = {
    'off': 0.686350,
    'O0': 0.653410,
    'O1': 0.355950,
    'O2': 0.322680,
    'O3': 0.321940
}

# Datos de las versiones 1 a 5
versiones = {
    '1': {'off': 0.542620,
          'O0': 0.539160,
          'O1': 0.294340,
          'O2': 0.102240,
          'O3': 0.028110},
          
    '2': {'off': 0.590532,
          'O0': 0.587926,
          'O1': 0.120157,
          'O2': 0.097767,
          'O3': 0.031861},
          
    '3': {'off': 0.516320,
          'O0': 0.541410,
          'O1': 0.430710,
          'O2': 0.081840,
          'O3': 0.078820},
          
    '4': {'off': 0.687710,
          'O0': 0.591850,
          'O1': 0.159950,
          'O2': 0.107720,
          'O3': 0.029960},
          
    '5': {'off': 0.134570,
          'O0': 0.129370,
          'O1': 0.023110,
          'O2': 0.020160,
          'O3': 0.019010}
}

# Preparar los datos para graficar
versiones_nombres = ['base', '1', '2', '3', '4', '5']
versiones_labels = ['off', 'O0', 'O1', 'O2', 'O3']
tiempos = [[versiones_base[label] for label in versiones_labels]]

for v in versiones_nombres[1:]:
    tiempos.append([versiones[v][label] for label in versiones_labels])

# Crear la figura y el subplot para los tiempos
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar tiempos
for tiempo, nombre in zip(tiempos, versiones_nombres):
    ax.plot(versiones_labels, tiempo, marker='o', label=f'Versión {nombre}')

ax.set_title('Tiempo de ejecución por versión y nivel de optimización')
ax.set_xlabel('Optimización')
ax.set_ylabel('Tiempo (ms)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
