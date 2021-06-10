"""
TRABAJO FINAL
Nombre Estudiantes: Victor Diaz Bustos y Yunhao Lin Pan
"""

import numpy as np
import pandas as pd

# Funcion para leer los datos del dataset de regresion
def readData(archivo):
    datos = pd.read_csv(archivo, delim_whitespace=True, header=None)
    d = np.array(datos)

    # Separamos los datos en los conjuntos X e Y
    X = d[:,:-1]
    Y = d[:, -1]

    return X, Y


X, Y = readData('housing.data')

print("X:", X)
print("Y:", Y)
