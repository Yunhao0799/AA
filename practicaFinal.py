"""
TRABAJO FINAL
Nombre Estudiantes: Victor Diaz Bustos y Yunhao Lin Pan
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Funcion para leer los datos del dataset de regresion
def readData(archivo):
    datos = pd.read_csv(archivo, delim_whitespace=True, header=None)
    d = np.array(datos)

    # Separamos los datos en los conjuntos X e Y
    X = d[:,:-1]
    Y = d[:, -1]

    return X, Y


# Lectura de datos
X, Y = readData('data/housing.data')

print("X:", X)
print("Y:", Y)

# PARTICION: 70% train, 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
