# %%

import pandas as pd
from pandas.core.frame import DataFrame

import numpy as np
import matplotlib.pyplot as plt

def read_data(file_name):
    # Leer datos 
    df = pd.read_csv(file_name,header=None, sep="  ")


    # Convertirlo en un array numpy
    data = np.array(df)
    
    # x es todo menos la ultima columna
    # cambiarlo si queremos predecir NOX en vez de MEDV
    x = data[:,:-1]
    # y es la ultima columna que contiene la etiqueta real
    y = data[:,-1]
    return x, y, df





# Lectura de datos
x, y, df =  read_data("data/housing.data")
