"""
TRABAJO FINAL
Nombre Estudiantes: Victor Diaz Bustos y Yunhao Lin Pan
"""
#  %%
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy.sparse.construct import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_colwidth', None)
# Funcion para leer los datos del dataset de regresion
def readData(archivo):
    datos = pd.read_csv(archivo, delim_whitespace=True, header=None)
    d = np.array(datos)

    df = DataFrame(d, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
    # Separamos los datos en los conjuntos X e Y
    X = d[:,:-1]
    Y = d[:, -1]

    return X, Y, df



def hyper_parameter_tuning_lineal_model(x_train, y_train):
    # --------------------------- Hyper parameter Tuning ---------------------------
    models_parameters = {
        'ridge':{
            'model': linear_model.Ridge(),
            'parameters': {
                # Regularization strenght
                'alpha':[9,8,7, 6, 5, 3, 2, 1, 0.1, 0.01],
                'max_iter': [10000, 100000],
                'fit_intercept':[True, False],
                'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'tol':[1e-3, 1e-4, 1e-5] # 1e-3 is default
            }
        },
        'lasso':{
            'model': linear_model.Lasso(),
            'parameters': {
                'alpha':[1, 0.1, 0.01, 0.05, 0.0010],
                'max_iter': [10000, 100000],
                'fit_intercept' : [True, False],
                'selection':['random', 'cyclic'],
                'tol':[1e-3, 1e-4, 1e-5] # 1e-4 is default

            }
        },
        'linear_regression':{
            'model': linear_model.LinearRegression(),
            'parameters': {
                'fit_intercept':[True, False], # Calcular w_0 o no
                'normalize':[False], # No normalizamos los datos ya que se ha estandarizado 
                                    # previamente
                'n_jobs':[-1] #Hacer uso de todos los procesadores
            }
        }



    } 

    scores = []
    for model_name, mp in models_parameters.items():
        clf = GridSearchCV(mp['model'], mp['parameters'], cv=5, return_train_score=False, n_jobs=-1, scoring='neg_mean_squared_error')
        clf.fit(x_train, y_train)
        scores.append({
            'model' : model_name,
            'best_score' : clf.best_score_,
            'best_parameters': clf.best_params_
        })


    df = DataFrame(scores, columns=["model", "best_score", "best_parameters"])
    return df


# Lectura de datos

X, Y, datos_df= readData('data/housing.data')

# %%

print(datos_df)
# %%
# print("X:", X)
# print("Y:", Y)

# PARTICION: 70% train, 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)




# # Eliminar los outliers o datos extremos 
# print("Samples antes de eliminar outliers: ", X_train.shape[0])
# arr = np.append(x_train, y_train.reshape(-1, 1), axis=1)
# df = DataFrame(arr)
# z_scores = stats.zscore(df)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# df = df[filtered_entries]
# data = np.array(df)
# x_train = data[:,:-1]
# y_train = data[:,-1]
# print("Samples después de eliminar outliers: ", x_train.shape[0])


# Estandarizacion de los datos usando el StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Búsqueda de los mejores parámetros para regresión lineal
df = hyper_parameter_tuning_lineal_model(x_train=X_train, y_train=Y_train)
print(df)

# %%
# El modelo ridge funciona ligeramente mejor
clf = linear_model.Ridge(max_iter=10000, alpha=2, fit_intercept=True, solver='lsqr', tol=0.001)
clf.fit(X_train, Y_train)
y_predicted = clf.predict(X_train)

# Metrica de la bondad de los resultados dentro de la muestra
print("Dentro de la muestra")
print("MSE:", mean_squared_error(y_true=Y_train, y_pred=y_predicted))
r2 = r2_score(y_true=Y_train, y_pred=y_predicted)
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1)))
print("R2:",r2)

# R2 ajustado, para comparar entre modelos que no usen la misma cantidad de 
#características
print("R2 ajustado:",adj_r2)

print("\nFuera de la muestra")
# clf.fit(x_test, y_test)
y_predicted = clf.predict(X_test)
print("MSE:", mean_squared_error(y_true=Y_test, y_pred=y_predicted))
r2 = r2_score(y_true=Y_test, y_pred=y_predicted)
adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)))
print("R2:",r2)
print("R2 ajustado:",adj_r2)