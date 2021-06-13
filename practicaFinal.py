"""
TRABAJO FINAL
Nombre Estudiantes: Victor Diaz Bustos y Yunhao Lin Pan
"""
#  %%
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy.sparse.construct import random
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# Funcion para leer los datos del dataset de regresion
def readData(archivo):
    datos = pd.read_csv(archivo, delim_whitespace=True, header=None)
    d = np.array(datos)

    # Separamos los datos en los conjuntos X e Y
    X = d[:,:-1]
    Y = d[:, -1]

    return X, Y



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
        clf = GridSearchCV(mp['model'], mp['parameters'], cv=5, return_train_score=False, n_jobs=-1, scoring='neg_root_mean_squared_error')
        clf.fit(x_train, y_train)
        scores.append({
            'model' : model_name,
            'best_score' : clf.best_score_,
            'best_parameters': clf.best_params_
        })

    df = DataFrame(scores, columns=["model", "best_score", "best_parameters"])
    return df


def hyper_parameter_tuning_mlp(x_train, y_train):
    # --------------------------- Hyper parameter Tuning ---------------------------
    models_parameters = {
        'MLP':{
            'model': MLPRegressor(solver='lbfgs', max_iter=200, random_state=42),
            'parameters': {
                # Total de neuro
                'hidden_layer_sizes' : [[14, 14], [14, 7], [7, 7], [4, 8], [8, 4], [7, 14]], 
                # 'hidden_layer_sizes' : [[14, 14, 14], [14, 8, 4], [7, 7, 7], [4, 8, 14]],
                # 'hidden_layer_sizes' : [[100, 100],[75, 75],[50, 50], [100, 75], [75, 100], [50, 75], [75, 50]], # , [25,25] ,[18,18], [16, 16], [16, 8],[14, 14], [14, 7], [7, 7], [4, 8], [8, 4], [7, 14]],
                'alpha':[15, 14, 12, 11, 10,9,8,7, 6, 5, 3, 2, 1],
                'learning_rate_init':[0.001, 0.01, 0.1]
                
            }
        }
    } 

    scores = []
    res = None
    for model_name, mp in models_parameters.items():
        clf = GridSearchCV(mp['model'], mp['parameters'], cv=5, return_train_score=False, n_jobs=-1, scoring='neg_root_mean_squared_error')
        clf.fit(x_train, y_train)
        scores.append({
            'model' : model_name,
            'best_score' : clf.best_score_,
            'best_parameters': clf.best_params_
        })
        res = clf.cv_results_
        
    df = DataFrame(scores, columns=["model", "best_score", "best_parameters"])
    return df, res


def hyper_parameter_tuning_rfr(x_train, y_train):
    # --------------------------- Hyper parameter Tuning ---------------------------
    models_parameters = {
        'rfr':{
            'model': RandomForestRegressor(),
            'parameters': {
                'n_estimators':[10,20,50,100,150,200,500],
                'max_depth':[5,7,10,15,20,30,50]
            }
        }
    } 

    scores = []
    res = None
    for model_name, mp in models_parameters.items():
        clf = GridSearchCV(mp['model'], mp['parameters'], cv=5, return_train_score=False, n_jobs=-1, scoring='neg_root_mean_squared_error')
        clf.fit(x_train, y_train)
        scores.append({
            'model' : model_name,
            'best_score' : clf.best_score_,
            'best_parameters': clf.best_params_
        })
        res = clf.cv_results_
        
    df = DataFrame(scores, columns=["model", "best_score", "best_parameters"])
    return df, res


def evolution_cv_score_with_iterations(X_train, Y_train):
    models_parameters = {
        'MLP':{
            'model': MLPRegressor(solver='lbfgs', max_iter=200, random_state=42),
            'parameters': {
                # Total de neuro
                'hidden_layer_sizes' : [[14, 14], [14, 7], [7, 7], [4, 8], [8, 4], [7, 14]], 
                # 'hidden_layer_sizes' : [[14, 14, 14], [14, 8, 4], [7, 7, 7], [4, 8, 14]],
                # 'hidden_layer_sizes' : [[100, 100],[75, 75],[50, 50], [100, 75], [75, 100], [50, 75], [75, 50]], # , [25,25] ,[18,18], [16, 16], [16, 8],[14, 14], [14, 7], [7, 7], [4, 8], [8, 4], [7, 14]],
                'alpha':[15, 14, 12, 11, 10,9,8,7, 6, 5, 3, 2, 1],
                'learning_rate_init':[0.001, 0.01, 0.1]
                
            }
        }
    } 

    scores = []
    res = None
    i = 200
    while i <= 2000:
        for model_name, mp in models_parameters.items():
            clf = GridSearchCV(mp['model'], mp['parameters'], cv=5, return_train_score=False, n_jobs=-1, scoring='neg_root_mean_squared_error')
            clf.fit(X_train, Y_train)
            scores.append({
                'iterations' : i,
                'best_score' : clf.best_score_
            })
            res = clf.cv_results_
        i += 200

    a = np.array(scores)
    x_values = []
    y_values = []
    for i in range(10):
        x_values.append(a[i]['iterations'])
        y_values.append(a[i]['best_score'])

    plt.figure()
    plt.xticks(x_values, x_values)
    plt.plot(x_values, y_values)
    plt.show()

def show_data_distribution(data_frame):
    fig = plt.figure(figsize = (15,20))
    ax = fig.gca()
    data_frame.hist(ax = ax, bins=50)
# Lectura de datos

X, Y= readData('data/housing.data')

# %%


# %%
# print("X:", X)
# print("Y:", Y)

# PARTICION: 70% train, 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)


print("\nMostrando las primeras 5 muestras y las 5 últimas")
d = np.insert(X_train, X_train.shape[1], Y_train, axis=1)
nombre_columnas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
ddd = DataFrame(d, columns=nombre_columnas)
# Mostrando las 5 primeras y las 5 ultimas entradas
print(ddd.head(5))
print("[...]")
print(ddd.tail(5))

print("\nEstadística de los datos de entrenamiento")
print(ddd.describe())

print("\nDistribución de los datos")
show_data_distribution(ddd)

# Correlation matrix
print("\nMatriz de correlaciones")
df = pd.DataFrame(d, columns = nombre_columnas)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, annot_kws={"size":5})
plt.show()


# # Eliminar los outliers o datos extremos 
# print("Samples antes de eliminar outliers: ", X_train.shape[0])
# arr = np.append(X_train, Y_train.reshape(-1, 1), axis=1)
# df = DataFrame(arr)
# z_scores = stats.zscore(df)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# df = df[filtered_entries]
# data = np.array(df)
# X_train = data[:,:-1]
# Y_train = data[:,-1]
# print("Samples después de eliminar outliers: ", X_train.shape[0])


# Estandarizacion de los datos usando el StandardScaler
print("\nNormalizando los datos...\n")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Mostrando las primeras 5 muestras tras normalización")
d = np.insert(X_train, X_train.shape[1], Y_train, axis=1)
nombre_columnas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
ddd = DataFrame(d, columns=nombre_columnas)
# Mostrando las 5 primeras y las 5 ultimas entradas
print(ddd.head(5))
print("[...]")
print(ddd.tail(5))


# Búsqueda de los mejores parámetros para regresión lineal
df = hyper_parameter_tuning_lineal_model(x_train=X_train, y_train=Y_train)
print("\n\n\n")
print("Mostrando los mejores parámetros de regresión lineal, con y sin penalización")
print(df)

# %%
# El modelo ridge funciona ligeramente mejor
clf = linear_model.Ridge(max_iter=10000, alpha=2, fit_intercept=True, solver='lsqr', tol=0.001)
clf.fit(X_train, Y_train)
y_predicted = clf.predict(X_train)

# Metrica de la bondad de los resultados dentro de la muestra
print("\n#################################################")
print(  "##        Linear Regression(L2 penalty)        ##")
print(  "#################################################")
print("Dentro de la muestra")
print("RMSE:", np.sqrt(mean_squared_error(y_true=Y_train, y_pred=y_predicted)))
r2 = r2_score(y_true=Y_train, y_pred=y_predicted)
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1)))
print("R2:",r2)

# R2 ajustado, para comparar entre modelos que no usen la misma cantidad de 
#características
print("R2 ajustado:",adj_r2)

print("\nFuera de la muestra")
# clf.fit(x_test, y_test)
y_predicted = clf.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_predicted)))
r2 = r2_score(y_true=Y_test, y_pred=y_predicted)
adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)))
print("R2:",r2)
print("R2 ajustado:",adj_r2)






# %%
print("\n#################################################")
print("##            Multilayer Perceptron            ##")
print("#################################################")

# evolution_cv_score_with_iterations(X_train, Y_train)

print("Búsqueda de los mejores hiperparámetros")
res, lolo = hyper_parameter_tuning_mlp(X_train, Y_train)
print("Mostrando resultados")
print(res)

# %%
aux = np.array(res['best_parameters'])
print("Los parámetros escogidos son: ")
print(aux[0])
print("\n")


# --------------------- Evolución del error in sample con las iteraciones ----

# rmse_s = []
# r2_s = []
# adj_r2_s = []
# for i in range(1, 1001, 50):
#     MLP = MLPRegressor(solver='lbfgs', alpha=aux[0]['alpha'], hidden_layer_sizes=aux[0]['hidden_layer_sizes'], learning_rate_init=float(aux[0]['learning_rate_init']), max_iter=i, random_state=42)
#     MLP.fit(X_train, Y_train)

   
#     Y_pred = MLP.predict(X_train)
#     ein = np.sqrt( mean_squared_error(Y_train, Y_pred) )
    
#     r2 = r2_score(y_true=Y_train, y_pred=Y_pred)
#     adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)))
    


#     rmse_s.append(ein)
#     r2_s.append(r2)
#     adj_r2_s.append(adj_r2)


# plt.figure()
# plt.plot(np.arange(1, 1001, 50), rmse_s)
# plt.title("Evolución del error respecto a las iteraciones")
# plt.xlabel("Iteraciones")
# plt.ylabel("RMSE")
# plt.show()

# --------------------- ------------------------------------------- ----




MLP = MLPRegressor(solver='lbfgs', alpha=aux[0]['alpha'], hidden_layer_sizes=aux[0]['hidden_layer_sizes'], learning_rate_init=float(aux[0]['learning_rate_init']), max_iter=200)
MLP.fit(X_train, Y_train)

print("\n\nDentro de la muestra")
Y_pred = MLP.predict(X_train)
ein = np.sqrt( mean_squared_error(Y_train, Y_pred) )
print("RMSE: ", ein)
r2 = r2_score(y_true=Y_train, y_pred=Y_pred)
adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)))
print("R2:",r2)
print("R2 ajustado:",adj_r2)

print("\nFuera de la muestra")
Y_pred = MLP.predict(X_test)
eout = np.sqrt( mean_squared_error(Y_test, Y_pred) )
print("RMSE: ", eout)
r2 = r2_score(y_true=Y_test, y_pred=Y_pred)
adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)))
print("R2:",r2)
print("R2 ajustado:",adj_r2)



# print("\n\n")
# scores = np.sqrt( abs(cross_val_score(MLP, X_train, Y_train, cv=5, scoring='neg_mean_squared_error') ) )
# print("Validacion: ", scores)
# print("Eval: ", np.mean(scores))

# %%

print("\n#################################################")
print("##                Random Forest                ##")
print("#################################################")


print("Búsqueda de los mejores hiperparámetros")
res, lolo = hyper_parameter_tuning_rfr(X_train, Y_train)
print("Mostrando resultados")
print(res)

# %%
aux = np.array(res['best_parameters'])
print("Los parámetros escogidos son: ")
print(aux[0])


rfr = RandomForestRegressor(n_estimators=aux[0]['n_estimators'], max_depth=aux[0]['max_depth'])
rfr.fit(X_train, Y_train)

print("\n\nDentro de la muestra")
Y_pred = rfr.predict(X_train)
ein = np.sqrt( mean_squared_error(Y_train, Y_pred) )
print("RMSE: ", ein)
r2 = r2_score(y_true=Y_train, y_pred=Y_pred)
adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)))
print("R2:",r2)
print("R2 ajustado:",adj_r2)

print("\nFuera de la muestra")
Y_pred = rfr.predict(X_test)
eout = np.sqrt( mean_squared_error(Y_test, Y_pred) )
print("RMSE: ", eout)
r2 = r2_score(y_true=Y_test, y_pred=Y_pred)
adj_r2 = (1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)))
print("R2:",r2)
print("R2 ajustado:",adj_r2)

