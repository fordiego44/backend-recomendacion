# -*- coding: utf-8 -*-
from routers import ingredients

#!pip install tensorflow lightfm pandas
from fastapi import FastAPI
import pandas as pd
import random
import seaborn as sns
import numpy as np
import json
import datetime
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import List

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Embedding,Flatten,Multiply,Concatenate,Dense
from tensorflow.keras.optimizers import Adam
from typing import List
from scipy import sparse 

app = FastAPI() 


app.df_train=''
app.df_test=''
app.df_recetas_global=''
app.modelo_ncf=''
app.top_ingrediente=''

app.include_router(ingredients.router)

@app.get("/")
async def root():    
    return "Hola FastAPI"

#response_model=list[dict]
"""@app.post("/lista_alimentos")
async def lista_alimentos(data: dict):
    dicc = data["ingredientes"]
    app.top_ingrediente = [i["ingrediente"] for i in dicc]
    return app.top_ingrediente"""

@app.get("/entrena")
async def entrena():     
    app.df_train, app.df_test,app.df_recetas_global  = procesa_dataframe()
    app.modelo_ncf = entrena_modelo(app.df_train)
    app.df_test = realiza_prediccion(app.modelo_ncf, app.df_test)
    return "Modelo y predicción listo "

#@app.get("/prediccion")
#async def prediccion():    
#    app.df_test = realiza_prediccion(app.modelo_ncf, app.df_test)
#    return "prediccion lista"   

@app.get("/metricas")
async def metricas():    
    return evaluar_modelo_colaborativo(app.df_test)    

@app.post("/recomendacion/{id_comedor}")  # Path
async def recomendacion(id_comedor: int, data: dict):          
    dicc = data["ingredients"]    
    app.top_ingrediente = [i["ingrediente"] for i in dicc]  
    
    return top_recomendacion_colaborativa(id_comedor, app.df_test, app.df_recetas_global)


@app.post("/recomendacion/")  # Query
async def recomendacion(id_comedor: int, data: dict):    
    dicc = data["ingredientes"]
    app.top_ingrediente = [i["ingrediente"] for i in dicc]  
    
    return top_recomendacion_colaborativa(id_comedor, app.df_test, app.df_recetas_global)


@app.get("/valida")  # Query
async def valida():
    print(app.df_train)
    print(app.df_test)
    print(app.app.df_recetas_global)
    print(app.modelo_ncf)
    return "VALIDA"

"""toma una matriz y una lista de posibles calificaciones y la convierte en un formato
largo donde cada fila contiene (fila, columna, calificación)"""
def matriz_amplia_a_larga(matriz_amplia: np.array, calificaciones_posibles: List[int]) -> np.array:

    """Esta función interna toma una matriz y una calificación objetivo, encuentra los índices (filas y columnas)
    de las celdas en la matriz que tienen esa calificación y construye una matriz que almacena estos índices
    junto con la calificación correspondiente."""
    def obtener_calificaciones(arr: np.array, calificacion: int) -> np.array:
        # Encuentra los índices de las celdas en la matriz que contienen el valor de calificación objetivo.
        indices = np.where(arr == calificacion)

        # Construye una matriz donde cada fila contiene (fila, columna, calificación) de las celdas con la calificación objetivo.
        return np.vstack(
            (indices[0], indices[1], np.ones(indices[0].size, dtype="int8") * calificacion)
        ).T

    matrices_largas = [] #Se crea una lista vacía para almacenar las matrices largas para cada calificación posible

    """En cada iteración, se llama a la función interna obtener_calificaciones para obtener los índices y
    calificaciones correspondientes a la calificación actual. Estos resultados se agregan a la lista matrices_largas"""
    for calif in calificaciones_posibles:
        # Obtiene los índices y calificaciones para la calificación objetivo y los agrega a la lista de matrices largas.
        matrices_largas.append(obtener_calificaciones(matriz_amplia, calif))

    # Une las matrices largas en una sola matriz de formato largo y la devuelve. Se apilan verticalmente usando np.vstack
    return np.vstack(matrices_largas)




def numpyarray_dataframe(npArray, lstValorUnico):
    # Convertimos la matriz amplia 'npArray' a formato largo usando la función 'matriz_amplia_a_larga' y la lista de valores unicos.
    long_train = matriz_amplia_a_larga(npArray, lstValorUnico)

    # Creamos un DataFrame 'df_train' a partir de la matriz en formato largo, con columnas 'id_comedor', 'id_receta' y 'valoracion'.
    df = pd.DataFrame(long_train, columns=["id_comedor", "id_receta", "valoracion"])    
    return df




def procesa_dataframe():
    #DataFrame df_comedores
                                
    df_comedores = pd.read_csv('../NCF/df/df_comedores.csv')
    """Agregar una nueva columna 'nombre_comedor' al DataFrame df_comedores.
    La nueva columna se crea mediante la aplicación de una función lambda a cada fila del DataFrame.
    La función lambda concatena el valor de la columna 'comedor' y 'nombre', convertidos a cadenas, separados por un espacio. """
    df_comedores['nombre_comedor'] = df_comedores.apply(lambda x: str(x['comedor'])+' '+ str(x['nombre']), axis=1 )
    
    #DataFrame df_recetas    
    #df_recetas_global = pd.read_csv('../NCF/df/df_Resetas.csv')
    df_recetas = pd.read_csv('../NCF/df/df_Resetas.csv')


    """
    canasta_basica_mexico = app.top_ingrediente

    df_receta_filtro = df_recetas_global[df_recetas_global['ingredientes'].str.contains('|'.join(canasta_basica_mexico), case=False)]
    # Resetear los índices
    df_recetas = df_receta_filtro.reset_index(drop=True)
    #Reasignación de IDs ordenados
    df_recetas.loc[:,'id_receta'] = range(len(df_recetas)) #range(1, len(df_recetas) + 1)    
    """


    data_test_interaccion = np.load('../NCF/numpyArray/data_test_interaccion.npy')
    data_train_interaccion = np.load('../NCF/numpyArray/data_train_interaccion.npy')




    """ seleccionar aleatoriamente elementos del df, de tal forma que un mismo elemento puede ser seleccionado varias veces,
    Calcular el número de combinaciones aleatorias deseado. En este caso, se multiplican las filas de df_recetas por 2. """
    seleccion = len(df_recetas)

    """Crear el DataFrame df_valoracion con columnas 'id_comedor' y 'id_receta', usando la función random.choices
    para seleccionar aleatoriamente elementos de los DataFrames df_comedores y df_recetas."""
    df_valoracion = pd.DataFrame({
        'id_comedor': random.choices(df_comedores['id_comedor'], k=seleccion),  # Seleccionar aleatoriamente 'k' veces 'id_comedor' del DataFrame df_comedores.
        'id_receta': random.choices(df_recetas['id_receta'], k=seleccion) # Seleccionar aleatoriamente 'k' veces 'id_receta' del DataFrame df_recetas.
    })




    """ Agregamos la columna valoracion y asignamos valores aleatorios a la columna 'valoracion' de entre 1 y 5 k veces del tamaño del df
    Usamos la función random.choices para generar una lista de valores aleatorios para la columna 'valoracion'.
    La función range(1, 6) genera valores del 1 al 5 (ambos inclusive), y 'k' es el número de elementos a generar.
    'k' se establece como la longitud (cantidad de filas) del DataFrame df_valoracion."""
    df_valoracion['valoracion'] = random.choices(range(1, 6), k=len(df_valoracion))

    # Dividimos el DataFrame df_valoracion en conjuntos de entrenamiento y prueba utilizando train_test_split de scikit-learn.
    train, test = train_test_split(df_valoracion, test_size=0.3)




    """ Crear una tabla pivote a partir del DataFrame 'train', donde los valores son 'valoracion',
    las filas son 'id_comedor' y las columnas son 'id_receta', llenando los valores faltantes con '0'"""
    df_interaccion_train = pd.pivot_table(train, values='valoracion', index='id_comedor', columns='id_receta', fill_value=0)

    # Convertimos el DataFrame 'df_interaccion_train' en un array de NumPy llamado 'dfTrain'.
    dfTrain = np.array(df_interaccion_train)
    dfTrain = dfTrain.astype(int)




    # Convertimos los valores en el array 'dfTrain' en valores binarios (0 o 1), donde se verifica si un valor es mayor que 0 y luego se convierte a "int8" (entero de 8 bits) para conservar la representación binaria eficientemente.
    dfTrain = (dfTrain > 0).astype("int8")

    # Imprime los valores únicos presentes en el array 'dfTrain' después de la conversión a valores binarios.
    unique_ratings = np.unique(dfTrain)




    # Copiar los valores del primer numpy array al segundo
    min_rows, min_cols = min(data_train_interaccion.shape[0], dfTrain.shape[0]), min(data_train_interaccion.shape[1], dfTrain.shape[1])
    dfTrain[:min_rows, :min_cols] = data_train_interaccion[:min_rows, :min_cols]




    df_train = numpyarray_dataframe(dfTrain, unique_ratings)
    
    
    
    
    """ Crear una tabla pivote a partir del DataFrame 'test', donde los valores son 'valoracion',
    las filas son 'id_comedor' y las columnas son 'id_receta', llenando los valores faltantes con 'None'"""
    df_interaccion_test = pd.pivot_table(test, values='valoracion', index='id_comedor', columns='id_receta', fill_value=0)

    # Convertimos el DataFrame 'df_interaccion_test' en un array de NumPy llamado 'dfTest'.
    dfTest = np.array(df_interaccion_test)




    # Copiar los valores del primer numpy array al segundo
    min_rows, min_cols = min(data_test_interaccion.shape[0], dfTest.shape[0]), min(data_test_interaccion.shape[1], dfTest.shape[1])
    dfTest[:min_rows, :min_cols] = data_test_interaccion[:min_rows, :min_cols]



    
    # Convertimos los valores en el array 'dfTest' en valores binarios (0 o 1), donde se verifica si un valor es mayor que 0 y luego se convierte a "int8" (entero de 8 bits) para conservar la representación binaria eficientemente.
    dfTest = (dfTest > 0).astype("int8")
           
    unique_ratings = np.unique(dfTest)    



    
    df_test = numpyarray_dataframe(dfTest, unique_ratings)




    return df_train,df_test,df_recetas







"""Esta función crea un modelo de filtrado colaborativo basado en matrices factorizadas (MF) y
perceptrones multicapa (MLP) para la recomendación de valoraciones. """
def filtrado_Colaborativo(
    num_comedores: int,
    num_recetas: int,
    mf_latent_dimension: int = 4,
    mlp_latent_dimension: int = 32,
    mf_regularization: int = 0,
    mlp_regularization: int = 0.01,
    dense_layers: List[int] = [8, 4],
    layer_regularization: List[int] = [0.01, 0.01],
    activation_function: str = "relu",
) -> keras.Model:

    # Entradas para el modelo
    comedor = Input(shape=(), dtype="int32", name="id_comedor")
    receta = Input(shape=(), dtype="int32", name="id_receta")

    """Se definen las capas de embedding para el modelo de factorización matricial (MF) y
    el modelo de perceptrón multicapa (MLP), cada una de ellas captura representaciones latentes
    para los IDs de comedores y recetas"""

    # Capas de embedding para el modelo de filtrado colaborativo basado en matrices factorizadas (MF)
    mf_comedor_embedding = Embedding(
        input_dim=num_comedores, #definimos el tamaño del vocabulario o número de categorías posibles
        output_dim=mf_latent_dimension, #definimos la dimensión del espacio de embedding, que determina cuántas características latentes se usarán para representar cada comedor
        name="mf_comedor_embedding", #nombre de la capa
        embeddings_initializer="RandomNormal", #inicializmos los pesos de embedding, en este caso, "RandomNormal" para inicializar con valores aleatorios de una distribución normal.
        embeddings_regularizer=l2(mf_regularization), #La regularización aplicada a los pesos de embedding, en este caso, regularización L2
        input_length=1, #longitud de la secuencia de entrada, que en este caso es 1 ya que solo se está pasando un índice de comedor a la vez.
    )
    mf_receta_embedding = Embedding(
        input_dim=num_recetas,
        output_dim=mf_latent_dimension,
        name="mf_receta_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(mf_regularization),
        input_length=1,
    )

    # Capas de embedding para el modelo de filtrado colaborativo basado en perceptrones multicapa (MLP)
    mlp_comedor_embedding = Embedding(
        input_dim=num_comedores,
        output_dim=mlp_latent_dimension,
        name="mlp_comedor_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(mlp_regularization),
        input_length=1,
    )
    mlp_receta_embedding = Embedding(
        input_dim=num_recetas,
        output_dim=mlp_latent_dimension,
        name="mlp_receta_embedding",
        embeddings_initializer="RandomNormal",
        embeddings_regularizer=l2(mlp_regularization),
        input_length=1,
    )

    """Los vectores latentes de comedor y receta obtenidos en el enfoque de MF se aplastan y
    luego Se multiplican componente a componente los vectores latentes del comedor y la receta
    para obtener un único vector latente combinado para el enfoque de factorización."""
    # Obtener vectores latentes MF
    comedor_latent_mf = mf_comedor_embedding(comedor)
    receta_latent_mf = mf_receta_embedding(receta)
    mf_vector = Multiply()([Flatten()(comedor_latent_mf), Flatten()(receta_latent_mf)])

    """Los vectores latentes de comedor y receta obtenidos en el enfoque de MLP se aplastan y
    luego se concatenan para formar un vector latente combinado"""
    # Obtener vectores latentes MLP
    comedor_latent_mlp = mlp_comedor_embedding(comedor)
    receta_latent_mlp = mlp_receta_embedding(receta)
    mlp_vector = Concatenate()([Flatten()(comedor_latent_mlp), Flatten()(receta_latent_mlp)])

    # Capas densas del MLP
    for i, units in enumerate(dense_layers):
        mlp_vector = Dense(
            units,
            activation=activation_function,
            name="dense_layer_{}".format(i),
            kernel_regularizer=l2(layer_regularization[i]),
        )(mlp_vector)

    # Concatenar los vectores latentes MF y MLP
    concatenated_vector = Concatenate()([mf_vector, mlp_vector])

    # Capa de salida para la valoración
    output = Dense(
        1, activation="sigmoid", kernel_initializer="lecun_uniform", name="valoracion"
    )(concatenated_vector)

    # Definir el modelo con las entradas y salida
    model = Model(
        inputs=[comedor, receta],
        outputs=[output],
    )

    return model




def compila_modelo(df_train):
    dfTrain = pd.pivot_table(df_train, values='valoracion', index='id_comedor', columns='id_receta', fill_value=0)
    dfTrain = np.array(dfTrain)
    # Obtenemos el número de comedores y recetas en la matriz dfTrain.
    n_comedores, n_recetas= dfTrain.shape

    #Creamos un modelo de recomendación colaborativa utilizando la función filtrado_Colaborativo con el número de comedores y recetas como parámetros.
    modelo_ncf = filtrado_Colaborativo(n_comedores, n_recetas)

    """ Compilamos el modelo con el optimizador Adam y la función de pérdida "binary_crossentropy".
    También se definen varias métricas para evaluar el rendimiento durante el entrenamiento. """
    modelo_ncf.compile(
        optimizer=Adam(),
        #optimizer=tf.keras.optimizers.RMSprop(),
        #optimizer=tf.keras.optimizers.SGD(),
        loss="binary_crossentropy",
        # definimos varias métricas de evaluación para monitorear el rendimiento del modelo durante el entrenamiento.
        metrics=[tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
                ],
    )

    # Cambiar el nombre del modelo para identificarlo más fácilmente.
    modelo_ncf._name = "recomendacion_colaborativa"

    # Imprimir un resumen del modelo, mostrando la arquitectura de capas y el número de parámetros.
    modelo_ncf.summary()
    
    return modelo_ncf




"""Este código es útil para preparar los conjuntos de datos en el formato adecuado para el entrenamiento y
validación de un modelo de aprendizaje automático en TensorFlow."""
def crear_conjunto_datos_tf(
    dataframe: pd.DataFrame,
    target_columns: List[str],
    validation_split: float = 0.2,
    batch_size: int = 512,
    random_seed=42,
):
   """Crea un conjunto de datos TensorFlow a partir de Pandas DataFrame.
    dfdataframe: El DataFrame de entrada que contiene las características y los objetivos.
    target_columns: Una lista de nombres de columnas correspondientes a los objetivos.
    validation_split: La fracción de los datos que se utilizará para la validación. Por defecto, es 0.2 (20%).
    batch_size: El tamaño del lote para el entrenamiento. Por defecto, es 512.
    random_seed: Una semilla aleatoria para reorganizar los datos. Si se proporciona, los datos se reorganizan. Si es None, los datos no se reorganizan."""

   """Se calcula el número de muestras que se utilizarán para el conjunto de validación basado en el tamaño total del DataFrame y
   la proporción de validación especificada."""
   num_validation_samples = round(dataframe.shape[0] * validation_split)

   """Si se especifica un random_seed, el DataFrame se reorganiza aleatoriamente para garantizar que las muestras de entrenamiento y
   validación sean aleatorias y reproducibles. Lo que ayuda a mejorar el rendimiento del modelo y la generalización"""
   if random_seed:
        # reorganiza aleatoriamente todas las filas del DataFrame con sample. Luego, el DataFrame creado se convierte a un diccionario de series
        x_dict = dataframe.sample(frac=1, random_state=random_seed).to_dict("series")
   else:
        x_dict = dataframe.to_dict("series")

   # Crear un diccionario para las columnas objetivo.
   y_dict = dict()

   # Separar las columnas objetivo del diccionario de entradas.
   for target_column in target_columns:
        # Extraer los targets de x_dict y guardarlos en el diccionario y se eliminan del diccionario de características original (x_dict)
        y_dict [target_column] = x_dict.pop(target_column)

   # Crear un conjunto de datos TensorFlow a partir de las características (x) y los targets (y)
   dataset = tf.data.Dataset.from_tensor_slices((x_dict, y_dict))

   """Se toma una muestra de validación del conjunto de datos y se agrupa en lotes utilizando el método take y batch.
   El resto de las muestras se utilizan para el conjunto de entrenamiento y también se agrupan en lotes."""
   validation_dataset = dataset.take(num_validation_samples).batch(batch_size) # crea un nuevo conjunto de datos que contiene solo los primeros
   train_dataset = dataset.skip(num_validation_samples).batch(batch_size) # crea un nuevo conjunto de datos que contiene los ejemplos restantes

   # Devolver los conjuntos de datos de entrenamiento y validación.
   return train_dataset, validation_dataset




def entrena_modelo(df_train):
    modelo_ncf = compila_modelo(df_train)
    """Se utiliza la función crear_conjunto_datos_tf para crear conjuntos de datos de entrenamiento y
    validación a partir del DataFrame df_train. y ['valoracion'] como argumento target_columns"""
    train_dataset, validation_dataset = crear_conjunto_datos_tf(df_train, ['valoracion'])

    # Commented out IPython magic to ensure Python compatibility.
    # #calcula el tiempo de ejecución de la celda de código completa
    # %%time
    N_EPOCHS = 1
    
    """Se crea una ruta para almacenar los registros de TensorBoard, que incluirán información sobre el progreso del entrenamiento y
    la visualización de histogramas de las capas. """
    #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1)
    
    """Se crea un callback de detención temprana que detendrá el entrenamiento si la métrica de pérdida en el conjunto de validación
    (val_loss) no mejora durante un número de épocas definido en patience. En este caso, patience se establece en 0, lo que significa
    que el entrenamiento se detendrá si no hay mejora inmediata."""
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=0
    )
    
    train_hist = modelo_ncf.fit(
        train_dataset, #conjuntos de datos de entrenamiento
        validation_data=validation_dataset, # conjuntos de datos de validación
        epochs=N_EPOCHS, #número de épocas
        callbacks=[tensorboard_callback], #los callbacks (en este caso, TensorBoard)
        verbose=1, #muestra información detallada del progreso del entrenamiento
    )

    return modelo_ncf


# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
#importamos extension de googleColab para tensorboard
# %tensorboard --logdir logs
#lazamos tensorbor e indicamos la carpeta donde se encuentran los logs




def realiza_prediccion(modelo_ncf, df_test):        
    """ Se utiliza la función crear_conjunto_datos_tf para crear un conjunto de datos de prueba a partir del DataFrame df_test y
    ['valoracion'] como argumento target_columns"""
    test_dataset, _ = crear_conjunto_datos_tf(df_test, ['valoracion'], validation_split=0, random_seed=None)




    # Commented out IPython magic to ensure Python compatibility.
    # %%time
    """Utiliza el modelo modelo_ncf para realizar predicciones en el conjunto de datos de prueba test_dataset utilizando el método predict."""
    ncf_predictions = modelo_ncf.predict(test_dataset)
    
    """Agregamos las predicciones obtenidas en una nueva columna llamada "ncf_predictions" en el DataFrame df_test.
    Cada valor en esta columna representa la predicción del modelo para una muestra específica en el conjunto de datos de prueba."""
    df_test["ncf_predictions"] = ncf_predictions
    
    return df_test




def evaluar_modelo_colaborativo(df_test):
    """Utiliza el método describe() del DataFrame df_test para calcular estadísticas descriptivas, incluyendo la desviación estándar
    de la columna "ncf_predictions"."""
    std = df_test.describe().loc["std", "ncf_predictions"]

    """Verifica si la desviación estándar calculada es menor a 0.01. Si es cierto, lanza una excepción ValueError con un mensaje
    indicando que las predicciones del modelo tienen una desviación estándar menor a 0.01."""
    if std < 0.01:
        raise ValueError("Las predicciones del modelo tienen una desviación estándar inferior a 1e-2.")




    """Utilizamos el método pivot del DataFrame df_test para reorganizar los datos. Los índices "id_comedor" se convierten en las
    filas de la matriz resultante, las columnas "id_receta" se convierten en las columnas de la matriz, y los valores
    de las predicciones "ncf_predictions" se llenan en la matriz."""
    df_ncf_predictions = df_test.pivot(
        index="id_comedor", columns="id_receta", values="ncf_predictions"
    ).values

    """Imprimimos las predicciones del modelo de recomendación colaborativa obtenidas a través del método pivot.
    Cada fila de la matriz representa las predicciones para un comedor específico y cada columna representa las
    predicciones para una receta específica."""
    print("Predicciones de filtrado colaborativo",df_ncf_predictions)




    """Este bloque de código se utiliza para calcular y mostrar las métricas de precisión y recall para el modelo de recomendación colaborativa
    en función del valor de K elegido, que indica la cantidad de elementos principales considerados en el cálculo de estas métricas."""

    """ Definimos el valor de TOP_K para la métrica Precision y Recall.
    Representa la cantidad de elementos principales que se considerarán al calcular estas métricas."""
    TOP_K = 5
    dfTest = pd.pivot_table(df_test, values='valoracion', index='id_comedor', columns='id_receta', fill_value=0)
    dfTest = np.array(dfTest)

    # Se crean instancias de las métricas de precisión y recall de TensorFlow, configuradas con el valor de K definido anteriormente.
    precision_ncf = tf.keras.metrics.Precision(top_k=TOP_K)
    recall_ncf = tf.keras.metrics.Recall(top_k=TOP_K)

    """actualizamos el estado de las métricas de precisión y recall utilizando los datos de prueba (dfTest) y
    las predicciones del modelo (df_ncf_predictions). Las métricas se calculan comparando las predicciones con los datos reales."""
    precision_ncf.update_state(dfTest, df_ncf_predictions)
    recall_ncf.update_state(dfTest, df_ncf_predictions)

    """imprimimos los resultados de las métricas calculadas. Se muestra la precisión y el recall calculados con el valor de K especificado.
    Los valores de precisión y recall se obtienen mediante el método result().numpy() de las instancias de las métricas."""
    print(
        f"Para K = {TOP_K}, tenemos una precisión de {precision_ncf.result().numpy():.5f} y un recall de {recall_ncf.result().numpy():.5f}",
    )


def top_recomendacion_colaborativa(id_comedor, df_test, df_recetas_global, TOP_K=5):
    
    
    canasta_basica_mexico = app.top_ingrediente    

    df_receta_filtro = df_recetas_global[df_recetas_global['ingredientes'].str.contains('|'.join(canasta_basica_mexico), case=False)]    
    # Resetear los índices
    df_recetas = df_receta_filtro.reset_index(drop=True)    
    #Reasignación de IDs ordenados
    df_recetas.loc[:,'id_receta'] = range(len(df_recetas)) #range(1, len(df_recetas) + 1)        
    
    
    mejores_predicciones = df_test.sort_values(by='ncf_predictions', ascending=False)    
    top_recetas_recomendadas = pd.merge(mejores_predicciones, df_recetas[["id_receta", "receta","ingredientes","preparacion"]], on=['id_receta'])
    recomendacion_receta = (top_recetas_recomendadas[(top_recetas_recomendadas.id_comedor == id_comedor) & (top_recetas_recomendadas.valoracion == 0)]).sort_values(by='ncf_predictions', ascending=False)
    
    print(recomendacion_receta.head(TOP_K))
    # Convertir el DataFrame a JSON
    json_recomendacion = {"recetas":json.loads(recomendacion_receta.head(TOP_K).to_json(orient='records'))}

    return json_recomendacion


"""
import joblib
import pickle
joblib.dump(modelo_ncf, "modeloRecomendacionMFMLP.plk")

model = joblib.load("/content/modeloRecomendacionMFMLP.plk")

ncf_predictions = model.predict(test_dataset)

# Agregamos las predicciones obtenidas como una nueva columna en el DataFrame .
df_test["ncf_predictions"] = ncf_predictions

mejores_predicciones = df_test.sort_values(by='ncf_predictions', ascending=False)

top_recetas_recomendadas = pd.merge(mejores_predicciones, df_recetas[["id_receta", "receta"]], on=['id_receta'])
recomendacion_receta = (top_recetas_recomendadas[(top_recetas_recomendadas.id_comedor == 55) & (top_recetas_recomendadas.valoracion == 0)]).sort_values(by='ncf_predictions', ascending=False)

print(recomendacion_receta.head(TOP_K))

"""