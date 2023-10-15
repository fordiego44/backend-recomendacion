from fastapi import FastAPI
import joblib
import pandas as pd
from typing import List
import tensorflow as tf
import pickle
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Embedding,Flatten,Multiply,Concatenate,Dense
from tensorflow.keras.optimizers import Adam

app = FastAPI()
                  
#pickle_in = open("C:/Users/serch/Documents/Projects/FastAPI/Backend/modelo/modeloRecomendacionMFMLPickleSAV.sav","rb")
#modelo_ncf = pickle.load(pickle_in)
#modelo_ncf = joblib.load('C:/Users/serch/Documents/Projects/FastAPI/Backend/modelo/recomendacion_colaborativa_modelo_MF_MLP.plk')

# modelo_ncf = {}
#if (modelo_ncf == None): modelo_ncf = joblib.load('C:/Users/serch/Documents/Projects/FastAPI/Backend/modelo/modeloRecomendacionMFMLP.plk')
#print(type(modelo_ncf))
#print(modelo_ncf)

print("YA LO CARGO!!! ")
df_recetas_global = pd.read_csv('../Backend/df/df_Resetas.csv')
df_test = pd.read_csv('../Backend/df/df_test_MF_MLP.csv')

lst_ingredientes = ["Arroz","Frijoles","Tortillas de maíz","Huevo","Leche","Pan","Azúcar","Aceite vegetal",
    "Pollo","Carne de res","Cebolla","Tomate","Chiles","Papa","Zanahoria","Plátano","Manzana","Naranja",
    "Cilantro","Ajo","Canela","Pasta","Atún enlatado"
]
 
df_recetas = []

@app.get("/")
async def root():    
    return "Hola FastAPI"

@app.get("/comedor/{id_comedor}")  # Path
async def comedor(id_comedor: int):
    return recomendacion(id_comedor)


@app.get("/comedor/")  # Query
async def comedor(id_comedor: int):
    return recomendacion(id_comedor)
    
def filtraRecetaPorIngrediente(lst_ingredientes):
    print("FILTRA RECETA")
    df_receta_filtro = df_recetas_global[df_recetas_global['ingredientes'].str.contains('|'.join(lst_ingredientes), case=False)]
    # Resetear los índices
    df_receta_ingrediente = df_receta_filtro.reset_index(drop=True)
    #Reasignación de IDs ordenados
    df_receta_ingrediente.loc[:,'id_receta'] = range(len(df_receta_ingrediente)) #range(1, len(df_recetas) + 1)
    # Imprimir información sobre el DataFrame df_recetas utilizando la función info().    
    return df_receta_ingrediente

def recomendacion(id_comedor: int): 
    filtrado_Colaborativo(573,8932)  
    modelo_ncf = joblib.load('C:/Users/serch/Documents/Projects/FastAPI/Backend/modelo/modeloRecomendacionMFMLP.plk') 
    print("RECOMENDACION") 
    TOP_K = 5  
    df_recetas = filtraRecetaPorIngrediente(lst_ingredientes)
    
    test_dataset, _ = crear_conjunto_datos_tf(df_test, ['valoracion'], validation_split=0, random_seed=None)
    print("CONJUNTO DE DATOS JSON") 
    ncf_predictions = modelo_ncf.predict(test_dataset)
        
    # Agregamos las predicciones obtenidas como una nueva columna en el DataFrame .
    df_test["ncf_predictions"] = ncf_predictions
    print("PREDICCION REALIZADA") 
    mejores_predicciones = df_test.sort_values(by='ncf_predictions', ascending=False)

    top_recetas_recomendadas = pd.merge(mejores_predicciones, df_recetas[["id_receta", "receta"]], on=['id_receta'])
    recomendacion_receta = (top_recetas_recomendadas[(top_recetas_recomendadas.id_comedor == id_comedor) & (top_recetas_recomendadas.valoracion == 0)]).sort_values(by='ncf_predictions', ascending=False)

    return recomendacion_receta.head(TOP_K)

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

