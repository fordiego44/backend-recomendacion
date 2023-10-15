from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
#modelo_lightfm = joblib.load('C:/Users/serch/Documents/Projects/FastAPI/Backend/modelo/recomendacion_colaborativa_modelo_lightfm.plk')
modelo_lightfm = joblib.load('../Backend/modelo/recomendacion_colaborativa_modelo_lightfm.plk')
df_recetas_global = pd.read_csv('../Backend/df/df_Resetas.csv')
df_test = pd.read_csv('../Backend/df/df_test.csv')
#with open('C:/Users/serch/Documents/Projects/FastAPI/Backend/modelo/recomendacion_colaborativa_modelo_lightfm.plk', 'rb') as f:
    #modelo_lightfm = pickle.load(f)

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
    return recomendacion(id)


@app.get("/comedor/")  # Query
async def comedor(id_comedor: int):
    return recomendacion(id_comedor)
    
def filtraRecetaPorIngrediente(lst_ingredientes):
    df_receta_filtro = df_recetas_global[df_recetas_global['ingredientes'].str.contains('|'.join(lst_ingredientes), case=False)]
    # Resetear los índices
    df_receta_ingrediente = df_receta_filtro.reset_index(drop=True)
    #Reasignación de IDs ordenados
    df_receta_ingrediente.loc[:,'id_receta'] = range(len(df_recetas)) #range(1, len(df_recetas) + 1)
    # Imprimir información sobre el DataFrame df_recetas utilizando la función info().    
    return df_receta_ingrediente

def recomendacion(id_comedor: int):  
    TOP_K = 5  
    df_recetas = filtraRecetaPorIngrediente(lst_ingredientes)
    lightfm_predicciones = modelo_lightfm.predict(
        id_comedor, df_test["id_receta"].values
    )
    
    # Agregamos las predicciones obtenidas como una nueva columna en el DataFrame .
    df_test["lightfm_predicciones"] = lightfm_predicciones

    top_recetas_recomendadas = pd.merge(df_test, df_recetas[["id_receta", "receta"]], on=['id_receta'])
    recomendacion_receta = (top_recetas_recomendadas[(top_recetas_recomendadas.id_comedor == id_comedor) & (top_recetas_recomendadas.valoracion == 0)]).sort_values(by='lightfm_predicciones', ascending=False)

    return recomendacion_receta.head(TOP_K)