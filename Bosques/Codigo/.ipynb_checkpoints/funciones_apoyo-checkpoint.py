import pandas as pd
import random


def entrenamiento_prueba_split(df, tamanio_prueba):
    '''
    Retorna dos dataframes distintos, uno de
    entrenamiento y otro de prueba.

    Parametros:
    df: dataframe completo, sin segmentar.
    tamanio_prueba: tamaño de la muestra de prueba.
    '''
    if isinstance(tamanio_prueba, float):
        tamanio_prueba = round(tamanio_prueba * len(df))

    indices = df.index.tolist()
    prueba_indices = random.sample(population=indices, k=tamanio_prueba)

    df_prueba = df.loc[prueba_indices]
    df_entrenamiento = df.drop(prueba_indices)

    return df_entrenamiento, df_prueba


def determinar_tipo_caracteristica(df):
    tipo_caracteristicas = []
    n_valores_unicos = 15
    for caracteristica in df.columns:
        if caracteristica != 'label':
            valores_unicos = df[caracteristica].unique()
            valor_ejemplo = valores_unicos[0]
            # Verificamos el tipo de caracteristica
            if (isinstance(valor_ejemplo, str)) or (len(valores_unicos) <= n_valores_unicos):
                tipo_caracteristicas.append("categorico")
            else:
                tipo_caracteristicas.append("continuo")
    return tipo_caracteristicas


def calcular_precision(predicciones, labels):
    predicciones_correctas = predicciones == labels
    precision = predicciones_correctas.mean()

    return precision
