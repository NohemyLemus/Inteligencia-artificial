import numpy as np
import pandas as pd
import random

from funciones_apoyo import determinar_tipo_caracteristica


def verificar_pureza(data):
    label_columna = data[:, -1]
    valores_unicos = np.unique(label_columna)

    if len(valores_unicos) == 1:
        return True
    else:
        return False


def clasificar_data(data):
    label_columna = data[:, -1]
    valores_unicos, cont_valores_unicos = np.unique(
        label_columna, return_counts=True)

    indice = cont_valores_unicos.argmax()
    clasificacion = valores_unicos[indice]
    return clasificacion


def obtener_posibles_segmentos(data, random_subespacio):
    posibles_segmentos = {}
    _, n_columnas = data.shape
    columnas_indices = list(range(n_columnas - 1))
    
    if random_subespacio and random_subespacio <= len(columnas_indices):
        columnas_indices = random.sample(population=columnas_indices, k=random_subespacio)
    
    for columna_indice in columnas_indices:
        valores = data[:, columna_indice]
        valores_unicos = np.unique(valores)
        posibles_segmentos[columna_indice] = valores_unicos

    return posibles_segmentos


def calcular_entropia(data):
    label_columna = data[:, -1]
    _, cantidad = np.unique(label_columna, return_counts=True)

    probabilidad = cantidad / cantidad.sum()
    entropia = sum(probabilidad * -np.log2(probabilidad))

    return entropia


def calcular_entropia_total(data_abajo, data_arriba):
    n = len(data_abajo) + len(data_arriba)
    p_data_abajo = len(data_abajo) / n
    p_data_arriba = len(data_arriba) / n

    entropia_total = (p_data_abajo * calcular_entropia(data_abajo)
                      + p_data_arriba * calcular_entropia(data_arriba))

    return entropia_total


def determinar_mejor_segmento(data, posibles_segmentos):

    entropia_total = 9999
    for columna_indice in posibles_segmentos:
        for valor in posibles_segmentos[columna_indice]:
            data_abajo, data_arriba = segmentar_data(data,
                                                     columna_segmentar=columna_indice,
                                                     valor_segmentar=valor)

            entropia_total_actual = calcular_entropia_total(
                data_abajo, data_arriba)

            if entropia_total_actual <= entropia_total:
                entropia_total = entropia_total_actual
                mejor_columna_segmento = columna_indice
                mejor_valor_segmento = valor

    return mejor_columna_segmento, mejor_valor_segmento


def segmentar_data(data, columna_segmentar, valor_segmentar):
    valores_columna_seg = data[:, columna_segmentar]

    tipo_de_caracteristica = TIPO_CARACTERISTICA[columna_segmentar]
    if tipo_de_caracteristica == 'continuo':
        data_abajo = data[valores_columna_seg <= valor_segmentar]
        data_arriba = data[valores_columna_seg > valor_segmentar]
    # En el caso de que nuestra caracteristica sea categorica
    else:
        data_abajo = data[valores_columna_seg == valor_segmentar]
        data_arriba = data[valores_columna_seg != valor_segmentar]

    return data_abajo, data_arriba


def algoritmo_arbol(df, contador=0, min_muestras=2, max_profundidad=5, random_subespacio=None):
    # Preparando los datos
    if contador == 0:
        global ENCABEZADOS, TIPO_CARACTERISTICA
        ENCABEZADOS = df.columns
        TIPO_CARACTERISTICA = determinar_tipo_caracteristica(df)
        data = df.values
    else:
        data = df

    # Casos base
    if (verificar_pureza(data)) or (len(data) < min_muestras) or (contador == max_profundidad):
        clasificacion = clasificar_data(data)
        return clasificacion

    # Parte Recursiva
    else:
        contador += 1

        # Llamar funciones de apoyo
        posibles_segmentos = obtener_posibles_segmentos(data, random_subespacio)
        columna_segmento, valor_segmento = determinar_mejor_segmento(
            data, posibles_segmentos)
        data_abajo, data_arriba = segmentar_data(
            data, columna_segmento, valor_segmento)

        # Verificar si existe algun segmento vacio
        if len(data_abajo) == 0 or len(data_arriba) == 0:
            clasificacion = clasificar_data(data)
            return clasificacion

        # Determinar las condiciones del arbol
        caracteristica = ENCABEZADOS[columna_segmento]
        tipo_caracteristica = TIPO_CARACTERISTICA[columna_segmento]
        if tipo_caracteristica == 'continuo':
            pregunta = "{} <= {}".format(caracteristica, valor_segmento)

        # Si nuestra caracteristica es categorica
        else:
            pregunta = "{} == {}".format(caracteristica, valor_segmento)

        # Instanciar arbol
        sub_arbol = {pregunta: []}

        # Encontrar Preguntas
        si_respuesta = algoritmo_arbol(
            data_abajo, contador, min_muestras, max_profundidad, random_subespacio)
        no_respuesta = algoritmo_arbol(
            data_arriba, contador, min_muestras, max_profundidad, random_subespacio)

        # En ciertos casos estas respuestas podrian ser iguales,
        # lo cual no tendria ningun sentido, esto surge cuando
        # se clasifican los datos a pesar de no ser puros

        if si_respuesta == no_respuesta:
            sub_arbol = si_respuesta
        else:
            sub_arbol[pregunta].append(si_respuesta)
            sub_arbol[pregunta].append(no_respuesta)

        return sub_arbol


def clasificar_ejemplo(ejemplo, arbol):
    pregunta = list(arbol.keys())[0]
    caracteristica, operador_comparativo, valor = pregunta.split(" ")

    if operador_comparativo == "<=":
        if ejemplo[caracteristica] <= float(valor):
            respuesta = arbol[pregunta][0]
        else:
            respuesta = arbol[pregunta][1]

    else:
        if str(ejemplo[caracteristica]) == valor:
            respuesta = arbol[pregunta][0]
        else:
            respuesta = arbol[pregunta][1]

    # caso base
    if not isinstance(respuesta, dict):
        return respuesta
    # parte recursiva
    else:
        arbol_residual = respuesta
        return clasificar_ejemplo(ejemplo, arbol_residual)


def prediccion_arbol_decision(df_prueba, arbol):
    predicciones = df_prueba.apply(clasificar_ejemplo, args=(arbol, ), axis=1)
    return predicciones
