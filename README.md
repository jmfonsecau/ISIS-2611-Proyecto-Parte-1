# Proyecto ML - Parte 1

Proyecto de Aprendizaje de Máquina de la Universidad de los Andes. Buscamos clasificar reseñas de productos en español como `positivo`, `negativo` o `neutral`, usando técnicas clásicas y modelos de scikit-learn.

## Integrantes

- Jose Manuel Fonseca — 202122456
- Julian Pinto
- Daniel Vergara

## Archivos

| Archivo | Contenido |
|---|---|
| `notebook_proyecto.ipynb` | Exploración, entrenamiento, evaluación y documentación de los experimentos. |
| `train.csv` | 12.000 reseñas con las columnas `id`, `text` y `label`. |
| `eval.csv` | 3.000 reseñas sin etiqueta para generar las predicciones de Kaggle. |
| `sample_submission.csv` | Ejemplo del formato de envío: `id,answer`. |


## Proceso y resultados actuales

Revisamos valores faltantes, textos vacíos, duplicados, distribución de clases y ejemplos de reseñas. No encontramos problemas que requieran eliminar filas. Algunas reseñas mezclan elogios y críticas, por lo que la clasificación no depende de una sola palabra.

Separamos el 80 % de los datos etiquetados para desarrollar los modelos y reservamos el 20 % para la evaluación final. Usamos una separación estratificada y `random_state=42`.

El primer modelo combina TF-IDF de palabras individuales con regresión logística. Se evalúa mediante cinco particiones estratificadas sobre las 9.600 reseñas de entrenamiento. TF-IDF se ajusta dentro del pipeline en cada partición para evitar usar información de las reseñas evaluadas.

| Modelo | Accuracy promedio CV | Desviación estándar |
|---|---|---|
| TF-IDF + regresión logística | 72,39 % | 1,59 puntos porcentuales |

El recall de neutral es 99,5 %, mientras que negativo y positivo tienen 60,4 % y 61,4 %, respectivamente. Las siguientes pruebas buscarán mejorar estas dos clases usando cambios basados en el material del curso.

## Estado del proyecto

El primer modelo está entrenado y evaluado con validación cruzada. Aún falta comparar otras configuraciones, evaluar la elegida en las 2.400 reseñas reservadas, guardar el modelo final y generar los envíos a Kaggle. No se ha registrado un score de Kaggle para este intento.

La primera parte no utiliza redes neuronales, transformers ni embeddings neuronales preentrenados.
