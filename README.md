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
| TF-IDF con bigramas + regresión logística | 72,32 % | 1,63 puntos porcentuales |
| TF-IDF del texto completo y última oración + regresión logística | 87,53 % | 1,55 puntos porcentuales |

El recall de neutral es 99,5 %, mientras que negativo y positivo tienen 60,4 % y 61,4 %, respectivamente. Las siguientes pruebas buscarán mejorar estas dos clases usando cambios basados en el material del curso.

## Estado del proyecto

Comparamos tres configuraciones con validación cruzada. Combinar el texto completo y la última oración mejoró el promedio en las cinco particiones. En Kaggle, los primeros dos envíos obtuvieron 0.72333 y 0.72777; el tercero queda pendiente de subir. Aún falta probar otras configuraciones, evaluar la elegida en las 2.400 reseñas reservadas y guardar el modelo final.

La primera parte no utiliza redes neuronales, transformers ni embeddings neuronales preentrenados.

## Archivos de envío

Al ejecutar todas las celdas, la sección final entrena cada configuración con las 12.000 reseñas y guarda:

- `submissions/submission_01_tfidf_logreg.csv`
- `submissions/submission_02_tfidf_bigramas_logreg.csv`
- `submissions/submission_03_tfidf_completo_ultima_oracion_logreg.csv`

Cada archivo contiene 3.000 filas con las columnas `id,answer`. Al repetir la ejecución se actualizan los mismos archivos. La generación no los sube a Kaggle; los envíos se realizan manualmente. Los nuevos modelos se agregan al diccionario `modelos_submission` al final del notebook.
