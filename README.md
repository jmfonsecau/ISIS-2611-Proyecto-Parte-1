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
| SVM lineal + TF-IDF de palabras y caracteres por bloques | 88,91 % | 0,95 puntos porcentuales |
| Rasgos estructurales + HistGradientBoosting | 90,91 % | 0,85 puntos porcentuales |

En el primer modelo, el recall de neutral fue 99,5 %, frente a 60,4 % y 61,4 % de negativo y positivo. El quinto mejora estos últimos a 88,2 % y 86,4 %.

## Estado del proyecto

Comparamos cinco configuraciones con validación cruzada. En Kaggle, los primeros cuatro envíos obtuvieron 0.72333, 0.72777, 0.88000 y 0.88444. El quinto supera 90 % en promedio local y repite 90,91 % al cambiar las particiones, pero su score público está pendiente.

También probamos combinar el estructural con SVM calibrada (pesos 70/30): obtuvo el mismo promedio y tardó más. Preferimos el estructural solo. Aún falta evaluar la configuración elegida en las 2.400 reseñas reservadas.

La primera parte no utiliza redes neuronales, transformers ni embeddings neuronales preentrenados.

## Archivos de envío

Al ejecutar todas las celdas, la sección final entrena cada configuración con las 12.000 reseñas y guarda:

- `submissions/submission_01_tfidf_logreg.csv`
- `submissions/submission_02_tfidf_bigramas_logreg.csv`
- `submissions/submission_03_tfidf_completo_ultima_oracion_logreg.csv`
- `submissions/submission_04_tfidf_bloques_svm.csv`
- `submissions/submission_05_estructural_hgb.csv`

Cada archivo contiene 3.000 filas con las columnas `id,answer`. Al repetir la ejecución se actualizan los mismos archivos. La generación no los sube a Kaggle; los envíos se realizan manualmente. Los nuevos modelos se agregan al diccionario `modelos_submission` al final del notebook.

## Últimos modelos

El cuarto usa una SVM lineal con palabras y caracteres sobre el texto completo, el fragmento desde el último conector y la última oración.

El quinto cuenta opiniones, distingue su posición y los conectores, y usa HistGradientBoosting para clasificarlas. La normalización de tildes ya se usaba en el cuarto intento; en el quinto añadimos corrección de escritura y estructura de opiniones.

Las expresiones están definidas para este tipo de reseñas. Pueden hacer optimista la validación y no necesariamente generalizan a otros datos. **90,91 % local no garantiza 0.90 en Kaggle.**

Todo el código está en el notebook. Al ejecutarlo solo se guardan los CSV en `submissions/`; no se crean archivos adicionales de modelos.
