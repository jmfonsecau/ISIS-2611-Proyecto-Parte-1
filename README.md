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

Separamos el 80 % de los datos etiquetados para desarrollar los modelos y reservamos el 20 % para la evaluación final, ya realizada sin reajustar parámetros. Usamos una separación estratificada y `random_state=42`.

El primer modelo combina TF-IDF de palabras individuales con regresión logística. Se evalúa mediante cinco particiones estratificadas sobre las 9.600 reseñas de entrenamiento. TF-IDF se ajusta dentro del pipeline en cada partición para evitar usar información de las reseñas evaluadas.

| Modelo | Accuracy promedio CV | Desviación estándar |
|---|---|---|
| TF-IDF + regresión logística | 72,39 % | 1,59 puntos porcentuales |
| TF-IDF con bigramas + regresión logística | 72,32 % | 1,63 puntos porcentuales |
| TF-IDF del texto completo y última oración + regresión logística | 87,53 % | 1,55 puntos porcentuales |
| SVM lineal + TF-IDF de palabras y caracteres por bloques | 88,91 % | 0,95 puntos porcentuales |
| Rasgos estructurales + HistGradientBoosting | 90,91 % | 0,85 puntos porcentuales |
| Detector gramatical flexible + HistGradientBoosting | 90,79 % | 0,80 puntos porcentuales |
| Detector gramatical + stacking + SVM calibrada | 91,13 % | 0,90 puntos porcentuales |
| Stacking con contexto de las opiniones | 91,33 % | 0,54 puntos porcentuales |

En el primer modelo, el recall de neutral fue 99,5 %, frente a 60,4 % y 61,4 % de negativo y positivo. El quinto mejora estos últimos a 88,2 % y 86,4 %.

## Estado del proyecto

Comparamos ocho configuraciones principales. Los scores públicos fueron 0.72333, 0.72777, 0.88000, 0.88444, 0.87333, 0.88666 y **0.90000**. El séptimo sigue siendo el mejor envío público confirmado. El octavo obtuvo 91,33 % local (frente a 91,13 % del séptimo) y está pendiente de subir. Alcanzamos el 90 % exacto en Kaggle; todavía no lo superamos.

El sexto amplía el detector para reconocer adverbios y variantes verbales. Mantiene 90,79 % local y obtiene 90,75 % al añadir adverbios a las reseñas evaluadas; el quinto cae a 48,38 % en esa prueba. No mejora el promedio normal, pero corrige esa fragilidad. Al final del notebook evaluamos el séptimo y el octavo sobre las 2.400 reseñas reservadas, sin usarlas para elegir parámetros. La exploración inicial revisó el archivo completo; esa limitación queda documentada.

La primera parte no utiliza redes neuronales, transformers ni embeddings neuronales preentrenados.

## Archivos de envío

Al ejecutar todas las celdas, la sección final entrena cada configuración con las 12.000 reseñas y guarda:

- `submissions/submission_01_tfidf_logreg.csv`
- `submissions/submission_02_tfidf_bigramas_logreg.csv`
- `submissions/submission_03_tfidf_completo_ultima_oracion_logreg.csv`
- `submissions/submission_04_tfidf_bloques_svm.csv`
- `submissions/submission_05_estructural_hgb.csv`
- `submissions/submission_06_estructural_robusto_hgb.csv`
- `submissions/submission_07_ensemble.csv`
- `submissions/submission_08_stacking_contextual.csv`

Cada archivo contiene 3.000 filas con las columnas `id,answer`. Al repetir la ejecución se actualizan los mismos archivos. La generación no los sube a Kaggle; los envíos se realizan manualmente. Los nuevos modelos se agregan al diccionario `modelos_submission` al final del notebook.

## Últimos modelos

El cuarto usa una SVM lineal con palabras y caracteres sobre el texto completo, el fragmento desde el último conector y la última oración.

El quinto cuenta opiniones, distingue su posición y los conectores, y usa HistGradientBoosting para clasificarlas. La normalización de tildes ya se usaba en el cuarto intento; en el quinto añadimos corrección de escritura y estructura de opiniones.

Las expresiones están definidas para este tipo de reseñas. Pueden hacer optimista la validación y no necesariamente generalizan a otros datos. **Superar 90 % local no garantiza 0.90 en Kaggle.**

El sexto mantiene las mismas características y árboles, pero flexibiliza cómo reconoce las opiniones. La prueba con adverbios mide esa mejora concreta, no la resistencia a cualquier texto nuevo.

El séptimo combina el detector gramatical, seis regresiones logísticas por bloques con árboles como meta-modelo (stacking), y SVM calibrada. Los pesos finales son fijos (50/30/20), elegidos antes de evaluar. Obtuvo 91,13 % local y 0.90000 público. Su entrenamiento tarda más que el de los modelos anteriores. El score público no garantiza el resultado de la tabla privada.

El octavo deja que un meta-modelo combine seis regresiones logísticas por bloques y un clasificador de 165 rasgos estructurales. Mejoró 0,21 puntos porcentuales de CV; no garantiza superar 0.91 en Kaggle.

En las 2.400 reseñas reservadas, el séptimo acertó 2.191 (0.91292) y el octavo 2.187 (0.91125). El octavo no confirmó su ventaja local en esa prueba. No reajustamos modelos con ese resultado y conservamos el séptimo como ganador público confirmado.

Todo el código está dentro del notebook. Se guardan los CSV en `submissions/` y un único `modelo_final.joblib` en la raíz, obligatorio para la entrega.

## Entrega y reproducción

1. Ejecutar el notebook en orden con los tres CSV en la misma carpeta. La ejecución completa verificada tardó unos 16 minutos en este entorno; depende del equipo. La tabla inicial registra las versiones usadas y las huellas de los archivos.
2. Subir `submission_08_stacking_contextual.csv` a Kaggle y registrar su score real en `scores_publicos`. Generar un CSV no cuenta como envío válido.
3. Ejecutar la sección final otra vez para guardar el modelo del mejor score público confirmado. Mientras el octavo esté pendiente, se guarda el séptimo.
4. Entregar el notebook ejecutado y `modelo_final.joblib` en Bloque Neón. El archivo contiene el clasificador y sus datos de reproducción.

Para usar el archivo guardado en otra sesión, ejecutar antes los imports y las definiciones del notebook, luego `paquete = joblib.load("modelo_final.joblib")` y `paquete["modelo"].predict(textos)`. Solo cargar archivos de confianza y con las mismas versiones. La comprobación del notebook verifica que coincide con las 3.000 predicciones del envío ganador. Más información: [persistencia de modelos de scikit-learn](https://scikit-learn.org/stable/model_persistence.html).

## Pendientes de la rúbrica

Confirmar la línea base oficial, el percentil final y completar códigos y aportes reales de los integrantes. El notebook documenta que los porcentajes del enunciado y de la rúbrica publicada difieren; se debe confirmar cuál aplica. Stacking, calibración y boosting son clásicos de scikit-learn, pero no aparecen implementados en las prácticas revisadas: confirmar con el docente que estos modelos y los rasgos estructurales cumplen la interpretación del requisito 4.1. Ningún texto puede garantizar la nota de sustentación ni el resultado privado.
