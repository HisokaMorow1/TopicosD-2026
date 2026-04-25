# Detección de SPAM con Regresión Logística

## 1. Objetivo técnico
Este proyecto implementa un pipeline de clasificación binaria para detectar correos SPAM mediante:
- preprocesamiento de texto,
- vectorización de texto,
- entrenamiento con Regresión Logística,
- evaluación con accuracy.

Archivo principal de trabajo:
- `5_Regresión+Logística+-+Detección+de+SPAM.ipynb`

Dataset local usado por el notebook:
- `processed_data.csv`

---

## 2. Estructura del proyecto
- `5_Regresión+Logística+-+Detección+de+SPAM.ipynb`: notebook principal de SPAM.
- `4_Regresión+Lineal+-+Predicción+del+coste+de+un+incidente+de+seguridad.ipynb`: notebook de otro ejercicio (regresión lineal).
- `processed_data.csv`: datos preprocesados/normalizados para clasificación de correos.

---

## 3. Especificación de datos
### 3.1 Esquema del CSV
Cabecera detectada:
- `label`
- `subject`
- `email_to`
- `email_from`
- `message`

Primera fila observada (ejemplo):
- `label=1` (correo SPAM)
- asunto y mensaje con contenido textual/HTML.

### 3.2 Semántica de etiquetas
- `0`: ham (correo legítimo)
- `1`: spam

### 3.3 Tamaño del dataset
- Archivo grande (millones de filas). Esto impacta memoria y tiempos de cómputo.

Implicación práctica:
- no conviene vectorizar/entrenar con todo el dataset en una sola pasada si el entorno tiene RAM limitada.

---

## 4. Pipeline implementado en el notebook
### 4.1 Carga de datos
- Se carga `processed_data.csv` con pandas.
- Se inspecciona tamaño y primeras filas para validación inicial.

### 4.2 Limpieza y normalización de texto
Componentes principales:
- eliminación de HTML (`MLStripper` + `strip_tags`),
- normalización básica (tabulaciones/saltos de línea/puntuación),
- tokenización,
- eliminación de stopwords en inglés,
- stemming (Porter Stemmer de NLTK).

Clase central:
- `Parser` con métodos `tokenize`, `get_email_body`, `get_email_content`, `parse`.

### 4.3 Construcción de dataset modelable
Función relevante:
- `create_prep_dataset_from_df(dataframe, parser, limit=None)`

Transformación aplicada:
1. combina `subject` + `message`,
2. tokeniza y limpia,
3. recompone texto procesado,
4. arma pares `X` (texto procesado) y `y` (label).

### 4.4 Vectorización
- Se usa `CountVectorizer` de scikit-learn.
- Salida: matriz dispersa de términos (bag-of-words).

### 4.5 Entrenamiento
- Modelo: `LogisticRegression(max_iter=1000)`.
- Entrenamiento inicial con subconjunto (ejemplo de 100) para validación rápida.
- Entrenamiento posterior con mayor volumen para mejorar representatividad.

### 4.6 Predicción y evaluación
- Se vectoriza el conjunto de test con el mismo vectorizador de entrenamiento.
- Se predice con `clf.predict(...)`.
- Métrica reportada: `accuracy_score`.

---

## 5. Dependencias técnicas
Librerías usadas:
- pandas
- scikit-learn
- nltk
- email (stdlib)
- html.parser (stdlib)
- string (stdlib)
- os (stdlib)

Notas de entorno:
- Python 3.10+ recomendado.
- `nltk.download('stopwords')` debe ejecutarse al menos una vez.

Instalación mínima sugerida:
```bash
pip install pandas scikit-learn nltk
```

---

## 6. Ejecución recomendada (orden)
1. Ejecutar imports e instalación de dependencias.
2. Ejecutar funciones auxiliares de parsing/limpieza.
3. Cargar `processed_data.csv`.
4. Generar muestra pequeña y validar pipeline extremo a extremo.
5. Escalar a dataset completo o subconjunto mayor.
6. Entrenar, predecir y evaluar.

Consejo:
- usar lotes/submuestras cuando haya límites de RAM.

---

## 7. Decisiones de diseño y cambios aplicados
El notebook fue adaptado para trabajar con CSV local en lugar de correos raw del corpus original.

Cambios clave:
- Se sustituyó la dependencia de lectura por índices/rutas externas por lectura directa de `processed_data.csv`.
- Se mantuvo la lógica de preprocesamiento textual (tokenización, stopwords, stemming).
- Se añadieron celdas de diagnóstico (tamaños, distribuciones, ejemplos, dimensiones de matrices).
- Se estructuró entrenamiento en dos escalas: prueba rápida y prueba ampliada.

Beneficio principal:
- reproducibilidad local inmediata sin depender del dataset raw externo.

---

## 8. Riesgos técnicos actuales
1. Celdas heredadas del flujo original
Hay celdas que hacen referencia a funciones/variables de un flujo anterior (por ejemplo lectura por índice raw), y pueden no ser coherentes con el flujo de `processed_data.csv`.

2. Posible fuga temporal por split secuencial
El split por posición (`[:N]`, `[N:]`) puede introducir sesgo si los datos están ordenados por tiempo o por fuente.

3. Escalabilidad
`CountVectorizer` sobre corpus masivo puede disparar consumo de RAM por vocabulario grande.

4. Métrica única
Solo accuracy puede ocultar rendimiento deficiente en una clase. Conviene añadir precision/recall/F1 y matriz de confusión.

---

## 9. Mejoras recomendadas
1. Split estratificado y aleatorio:
- `train_test_split(..., stratify=y, random_state=42)`.

2. Pipeline integrado de sklearn:
- `Pipeline([('vect', TfidfVectorizer(...)), ('clf', LogisticRegression(...))])`.

3. Métricas completas:
- `classification_report`, `confusion_matrix`, ROC-AUC.

4. Limpieza de notebook:
- eliminar celdas obsoletas,
- dejar un flujo lineal ejecutable de arriba a abajo.

5. Control de experimento:
- fijar semillas,
- registrar hiperparámetros,
- versionar resultados.

---

## 10. Validaciones mínimas antes de dar por bueno el modelo
- El notebook corre completo sin errores desde kernel limpio.
- El vectorizador de test es exactamente el de train.
- No hay NaN sin tratar en campos de texto.
- Se reportan métricas por clase además de accuracy.
- El tiempo y memoria de entrenamiento son aceptables para el entorno.

---

## 11. Resumen ejecutivo
Este proyecto ya tiene implementado el flujo esencial de NLP clásico para SPAM (limpieza + bag-of-words + regresión logística). Técnicamente funciona como baseline, pero para uso robusto en producción o evaluación académica más sólida se recomienda:
- depurar celdas heredadas,
- robustecer evaluación,
- mejorar estrategia de partición,
- controlar escalabilidad del vectorizado.
