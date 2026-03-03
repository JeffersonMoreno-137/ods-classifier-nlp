# 🌍 Clasificador Automático de ODS

Este proyecto implementa una solución de Machine Learning y Procesamiento de Lenguaje Natural (NLP) basada en el dataset *OSDG-CD* para clasificar automáticamente textos de acuerdo a los 16 Objetivos de Desarrollo Sostenible (ODS) de la ONU evaluados.

## 🚀 Características del Proyecto
* **Scikit-Learn Pipeline:** Pipeline 100% encapsulado. Incluye transformación de limpieza con SpaCy, vectorización TF-IDF y reduccíón de dimensionalidad con TruncatedSVD.
* **Random Forest Classifier:** Modelo multiclase optimizado exhaustivamente con GridSearchCV para maximizar el F1-Score macro debido al desbalanceo de clases natural en estos textos.
* **Interfaz de Usuario (UI):** Aplicativo web fácil de usar desarrollado en Streamlit que carga el modelo `.joblib` consolidado y permite predicciones en tiempo real.

* **Link de la app en Streamlit:** https://uniandes-maia-ods-classifier-nlp-mp-67.streamlit.app/

## 📂 Estructura del repositorio

```text
ODS_Classifier/
│
├── Microproyecto2.ipynb      # Libreta original de entrenamiento (Scikit-Learn, Grillas, NLP)
├── streamlit_app.py          # Aplicativo Web Streamlit para las inferencias del modelo
├── modelo_final_ods.joblib   # Serialización del Pipeline final completo
├── requirements.txt          # Dependencias (Streamlit, SKlearn, NLTK, Spacy, etc.)
└── images/                   # (Opcional) Carpeta para alojar logos o recursos estáticos
```

## ⚙️ Instalación y Uso Local

Para probar la interfaz por tu cuenta, es recomendable utilizar un entorno virtual (venv o conda):

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Nota: El script de Streamlit se encargará internamente de descargar los diccionarios Stop-Words de NLTK y el modelo `es_core_news_sm` de Spacy si no los detecta en el sistema).*

2. **Ejecutar la Interfaz Gráfica:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Interactuar:** Abre el enlace local proporcionado por la terminal (usualmente `http://localhost:8501`), ingresa el texto medioambiental, económico o social que desees y descubre con qué Objetivo Global se asocia. 🎯

---
***Desarrollado para el Microproyecto 2 - MLNS / MAIA de la Universidad de los Andes.***
