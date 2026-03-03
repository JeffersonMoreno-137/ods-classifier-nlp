import streamlit as st
import joblib
import spacy
from nltk.corpus import stopwords
import os

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Clasificador ODS (ONU) - MLNS",
    page_icon="🌍",
    layout="centered"
)

# Estilo para forzar que los botones midan exactamente lo mismo sin importar el emoji
st.markdown("""
    <style>
        .stButton > button {
            height: 45px !important;
            min-height: 45px !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
            display: flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap !important;
        }
        .stButton > button * {
            white-space: nowrap !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# PREPARACIÓN DE DEPENDENCIAS DEL PIPELINE
# ==========================================
# Debido a que el Pipeline de scikit-learn guardado con joblib utiliza 
# funciones personalizadas en su interior (__main__.pipeline_limpieza), 
# se deben definir estas funciones exactas antes de cargar el modelo.

# ==========================================
# RECURSOS NLP Y FUNCIONES DE SOPORTE
# ==========================================
@st.cache_resource
def load_nlp_resources():
    # Carga el modelo que instalamos vía requirements.txt
    nlp_model = spacy.load("es_core_news_sm")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stops = set(stopwords.words('spanish'))
    except:
        stops = set()
    return nlp_model, stops

# Instanciamos los recursos una sola vez
nlp, stop_words = load_nlp_resources()

# Estas funciones DEBEN estar definidas antes de cargar el modelo .joblib
def limpieza_textos(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    doc = nlp(text)
    tokens_limpios = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
        and token.lemma_ not in stop_words
    ]
    return " ".join(tokens_limpios)

def pipeline_limpieza(textos):
    return [limpieza_textos(t) for t in textos]

# ==========================================
# CARGA DEL MODELO (CON RUTA DINÁMICA)
# ==========================================
@st.cache_resource
def load_ml_model():
    # Detecta la carpeta donde está este script
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "modelo_final_ods.joblib")
    
    try:
        # joblib buscará automáticamente 'pipeline_limpieza' en este mismo archivo
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el modelo en {model_path}")
        return None

with st.spinner("Inicializando inteligencia artificial..."):
    model = load_ml_model()
# ==========================================
# DICCIONARIO ODS
# ==========================================
ODS_NOMBRES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas"
}

# ==========================================
# INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🌍 Clasificador Automático de ODS")
st.markdown("""
<div style="text-align: justify;">
¡Bienvenid@! Esta herramienta utiliza inteligencia artificial para leer tus textos y detectar automáticamente con qué <a href="https://www.un.org/sustainabledevelopment/es/sustainable-development-goals/" target="_blank">Objetivo de Desarrollo Sostenible (ODS) de la ONU</a> se relacionan. 
Solo pega tu párrafo y nuestro sistema identificará si hablas de educación, medio ambiente, igualdad o alguna otra de las 17 metas globales.
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("📝 Ingresa un texto para clasificar:")

# Manejo de estado para el botón de limpiar
if 'texto_input' not in st.session_state:
    st.session_state.texto_input = ""

def clear_text():
    st.session_state.texto_input = ""

user_input = st.text_area(
    "Escribe o pega aquí el documento, reporte o resumen que desees evaluar:", 
    height=200,
    key="texto_input",
    placeholder="Ejemplo: Se busca mejorar el acceso al agua potable y construir sistemas de saneamiento adecuados en comunidades rurales..."
)

col1, col2, col3 = st.columns([1.5, 1.5, 2])
with col1:
    predict_button = st.button("Clasificar 🚀 ", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Borrar Texto 🗑️", on_click=clear_text, use_container_width=True)

# Lógica de Inferencia
if predict_button:
    if not user_input.strip():
        st.warning("⚠️ Por favor ingresa algún texto para clasificar.")
    else:
        if model is None:
            st.error("❌ El modelo no está cargado correctamente. Revisa la ruta.")
        else:
            with st.spinner("Realizando inferencia..."):
                # El Pipeline se encarga de la limpieza, TF-IDF y SVD automáticamente
                prediccion = model.predict([user_input])[0]
                
                st.success("¡Predicción completada exitosamente!")
                
                st.markdown("### Resultado:")
                
                nombre_ods = ODS_NOMBRES.get(prediccion, "Objetivo desconocido")
                
                # Usamos un cuadro de información grande en lugar de st.metric para que no corte el texto
                st.info(f"El texto corresponde al Objetivo de Desarrollo Sostenible:\n\n### 🎯 ODS {prediccion}: {nombre_ods}")                 
                        
                        
st.divider()

# ==========================================
# FOOTER Y LOGOS
# ==========================================
col_logo1, col_space, col_logo2 = st.columns([1, 2, 1])

with col_logo1:
    try:
        st.image("../logo_uniandes.png", use_container_width=True)
    except:
        pass # Si falla cargando desde la raíz, ignora silenciosamente

with col_logo2:
    try:
        st.image("../maia_logo.png", use_container_width=True)
    except:
        pass

st.markdown("<p style='text-align: center; color: gray; font-size: 0.9em;'>Universidad de los Andes - MAIA - Autor: Jefferson Moreno<br>Construído con Streamlit y Scikit-Learn</p>", unsafe_allow_html=True)
