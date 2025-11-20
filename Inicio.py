# Coachify.py
import streamlit as st
import pandas as pd
from utils import initialize_session_state # Importamos la función de setup
import os # Necesitas importar 'os' para manejar rutas y verificar archivos

st.set_page_config(layout="wide", page_title="Coachify: Análisis Táctico")
st.markdown(
    """
    <style>
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
        max-width: 1200px !important; /* Adjust this value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- ESTILOS ---

def load_css_file(css_file_name):
    """Lee el archivo CSS e inyecta su contenido en Streamlit."""
    css_path = os.path.join(os.path.dirname(__file__), css_file_name)
    
    if not os.path.exists(css_path):
        st.warning(f"⚠️ Archivo de estilos '{css_file_name}' no encontrado en la ruta: {css_path}")
        return
        
    try:
        with open(css_path, encoding='utf-8') as f: 
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al leer o inyectar el archivo CSS: {e}")

# Llama a la función para cargar los estilos globales
load_css_file("styles.css")

# --- 1. CARGA DE DATOS CENTRALIZADA ---
# Se asegura que la sesión de Streamlit tenga todos los datos cargados antes de continuar.
initialize_session_state()

# Introducción
st.markdown("""
## Bienvenido

**Gemelo Táctico** es una herramienta avanzada de análisis de jugadores de fútbol que utiliza 
técnicas de machine learning para identificar similitudes entre jugadores y proporcionar 
insights profundos sobre su rendimiento.

### ¿Qué puedes hacer?

""")

# Features en columnas
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>Encuentra el Match</h3>
        <p>Encuentra jugadores con perfiles tácticos similares</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>Face to Face</h3>
        <p>Análisis cara a cara de métricas y perfiles tácticos</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>Visualizar Perfiles</h3>
        <p>Gráficos de radar interactivos con métricas estandarizadas</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Instrucciones
st.markdown("""
## Cómo empezar

1. **Navega** a la página de **Coachify Match** o **Face to Face** en el sidebar
2. **Selecciona** una posición (Portero, Defensa, Mediocampista, Delantero)
3. **Elige** un jugador objetivo
4. **Explora** los gemelos tácticos o compara directamente con otro jugador

### Metodología

Nuestro sistema utiliza:
- **Estandarización Z-Score** por posición y liga
- **Similitud de Coseno** para encontrar perfiles similares
- **Métricas por 90 minutos (P90)** para comparaciones justas
- **Análisis multidimensional** de fortalezas y debilidades

""")

# Sidebar info
st.sidebar.markdown("""
### ℹ️ Información
Esta es la página principal del sistema.

**Para comenzar el análisis:**
→ Ve a la página **⚽ Análisis**
""")

# Footer
st.markdown("""
<div class="footer">
    <p>Desarrollado con ❤️ usando Streamlit | © 2025 Coachify | Julio Avelar </p>
</div>
""", unsafe_allow_html=True)
