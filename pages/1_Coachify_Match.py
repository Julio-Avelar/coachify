# pages/1_Coachify_Match.py
import streamlit as st
import pandas as pd
from utils import (
    initialize_session_state, draw_player_card, obtener_jugadores_similares, 
    crear_grafico_radar, strengths_and_weaknesses, get_absolute_metrics_for, 
    format_val_abs, make_html_report_pro
)
import os

# --- ESTILOS ---

def load_css_file(css_file_name):
    """Lee el archivo CSS e inyecta contenido en Streamlit."""
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
load_css_file("./../styles.css")

st.title("Coachify Match")

# ---  CARGA DE DATOS DESDE SESSION STATE ---
initialize_session_state()

# Asignar variables locales
df_referencia = st.session_state['df_referencia']
matriz_features_modelos = st.session_state['matriz_features_modelos']
mapa_id_indice = st.session_state['mapa_id_indice']
df_modelos_plot = st.session_state['df_modelos_plot']
df_stats_absolutas = st.session_state['df_stats_absolutas']


# --- STREAMLIT UI: SIDEBAR y SELECCIÓN ---
st.sidebar.header("Controles de Análisis")

posiciones_disponibles = sorted(matriz_features_modelos.keys())

posicion_seleccionada = st.sidebar.selectbox('1. Selecciona la Posición:', posiciones_disponibles)

df_filtrado_posicion = df_referencia[df_referencia['games.position'] == posicion_seleccionada]
nombres_disponibles = sorted(df_filtrado_posicion['player.lastname'].unique())

if not nombres_disponibles:
    st.error(f"❌ No se encontraron jugadores en df_referencia.csv para la posición **{posicion_seleccionada}**.")
    st.stop() 


# Jugador Objetivo 
nombre_objetivo = st.sidebar.selectbox('2. Jugador Objetivo (Base):', nombres_disponibles)
jugador_obj_data = df_filtrado_posicion[df_filtrado_posicion['player.lastname'] == nombre_objetivo].iloc[0]
id_objetivo = jugador_obj_data['player.id']


# --- DIBUJAR FICHA Y MÉTRICAS ABSOLUTAS ---
draw_player_card(jugador_obj_data, title="Ficha del Jugador Objetivo")

# --- Lógica para BUSCAR SIMILARES ---

st.subheader("Perfil Táctico y Top Matches")

k_similares = st.slider('Número de Matches:', 1, 10, 5)

resultados = obtener_jugadores_similares(
    id_objetivo, posicion_seleccionada, matriz_features_modelos, mapa_id_indice, df_modelos_plot, k=k_similares
)

if isinstance(resultados, pd.DataFrame):
    resultados.columns = ['player.id', 'Similitud_Coseno']
    
    # Preparación de la tabla de similares
    df_obj = df_referencia[df_referencia['player.id'] == id_objetivo].copy()
    df_obj['Similitud_Coseno'] = 1.0
    
    df_sim = pd.merge(resultados, df_referencia, on='player.id', how='left')
    df_final = pd.concat([df_obj, df_sim], ignore_index=True)

    df_display = df_final[['player.lastname', 'team.name', 'player.age', 'games.position']]
    
# Mostrar la tabla en Streamlit con nombres de columna amigables
    st.dataframe(
        df_display.rename(columns={
            'player.lastname': 'Apellido',
            'team.name': 'Equipo',
            'player.age': 'Edad',
            'games.position': 'Posición'
        }), 
        use_container_width=True, 
        hide_index=True
    )

    # strengths & weaknesses
    df_model = df_modelos_plot[posicion_seleccionada]
    strengths, weaknesses = strengths_and_weaknesses(id_objetivo, df_model, top_n=5)
    col_s, col_w = st.columns(2)
    with col_s:
        st.subheader("Fortalezas")
        if not strengths.empty:
            strengths.columns = ['Métrica', 'vs Prom']
            strengths['vs Prom'] = strengths['vs Prom'].apply(lambda x: f"{x:+.2f}")
            st.dataframe(strengths, use_container_width=True, hide_index=True)
        else:
            st.write("No disponible.")
    with col_w:
        st.subheader("Debilidades")
        if not weaknesses.empty:
            weaknesses.columns = ['Métrica', 'vs Prom']
            weaknesses['vs Prom'] = weaknesses['vs Prom'].apply(lambda x: f"{x:+.2f}")
            st.dataframe(weaknesses, use_container_width=True, hide_index=True)
        else:
            st.write("No disponible.")

    st.markdown("### Jugador vs. Similares")
    fig = crear_grafico_radar([id_objetivo], df_referencia, df_final, df_model, posicion_seleccionada)
    st.pyplot(fig)

    # Preparar reporte descargable (HTML)
    with st.expander("Generar Reporte Descargable (HTML)"):
        if st.button("Generar y preparar descarga del reporte HTML"):
            df_abs = get_absolute_metrics_for(id_objetivo, df_procesado=df_stats_absolutas)
            df_abs.columns = ['Métrica', 'Valor']
            
            similares_table = df_display.rename(columns={'player.lastname':'Apellido','team.name':'team','Similitud':'similitud','player.age':'age','games.position':'position'}).copy()
            html = make_html_report_pro(id_objetivo, jugador_obj_data, df_abs if not df_abs.empty else pd.DataFrame(), strengths, weaknesses, similares_table, fig)
            b = html.encode('utf-8')
            st.download_button("Descargar Reporte (HTML)", data=b, file_name=f"reporte_{id_objetivo}.html", mime="text/html")
else:
    st.error(f"Error al buscar similares: {resultados}")