# pages/2_Comparacion_Directa.py
import streamlit as st
import pandas as pd
from utils import (
    initialize_session_state, draw_player_card_2, crear_grafico_radar, 
    strengths_and_weaknesses, get_absolute_metrics_for, format_val_abs
)
import os

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
load_css_file("./../styles.css")

st.title("Face to Face")

# --- 1. CARGA DE DATOS DESDE SESSION STATE ---
initialize_session_state()

# Asignar variables locales
df_referencia = st.session_state['df_referencia']
df_modelos_plot = st.session_state['df_modelos_plot']
df_stats_absolutas = st.session_state['df_stats_absolutas']
matriz_features_modelos = st.session_state['matriz_features_modelos']
posiciones_disponibles = sorted(df_modelos_plot.keys())


# --- 2. UI Y CONTROLES DE COMPARACIÓN ---
st.markdown("Selecciona dos jugadores de la misma posición para comparar sus perfiles tácticos y métricas absolutas.")
st.markdown("---")

# Controles de posición y jugadores
posicion_seleccionada = st.sidebar.selectbox('1. Posición:', posiciones_disponibles)
df_filtrado_posicion = df_referencia[df_referencia['games.position'] == posicion_seleccionada]
nombres_disponibles = sorted(df_filtrado_posicion['player.lastname'].unique())

if len(nombres_disponibles) < 2:
    st.warning("No hay suficientes jugadores en esta posición para una comparación directa.")
    st.stop()

col_obj, col_comp = st.columns(2)

with col_obj:
    nombre_objetivo = st.selectbox('Jugador A (Base):', nombres_disponibles, key='obj_comp')
    jugador_obj_data = df_filtrado_posicion[df_filtrado_posicion['player.lastname'] == nombre_objetivo].iloc[0]
    id_objetivo = jugador_obj_data['player.id']
    draw_player_card_2(jugador_obj_data, title=f"Ficha: {nombre_objetivo}")

# Lista de comparación sin el jugador base
nombres_comparacion = nombres_disponibles[:] 
if nombre_objetivo in nombres_comparacion:
    nombres_comparacion.remove(nombre_objetivo) 

with col_comp:
    if not nombres_comparacion:
        st.warning("No hay otro jugador en esta posición para comparar.")
        st.stop()
        
    nombre_comparacion = st.selectbox('Jugador B (Comparado):', nombres_comparacion, key='comp_comp')
    jugador_comp_data = df_filtrado_posicion[df_filtrado_posicion['player.lastname'] == nombre_comparacion].iloc[0]
    id_comparacion = jugador_comp_data['player.id']
    draw_player_card_2(jugador_comp_data, title=f"Ficha: {nombre_comparacion}")


# --- 3. RADAR COMPARATIVO ---
st.markdown("---")
st.markdown("### Radar Comparativo: Face to Face")
df_model = df_modelos_plot[posicion_seleccionada]

fig = crear_grafico_radar(
    [id_objetivo, id_comparacion], 
    df_referencia,
    None, 
    df_model, 
    posicion_seleccionada
)
st.pyplot(fig)


# --- 4. FORTALEZAS Y DEBILIDADES ---
s_a, w_a = strengths_and_weaknesses(id_objetivo, df_model, top_n=5)
s_b, w_b = strengths_and_weaknesses(id_comparacion, df_model, top_n=5)

st.markdown("---")
st.subheader("Análisis de Fortalezas y Debilidades")
colA, colB = st.columns(2)
with colA:
    st.markdown(f"**Fortalezas de {nombre_objetivo}**")
    if not s_a.empty: 
        s_a.columns = ['Métrica', 'vs Prom']
        s_a['vs Prom'] = s_a['vs Prom'].apply(lambda x: f"{x:+.2f}")
        st.dataframe(s_a, use_container_width=True, hide_index=True)
with colB:
    st.markdown(f"**Fortalezas de {nombre_comparacion}**")
    if not s_b.empty: 
        s_b.columns = ['Métrica', 'vs Prom']
        s_b['vs Prom'] = s_b['vs Prom'].apply(lambda x: f"{x:+.2f}")
        st.dataframe(s_b, use_container_width=True, hide_index=True)


# --- 5. TABLA COMPARATIVA DE MÉTRICAS ABSOLUTAS ---
st.markdown("---")
st.subheader("Métricas Absolutas (Cara a Cara)")

df_abs_a = get_absolute_metrics_for(id_objetivo, df_procesado=df_stats_absolutas)
df_abs_b = get_absolute_metrics_for(id_comparacion, df_procesado=df_stats_absolutas)

if df_abs_a.empty or df_abs_b.empty:
    st.warning("No se pudieron cargar las métricas absolutas para uno o ambos jugadores. Verifique 'df_procesado.csv'.")
else:
    df_abs_a = df_abs_a.rename(columns={'Valor': nombre_objetivo}).set_index('Métrica')
    df_abs_b = df_abs_b.rename(columns={'Valor': nombre_comparacion}).set_index('Métrica')
    
    df_comparativa_abs = pd.concat([df_abs_a, df_abs_b], axis=1).reset_index()
    
    for col in [nombre_objetivo, nombre_comparacion]:
        df_comparativa_abs[col] = df_comparativa_abs[col].apply(format_val_abs)
        
    st.dataframe(df_comparativa_abs, use_container_width=True, hide_index=True)


# --- 6. DESCARGA DE REPORTE ---
if st.button("Generar reporte comparativo (HTML)"):
    html = "<html><body>"
    html += f"<h1>Comparativo: {nombre_objetivo} vs {nombre_comparacion}</h1>"
    
    # Añadir radar (guardar fig)
    import io, base64 # Importar localmente
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    html += f'<img src="data:image/png;base64,{img_b64}" width="700"/>'
    
    html += "<h2>Métricas Absolutas - Jugador Objetivo</h2>"
    html += (df_abs_a.reset_index().to_html(index=False) if not df_abs_a.empty else "<p>No hay métricas absolutas</p>")
    html += "<h2>Métricas Absolutas - Jugador Comparado</h2>"
    html += (df_abs_b.reset_index().to_html(index=False) if not df_abs_b.empty else "<p>No hay métricas absolutas</p>")
    
    html += "</body></html>"
    st.download_button("Descargar comparativo HTML", data=html.encode('utf-8'), file_name=f"comparativo_{id_objetivo}_{id_comparacion}.html", mime="text/html")