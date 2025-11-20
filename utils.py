# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import io
import base64

warnings.filterwarnings('ignore')

# 🚨 RUTA CRÍTICA: AJUSTA ESTA RUTA A TU DIRECTORIO EXACTO DONDE ESTÁN LOS CSVs 🚨
base_path = r'C:/Users/elisa/Documents/Proyecto Julio/coachify/Data/' 

# --- CONSTANTES GLOBALES ---
COLS_ABSOLUTAS_RAW = [
    'games.minutes', 'shots.total', 'shots.on', 'goals.total', 'goals.assists',
    'passes.key', 'tackles.interceptions', 'duels.total', 'duels.won',
    'dribbles.attempts', 'dribbles.success', 'fouls.committed', 'cards.yellow'
]

mapeo_posiciones = {
    'Attacker': 'Delantero', 'Midfielder': 'Mediocampista', 'Defender': 'Defensa', 'Goalkeeper': 'Portero',
    'attacker': 'Delantero', 'midfielder': 'Mediocampista', 'defender': 'Defensa', 'goalkeeper': 'Portero'
}

def format_val_abs(val):
    if pd.isna(val): return ''
    if val == int(val): return f"{int(val):,}".replace(",", "_TEMP_").replace(".", ",").replace("_TEMP_", ".")
    return f"{val:,.2f}".replace(",", "_TEMP_").replace(".", ",").replace("_TEMP_", ".")

# --- FUNCIONES DE CARGA Y PREPROCESAMIENTO ---

@st.cache_resource 
def load_data_core():
    """Carga df_referencia y los modelos de radar (df_posicion.csv). Retorna los DataFrames/Dicts."""
    
    file_path_ref = os.path.join(base_path, 'df_referencia.csv')
    
    try:
        df_referencia = pd.read_csv(file_path_ref) 
    except FileNotFoundError:
        st.error(f"Error fatal: No se encontró 'df_referencia.csv' en {file_path_ref}")
        st.stop()
    except Exception as e:
        st.error(f"Error al leer 'df_referencia.csv'. Detalle: {e}")
        st.stop()
        
    df_referencia.columns = [col.strip() for col in df_referencia.columns]
    if 'games.position' not in df_referencia.columns:
        st.error("Error fatal: La columna 'games.position' no se encontró en 'df_referencia.csv'.")
        st.stop() 

    df_referencia['games.position'] = df_referencia['games.position'].astype(str).str.strip().map(mapeo_posiciones)

    def load_model_data(posicion):
        ruta = os.path.join(base_path, f'df_{posicion.lower()}.csv')
        try:
            df = pd.read_csv(ruta)
        except FileNotFoundError:
            st.warning(f"Archivo no encontrado para {posicion}: {ruta}")
            return np.array([]), {}, pd.DataFrame()
        except Exception as e:
            st.warning(f"Error al leer {ruta}. Detalle: {e}")
            return np.array([]), {}, pd.DataFrame()

        df.columns = [col.strip() for col in df.columns]
        cols_to_drop = ['player.id', 'player.age', 'player.height', 'player.weight']
        df_features = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore') 
        df_features = df_features.fillna(0) 
        matriz = df_features.values

        if 'player.id' not in df.columns:
            st.error(f"Error: Columna 'player.id' no encontrada en df_{posicion.lower()}.csv.")
            return np.array([]), {}, pd.DataFrame()

        df['player.id'] = df['player.id'].astype(str)
        id_a_indice = {id_val: idx for idx, id_val in enumerate(df['player.id'])}
        df_plot = df_features.set_index(df['player.id']).select_dtypes(include=np.number)
        
        return matriz, id_a_indice, df_plot

    matriz_features_modelos = {}
    mapa_id_indice = {}
    df_modelos_plot = {}

    for pos in ['Portero', 'Defensa', 'Mediocampista', 'Delantero']:
        if pos in df_referencia['games.position'].unique():
            matriz, mapa, df_plot = load_model_data(pos)
            if matriz.size > 0:
                matriz_features_modelos[pos] = matriz
                mapa_id_indice[pos] = mapa
                df_modelos_plot[pos] = df_plot

    file_path_proc = os.path.join(base_path, 'df_procesado.csv')
    try:
        df_stats_absolutas = pd.read_csv(file_path_proc)
        df_stats_absolutas.columns = [c.strip() for c in df_stats_absolutas.columns]
        df_stats_absolutas = df_stats_absolutas.astype({'player.id': str})
    except Exception as e:
        st.error(f"Error al cargar df_procesado.csv: {e}")
        df_stats_absolutas = pd.DataFrame() 

    df_referencia = df_referencia.astype({'player.id': str, 'games.position': str})

    return df_referencia, matriz_features_modelos, mapa_id_indice, df_modelos_plot, df_stats_absolutas

# Nueva función centralizada para el manejo del estado de sesión
def initialize_session_state():
    """Inicializa y carga todos los datos necesarios en st.session_state si no están presentes."""
    
    if 'data_loaded' not in st.session_state or not st.session_state.get('data_loaded'):
        with st.spinner('Cargando modelos y datos... esto solo ocurre una vez.'):
            try:
                # Llama a la función que SÍ lee los archivos del disco
                (
                    df_referencia, matriz_features_modelos, mapa_id_indice, 
                    df_modelos_plot, df_stats_absolutas
                ) = load_data_core()
                
                # Almacena en el estado de sesión para que todas las páginas lo vean
                st.session_state['df_referencia'] = df_referencia
                st.session_state['matriz_features_modelos'] = matriz_features_modelos
                st.session_state['mapa_id_indice'] = mapa_id_indice
                st.session_state['df_modelos_plot'] = df_modelos_plot
                st.session_state['df_stats_absolutas'] = df_stats_absolutas
                st.session_state['data_loaded'] = True
            
            except Exception as e:
                # El error se maneja en load_data_core, pero aseguramos la parada aquí
                st.session_state['data_loaded'] = False
                st.stop()


def get_absolute_metrics_for(player_id, df_procesado):
    """Devuelve dataframe de métricas absolutas para el player_id con nombres legibles."""
    df_proc = df_procesado
    df_proc.columns = [c.strip() for c in df_proc.columns]
    
    pid = str(player_id)
    if 'player.id' not in df_proc.columns: return pd.DataFrame()

    df_proc['player.id'] = df_proc['player.id'].astype(str)
    row = df_proc[df_proc['player.id'] == pid]
    if row.empty: return pd.DataFrame()
        
    cols_present = [c for c in COLS_ABSOLUTAS_RAW if c in df_proc.columns]
    
    df_out = row[cols_present].iloc[0].to_frame().reset_index().rename(columns={'index':'Métrica', row.index[0]: 'Valor'})
    
    df_out['Métrica'] = df_out['Métrica'].astype(str).apply(lambda x: x.replace('.', ' ').title())
    df_out['Valor'] = pd.to_numeric(df_out['Valor'], errors='coerce').fillna(0)
    
    return df_out


# --- FUNCIONES DE ANÁLISIS Y GRÁFICOS (RESTO DEL CÓDIGO) ---
# ... (El resto de las funciones: obtener_jugadores_similares, crear_grafico_radar, 
# strengths_and_weaknesses, recommend_position, make_html_report, draw_player_card, 
# se mantienen sin cambios y se pegan aquí en utils.py) ...


def obtener_jugadores_similares(player_id_objetivo, posicion, matriz_features_modelos, mapa_id_indice, df_modelos_plot, k=5):
    """Calcula la similitud de coseno y devuelve los K jugadores más similares."""
    matriz_universo = matriz_features_modelos.get(posicion)
    mapa_indices = mapa_id_indice.get(posicion)

    if matriz_universo is None:
        return "No hay matriz de features para esta posición."

    try:
        indice_objetivo = mapa_indices[player_id_objetivo]
        vector_objetivo = matriz_universo[indice_objetivo].reshape(1, -1)
    except KeyError:
        return "ID no encontrado en el universo."

    sim_scores = cosine_similarity(vector_objetivo, matriz_universo)

    df_temp = df_modelos_plot[posicion].reset_index()
    ids_del_modelo = df_temp.iloc[:,0].astype(str)
    
    sim_series = pd.Series(sim_scores[0], index=ids_del_modelo)
    sim_series.index = sim_series.index.astype(str)
    jugadores_similares = sim_series.drop(str(player_id_objetivo), errors='ignore').nlargest(k)

    return jugadores_similares.reset_index().rename(columns={'index':'player.id', 0:'similaridad'})

def crear_grafico_radar(player_ids, df_referencia, df_reporte, df_modelo_plot, nombre_modelo):
    """Genera un gráfico de radar para 1 o 2 jugadores, con recorte de valores."""
    
    df_modelo_plot_clean = df_modelo_plot.select_dtypes(include=np.number)
    columnas_features = df_modelo_plot_clean.columns.tolist() 
    N = len(columnas_features)
    
    if N < 3:
        # st.error(f"Error: El modelo de {nombre_modelo} solo tiene {N} features.") # Solo debe mostrarse en la UI
        return plt.figure()
        
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={'projection': 'polar'})
    ax.tick_params(
        axis='x', 
        pad=4, 
        # labelsize=2 # <--- CAMBIA ESTE VALOR (e.g., a 12 para más grande, o 8 para más pequeño)
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    
    feature_labels = [label.replace('_p90', '').replace('_pct', '%').replace('.', ' ').replace('_', ' ') for label in columnas_features]
    ax.set_xticklabels(feature_labels, fontsize=7)
    ax.set_yticks(np.arange(-3, 4, 1))
    ax.set_yticklabels(["-3", "-2", "-1", "0 (Media)", "+1", "+2", "+3"], color="grey", size=7)
    ax.set_ylim(-3, 3) 
    
    for i, player_id in enumerate(player_ids):
        try:
            player_id_str = str(player_id)
            vector_raw = df_modelo_plot_clean[df_modelo_plot_clean.index.astype(str) == player_id_str].iloc[0].values.flatten()
            vector = np.clip(vector_raw, -3, 3).tolist() 
            
        except (IndexError, KeyError):
            # st.warning(f"⚠️ El jugador (ID: {player_id}) no se encuentra en el archivo de modelo '{nombre_modelo}'.")
            continue
            
        vector += [vector[0]]
        
        jugador_data = df_referencia[df_referencia['player.id'] == player_id_str].iloc[0] 
        nombre_completo = jugador_data['player.lastname']
        
        color = '#069245' if i == 0 else 'red'
        label = f'Jugador Objetivo: {nombre_completo}' if i == 0 else f'Comparación: {nombre_completo}'
        
        ax.plot(angles, vector, linewidth=1, linestyle='solid', color=color, label=label)
        ax.fill(angles, vector, color, alpha=0.1)

    if len(player_ids) == 1 and df_reporte is not None:
        # Añadir promedio de similares
        df_reporte['Similitud_Coseno'] = pd.to_numeric(df_reporte['Similitud_Coseno'], errors='coerce')
        ids_gemelos = df_reporte[df_reporte['Similitud_Coseno'] < 1.0]['player.id'].astype(str).tolist()
        ids_gemelos_validos = [pid for pid in ids_gemelos if pid in df_modelo_plot_clean.index.astype(str).tolist()]
        
        if ids_gemelos_validos:
            vector_promedio_gemelos = df_modelo_plot_clean.loc[ids_gemelos_validos].mean().values.flatten()
            vector_promedio_gemelos = np.clip(vector_promedio_gemelos, -3, 3).tolist()
            vector_promedio_gemelos += [vector_promedio_gemelos[0]] 
            ax.plot(angles, vector_promedio_gemelos, linewidth=1, linestyle='dashed', color='red', label='Promedio de Similares')
            ax.fill(angles, vector_promedio_gemelos, 'red', alpha=0.15)

    ax.set_title(f'Perfil Táctico ({nombre_modelo})', va='bottom', fontsize=8)
    ax.legend(loc='best', bbox_to_anchor=(1.05, 0.5), fontsize=8)
    plt.tight_layout()
    return fig

def strengths_and_weaknesses(player_id, df_modelo_plot, top_n=5):
    """
    Retorna (strengths, weaknesses) como DataFrames con nombre_feature y valor (Z-Score).
    """
    df_num = df_modelo_plot.select_dtypes(include=np.number)
    pid = str(player_id)
    if pid not in df_num.index.astype(str).tolist():
        return pd.DataFrame(), pd.DataFrame()
    
    vector = df_num.loc[df_num.index.astype(str) == pid].iloc[0]
    s = vector.sort_values(ascending=False)
    
    strengths = pd.DataFrame({'feature': s.index[:top_n].tolist(), 'value': s.values[:top_n].tolist()})
    
    weaknesses_data = s.tail(top_n)
    weaknesses = pd.DataFrame({'feature': weaknesses_data.index.tolist()[::-1], 'value': weaknesses_data.values.tolist()[::-1]})
    
    label_func = lambda x: x.replace('_p90','').replace('_pct','%').replace('.', ' ').replace('_',' ')
    strengths['feature'] = strengths['feature'].astype(str).apply(label_func)
    weaknesses['feature'] = weaknesses['feature'].astype(str).apply(label_func)
    
    return strengths, weaknesses

def recommend_position(player_id, df_modelo_plot, matriz_features_modelos):
    """
    Compara el vector del jugador con la media por posición usando cosine_similarity.
    Devuelve lista ordenada (posicion, score).
    """
    pid = str(player_id)
    df_num = df_modelo_plot.select_dtypes(include=np.number)
    
    if pid not in df_num.index.astype(str).tolist():
        return []
        
    vec = df_num.loc[df_num.index.astype(str) == pid].iloc[0].values.reshape(1,-1)
    scores = []
    
    for pos, matriz in matriz_features_modelos.items():
        if matriz is None or matriz.size == 0: continue
            
        mean_pos = np.nanmean(matriz, axis=0).reshape(1,-1)
        
        if vec.shape[1] != mean_pos.shape[1]:
            minlen = min(vec.shape[1], mean_pos.shape[1])
            v = vec[:, :minlen]
            m = mean_pos[:, :minlen]
        else:
            v, m = vec, mean_pos
            
        try:
            score = cosine_similarity(v, m)[0][0]
        except Exception:
            score = 0.0
            
        scores.append((pos, float(score)))
        
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores_sorted

def make_html_report(player_id, player_row, df_stats_table, strengths, weaknesses, similares_df, radar_fig):
    """
    Genera un string HTML para el reporte descargable.
    """
    buf = io.BytesIO()
    radar_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    df_stats_html = df_stats_table.to_html(index=False)
    strengths_html = strengths.to_html(index=False).replace('<th>value</th>', '<th>Valor (Z-Score)</th>')
    weaknesses_html = weaknesses.to_html(index=False).replace('<th>value</th>', '<th>Valor (Z-Score)</th>')
    similares_html = similares_df.to_html(index=False)

    html = f"""
    <html>
    <head><meta charset="utf-8"><title>Reporte - {player_row['player.lastname']}</title></head>
    <body>
    <h1>Reporte: {player_row.get('player.firstname','')} {player_row.get('player.lastname','')}</h1>
    <h3>Equipo: {player_row.get('team.name','')} &nbsp; | &nbsp; Liga: {player_row.get('league.name','')}</h3>
    <img src="data:image/png;base64,{img_b64}" width="600"/>
    <h2>Métricas Absolutas</h2>
    {df_stats_html}
    <h2>Fortalezas (Z-Score > 0)</h2>
    {strengths_html}
    <h2>Debilidades (Z-Score < 0)</h2>
    {weaknesses_html}
    <h2>Jugadores Similares</h2>
    {similares_html}
    <hr>
    <p>Generado por Gemelo Táctico.</p>
    </body>
    </html>
    """
    return html

def draw_player_card(player_data, title="Ficha del Jugador"):
    """
    Muestra la ficha de un jugador con un diseño mejorado.
    Utiliza la clase CSS 'player-card' para el estilo.
    """
    
    # 1. Aplica el estilo base del card
    # st.markdown(f'<div class="player-card">', unsafe_allow_html=True)
    
    # Usamos subheader y markdown para el nombre principal y la posición
    col_data, col_img = st.columns([7, 3])
    with col_data:
        full_name = f"Jugador base: {player_data.get('player.firstname', '')} {player_data.get('player.lastname', 'N/A')}"
        st.markdown(f"<h2>{full_name}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: -15px;'>País del club: {player_data.get('league.country', 'N/A')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: var(--accent-color); margin-top: -15px;'>{player_data.get('games.position', 'Posición N/A')}</h4>", unsafe_allow_html=True)

    with col_img:
        try:
            st.image(player_data['player.photo'], width=160)
        except:
            # Usar un placeholder si la URL de la foto falla
            st.image("https://via.placeholder.com/110x110?text=N/A", width=110)
     
    st.markdown("---")

    # 2. Layout principal: Foto (2) | Datos Básicos (3) | Características Físicas (3) | Logos (2)
    # Total de 10 unidades para un buen espaciado
    col_data1, col_data2, col_logos = st.columns([3, 3, 2])
    
    # COLUMNA 2: DATOS BÁSICOS
    with col_data1:
        st.markdown("<h4>Datos generales:</h4>", unsafe_allow_html=True)
        st.metric("Equipo", player_data['team.name'])
        st.metric("País", player_data['player.nationality'])
        st.metric("Edad", f"{player_data['player.age']} años")

    # COLUMNA 3: CARACTERÍSTICAS FÍSICAS
    with col_data2:
        st.markdown("<h4>Características Físicas</h4>", unsafe_allow_html=True)
        st.metric("Altura", f"{player_data['player.height']} cm")
        st.metric("Peso", f"{player_data['player.weight']} kg")

    # COLUMNA 4: LOGOS
    with col_logos:        
        # Logotipo de la Liga
        try:
            st.markdown("<p style='font-weight: 600;'>Liga:</p>", unsafe_allow_html=True)
            st.image(player_data['league.logo'], width=70, caption=player_data['league.name'])
        except:
            st.warning("Logo Liga N/A.", icon='🌐')
            
        # Logotipo del Equipo
        try:
            st.markdown("<p style='font-weight: 600;'>Equipo:</p>", unsafe_allow_html=True)
            st.image(player_data['team.logo'], width=70, caption=player_data['team.name'])
        except:
            st.warning("Logo Equipo N/A.", icon='🛡️')

    # 3. Cierre del custom card
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# utils.py (Nueva versión de draw_player_card_2)

# utils.py (BLOQUE DE CÓDIGO ACTUALIZADO Y COMPLETO)

def draw_player_card_2(player_data, title="Ficha del Jugador"):
    
    # --- INICIO DE LA TARJETA con la NUEVA CLASE ---
    st.markdown(f'<div class="ficha-jugador-pequeno">', unsafe_allow_html=True)
    
    # --- CABECERA (Nombre vs Foto) ---
    col_data, col_img = st.columns([7, 3])
    with col_data:
        full_name = f"{player_data.get('player.firstname', '')} {player_data.get('player.lastname', 'N/A')}"
        st.markdown(f"<h3>{full_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin-top: -15px;'>País del club: {player_data.get('league.country', 'N/A')}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='color: var(--accent-color); margin-top: -15px;'>{player_data.get('games.position', 'Posición N/A')}</h5>", unsafe_allow_html=True)

    with col_img:
        try:
            st.image(player_data['player.photo'], width=150)
        except:
            st.image("https://via.placeholder.com/150x150?text=N/A", width=150)
            
    st.markdown("---") # Separador visual

    # --- DATOS GENERALES Y FÍSICOS (EN DOS COLUMNAS) ---
    col_data1, col_data2 = st.columns([5, 5])
    with col_data1:
        st.markdown("<h5>Datos generales:</h5>", unsafe_allow_html=True)
        st.markdown(f"<p>{full_name}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>Equipo: {player_data.get('team.name', 'N/A')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>País: {player_data.get('player.nationality', 'N/A')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>Edad: {player_data.get('player.age', 'N/A')} años</p>", unsafe_allow_html=True)
    
    with col_data2:
        st.markdown("<h5>Características Físicas</h5>", unsafe_allow_html=True)
        st.markdown(f"<p>Altura: {player_data.get('player.height', 'N/A')} cm</p>", unsafe_allow_html=True)
        st.markdown(f"<p>Peso: {player_data.get('player.weight', 'N/A')} kg</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True) # Separador pequeño

    # --- LOGOS (UNO JUNTO A OTRO) ---
    st.markdown("<h5>Contexto de Competición</h5>", unsafe_allow_html=True)
    col_logo_liga, col_logo_equipo = st.columns([5, 5]) 

    with col_logo_liga:
        try:
            st.image(player_data['league.logo'], width=70, caption=player_data['league.name'])
        except:
            st.warning("Logo Liga N/A.", icon='🌐')
            
    with col_logo_equipo:
        try:
            st.image(player_data['team.logo'], width=70, caption=player_data['team.name'])
        except:
            st.warning("Logo Equipo N/A.", icon='🛡️')

    # --- CIERRE DE LA TARJETA ---
    st.markdown("</div>", unsafe_allow_html=True)