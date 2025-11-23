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
import matplotlib.figure

warnings.filterwarnings('ignore')

# 🚨 importante!  AJUSTA ESTA RUTA A TU DIRECTORIO EXACTO DONDE ESTÁN LOS CSVs 🚨
base_path = r'Data/' 

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

#  función para el manejo del estado de sesión
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


# --- FUNCIONES DE ANÁLISIS Y GRÁFICOS ---

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
        # st.error(f"Error: El modelo de {nombre_modelo} solo tiene {N} features.") 
        return plt.figure()
        
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={'projection': 'polar'})
    ax.tick_params(
        axis='x', 
        pad=4, 
        # labelsize=2 
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
            st.warning(f"⚠️ El jugador (ID: {player_id}) no se encuentra en el archivo de modelo '{nombre_modelo}'.")
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

def make_html_report_pro(player_id: int, 
                         player_row: pd.Series, 
                         df_stats_table: pd.DataFrame, 
                         strengths: pd.DataFrame, 
                         weaknesses: pd.DataFrame, 
                         similares_df: pd.DataFrame, 
                         radar_fig: matplotlib.figure.Figure) -> str:
    """
    Generar string HTML para el reporte descargable
    """
    
    # 1. Convertir la figura del radar a Base64
    # -------------------------------------------------------------------
    buf = io.BytesIO()
    radar_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    # Obtener URL de la foto del jugador 
    player_photo_url = player_row.get('player.photo', 'https://placehold.co/100x100/A0AEC0/ffffff?text=No+Photo')

    # 2. Convertir DataFrames a HTML con estilos de tabla limpios
    # -------------------------------------------------------------------
    
    # Clase CSS base para tablas
    TABLE_CSS_CLASS = "styled-table"
    
    # Función auxiliar para aplicar estilos uniformes a las tablas
    def style_dataframe(df, header_map=None):
        if header_map:
            df = df.rename(columns=header_map)
            
        # Aplica estilos básicos para hacerlo limpio y responsivo
        styled_html = (df.style
                       .set_table_attributes(f'class="{TABLE_CSS_CLASS}"')
                       .set_table_styles([
                           {'selector': 'th', 'props': [('background-color', '#4A90E2'), ('color', 'white'), ('font-size', '10px'), ('text-align', 'center')]},
                           {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'center')]}
                       ])
                       .hide(axis='index') # Oculta el índice de pandas
                       .to_html())
        return styled_html.replace('<th>value</th>', '<th>Valor (Z-Score)</th>') # Reemplazo específico para las tablas de Z-Score

    # Generación de HTML para las tablas
    df_stats_html = style_dataframe(df_stats_table)
    strengths_html = style_dataframe(strengths, header_map={'feature': 'Fortaleza', 'value': 'Valor (Z-Score)'})
    weaknesses_html = style_dataframe(weaknesses, header_map={'feature': 'Debilidad', 'value': 'Valor (Z-Score)'})
    similares_html = style_dataframe(similares_df)

    # 3. Estructura HTML y CSS
    # -------------------------------------------------------------------

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Reporte Táctico - {player_row.get('player.lastname','')}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Estilos Globales (simulando un diseño limpio y moderno) */
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f7f6;
            color: #333;
        }}
        .report-container {{
            max-width: 1200px;
            margin: auto;
            background: white;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
        }}
        
        /* Cabecera del Reporte */
        .header-section {{
            display: flex;
            align-items: center;
            border-bottom: 3px solid #4A90E2;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .player-info {{
            margin-left: 20px;
        }}
        .player-name {{
            font-size: 2.2em;
            color: #1a1a1a;
            margin: 0;
        }}
        .player-details {{
            font-size: 1em;
            color: #666;
            margin-top: 5px;
        }}
        .player-details span {{
            font-weight: bold;
            color: #4A90E2;
        }}
        .player-photo {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid #4A90E2;
        }}

        /* Estilos de las Tarjetas/Secciones */
        .section-title {{
            font-size: 1.5em;
            color: #4A90E2;
            border-left: 4px solid #1a1a1a;
            padding-left: 10px;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .content-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        /* Estilos de Tablas de Pandas (CSS de la clase 'styled-table') */
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
            min-width: 400px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden; /* Para las esquinas redondeadas */
        }}
        .styled-table thead tr {{
            background-color: #4A90E2;
            color: #ffffff;
            text-align: left;
        }}
        .styled-table th, .styled-table td {{
            padding: 12px 15px;
        }}
        .styled-table tbody tr {{
            border-bottom: 1px solid #dddddd;
        }}
        .styled-table tbody tr:nth-of-type(even) {{
            background-color: #f3f3f3;
        }}
        .styled-table tbody tr:last-of-type {{
            border-bottom: 2px solid #4A90E2;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
            color: #777;
        }}

        /* Media Query para móvil */
        @media (max-width: 800px) {{
            .content-grid {{
                grid-template-columns: 1fr; /* Una sola columna en móvil */
            }}
            .report-container {{
                padding: 10px;
            }}
            .header-section {{
                flex-direction: column;
                text-align: center;
            }}
            .player-info {{
                margin-left: 0;
                margin-top: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        
        <!-- SECCIÓN DE CABECERA Y DATOS DEL JUGADOR -->
        <div class="header-section">
            <img src="{player_photo_url}" class="player-photo" onerror="this.onerror=null;this.src='https://placehold.co/100x100/A0AEC0/ffffff?text=N/A';" alt="Foto del Jugador"/>
            <div class="player-info">
                <h1 class="player-name">{player_row.get('player.firstname','')} {player_row.get('player.lastname','')}</h1>
                <p class="player-details">
                    <span>Equipo:</span> {player_row.get('team.name','N/A')} &nbsp; | &nbsp; 
                    <span>Liga:</span> {player_row.get('league.name','N/A')} &nbsp; | &nbsp; 
                    <span>Edad:</span> {player_row.get('player.age','N/A')}
                </p>
                <p class="player-details">
                    <span>Altura:</span> {player_row.get('player.height','N/A')} &nbsp; | &nbsp; 
                    <span>Peso:</span> {player_row.get('player.weight','N/A')} &nbsp; | &nbsp; 
                    <span>Nacionalidad:</span> {player_row.get('player.nationality','N/A')}
                </p>
            </div>
        </div>

        <!-- SECCIÓN DE RENDIMIENTO Y ESTADÍSTICAS -->
        <h2 class="section-title">Análisis Táctico (Rendimiento por Posición)</h2>
        
        <div class="content-grid">
            <!-- Columna 1: Gráfico de Radar -->
            <div class="radar-chart">
                <p style="text-align: center; font-style: italic; color: #555;">Comparación de métricas clave vs. la media de jugadores de su posición.</p>
                <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;"/>
            </div>
            
            <!-- Columna 2: Estadísticas Absolutas -->
            <div class="absolute-stats">
                <h3 style="color: #1a1a1a;">Métricas Básicas</h3>
                {df_stats_html}
            </div>
        </div>

        <!-- SECCIÓN DE FORTALEZAS Y DEBILIDADES -->
        <h2 class="section-title">Análisis de Z-Scores (Fortalezas y Debilidades)</h2>
        
        <div class="content-grid">
            <!-- Columna 1: Fortalezas -->
            <div>
                <h3 style="color: #27ae60;">🟢 Fortalezas Clave</h3>
                {strengths_html}
            </div>
            
            <!-- Columna 2: Debilidades -->
            <div>
                <h3 style="color: #e74c3c;">🔴 Debilidades a Mejorar</h3>
                {weaknesses_html}
            </div>
        </div>

        <!-- SECCIÓN DE JUGADORES SIMILARES -->
        <h2 class="section-title">Jugadores con Perfil Simétrico</h2>
        {similares_html}
        
        <!-- FOOTER -->
        <div class="footer">
            <p>Reporte Generado por Coachify: Gemelo Táctico | ID de Jugador: {player_id}</p>
            <p>© {pd.Timestamp.now().year} Coachify Analytics.</p>
        </div>
    </div>
</body>
</html>
    """
    return html

"""
    Muestra la ficha de un jugador    
"""
def draw_player_card(player_data, title="Ficha del Jugador"):
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

def to_html_table_pro(df: pd.DataFrame, title: str, css_class: str) -> str:
    """Convierte un DataFrame a HTML con estilos específicos para el reporte."""
    
    # Mapeo de encabezados para tablas de fortalezas/debilidades
    header_map = {'Métrica': 'Métrica', 'vs Prom': 'Valor (Z-Score)'}
    
    if 'vs Prom' in df.columns:
        df_styled = df.rename(columns=header_map)
    else:
        df_styled = df

    # Aplicar estilos básicos
    styled_html = (df_styled.style
                   .set_table_attributes(f'class="{css_class}"')
                   .set_table_styles([
                       {'selector': 'th', 'props': [('background-color', '#3498db'), ('color', 'white'), ('font-size', '10px'), ('text-align', 'center'), ('padding', '8px')]},
                       {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'center'), ('padding', '8px')]}
                   ])
                   .hide(axis='index')
                   .to_html())
    
    # Envuelve la tabla en un contenedor con título
    return f"""
    <div class="table-container">
        <h3>{title}</h3>
        {styled_html}
    </div>
    """


def make_html_comparison_report(player_a_data: pd.Series, 
                                player_b_data: pd.Series, 
                                radar_fig: matplotlib.figure.Figure, 
                                df_comparativa_abs: pd.DataFrame, 
                                s_a: pd.DataFrame, 
                                w_a: pd.DataFrame, 
                                s_b: pd.DataFrame, 
                                w_b: pd.DataFrame) -> str:
    # 1. Convertir la figura del radar a Base64
    buf = io.BytesIO()
    radar_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    nombre_a = player_a_data.get('player.lastname', 'Jugador A')
    nombre_b = player_b_data.get('player.lastname', 'Jugador B')
    
    # 2. Preparar la tabla de métricas absolutas para el HTML
    # La tabla comparativa absoluta es el foco central.
    abs_table_html = to_html_table_pro(
        df_comparativa_abs, 
        f"Métricas Absolutas - {nombre_a} vs {nombre_b}", 
        "comparative-table"
    )

    # 3. Preparar tablas de Fortalezas/Debilidades individuales
    strengths_a_html = to_html_table_pro(s_a, f"Fortalezas de {nombre_a}", "metric-table green-header")
    weaknesses_a_html = to_html_table_pro(w_a, f"Debilidades de {nombre_a}", "metric-table red-header")
    
    strengths_b_html = to_html_table_pro(s_b, f"Fortalezas de {nombre_b}", "metric-table green-header")
    weaknesses_b_html = to_html_table_pro(w_b, f"Debilidades de {nombre_b}", "metric-table red-header")

    # 4. Estructura HTML y CSS
    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Comparación Táctica: {nombre_a} vs {nombre_b}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Estilos Base del Reporte */
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .report-container {{
            max-width: 1400px;
            margin: auto;
            background: white;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
        }}
        
        /* Encabezado */
        .header {{
            text-align: center;
            padding-bottom: 20px;
            border-bottom: 3px solid #3498db;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin: 0;
        }}
        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
            margin-top: 5px;
        }}

        /* Secciones de Contenido */
        .section-title {{
            font-size: 1.8em;
            color: #34495e;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 35px;
            margin-bottom: 20px;
        }}
        
        /* GRID DE COMPARACIÓN */
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}

        /* Bloques de Fortalezas/Debilidades */
        .analysis-block {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}

        /* Estilos de Tablas (Aplicados por Pandas to_html) */
        .comparative-table, .metric-table {{
            width: 100%;
            border-collapse: collapse;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            border-radius: 6px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .comparative-table tbody tr:nth-of-type(odd) {{ background-color: #f7f9fc; }}
        .comparative-table tbody tr:nth-of-type(even) {{ background-color: #ecf0f1; }}

        .metric-table.green-header th {{ background-color: #2ecc71 !important; }}
        .metric-table.red-header th {{ background-color: #e74c3c !important; }}

        /* Radar Chart */
        .radar-section {{
            text-align: center;
            padding: 20px 0;
            background: #ecf0f1;
            border-radius: 8px;
        }}
        .radar-section img {{
            max-width: 90%;
            height: auto;
            border: 1px solid #bdc3c7;
            border-radius: 8px;
        }}

        /* Media Queries para Responsividad */
        @media (max-width: 1000px) {{
            .comparison-grid {{
                grid-template-columns: 1fr; /* Una columna en pantallas pequeñas */
            }}
        }}
        
        /* Info de Jugadores en Bloque */
        .player-block {{
            padding: 15px;
            border-radius: 8px;
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            text-align: center;
            margin-bottom: 20px;
        }}
        .player-block h2 {{
            color: #3498db;
            margin: 5px 0;
            font-size: 1.5em;
        }}
        .player-block p {{
            margin: 0;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        
        <!-- ENCABEZADO DEL REPORTE -->
        <div class="header">
            <h1>Confrontación Táctica: {nombre_a} vs {nombre_b}</h1>
            <p>Análisis detallado de perfiles en la posición: {player_a_data.get('games.position', 'N/A')}</p>
        </div>

        <!-- 1. BLOQUES DE JUGADOR -->
        <div class="comparison-grid">
            <div class="player-block">
                <h2>{player_a_data.get('player.firstname','')} {nombre_a}</h2>
                <p>Equipo: {player_a_data.get('team.name','N/A')} | Liga: {player_a_data.get('league.name','N/A')}</p>
            </div>
            <div class="player-block">
                <h2>{player_b_data.get('player.firstname','')} {nombre_b}</h2>
                <p>Equipo: {player_b_data.get('team.name','N/A')} | Liga: {player_b_data.get('league.name','N/A')}</p>
            </div>
        </div>

        <!-- 2. RADAR COMPARATIVO -->
        <h2 class="section-title">Rendimiento Relativo (Gráfico de Radar)</h2>
        <div class="radar-section">
            <img src="data:image/png;base64,{img_b64}" alt="Gráfico de Radar Comparativo"/>
            <p style="font-style: italic; font-size: 0.9em; color: #555;">El gráfico ilustra el rendimiento de ambos jugadores en métricas clave respecto al promedio de su posición (centro).</p>
        </div>

        <!-- 3. MÉTRICAS ABSOLUTAS CARA A CARA -->
        <h2 class="section-title">Métricas Absolutas (Comparativa)</h2>
        {abs_table_html}

        <!-- 4. FORTALEZAS Y DEBILIDADES -->
        <h2 class="section-title">Análisis de Z-Scores Individuales</h2>
        <div class="comparison-grid">
            
            <!-- Análisis Jugador A -->
            <div>
                <h3 style="color: #2980b9;">Análisis de {nombre_a}</h3>
                <div class="analysis-block">
                    {strengths_a_html}
                    {weaknesses_a_html}
                </div>
            </div>
            
            <!-- Análisis Jugador B -->
            <div>
                <h3 style="color: #2980b9;">Análisis de {nombre_b}</h3>
                <div class="analysis-block">
                    {strengths_b_html}
                    {weaknesses_b_html}
                </div>
            </div>
        </div>
        
        <!-- FOOTER -->
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.8em; color: #777;">
            <p>Reporte Generado por Coachify: Gemelo Táctico</p>
            <p>© {pd.Timestamp.now().year} Coachify Analytics.</p>
        </div>
    </div>
</body>
</html>
    """
    return html

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