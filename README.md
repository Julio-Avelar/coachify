# ⚽ Coachify - Gemelo Táctico

Sistema inteligente de análisis de jugadores de fútbol usando Machine Learning.

## 📁 Estructura del Proyecto

```
coachify/
│
├── app.py                     # Página principal (Home)
├── styles.css                 # Estilos CSS globales
├── requirements.txt           # Dependencias
│
├── pages/                     # Páginas de Streamlit
│   ├── 1_🏠_Home.py          # Homepage (opcional si usas app.py)
│   ├── 2_⚽_Analisis.py      # Dashboard de análisis
│   └── 3_ℹ️_Ayuda.py         # Página de ayuda
│
├── utils/                     # Módulos de utilidades
│   ├── __init__.py           # Inicializador del módulo
│   ├── data_loader.py        # Funciones de carga de datos
│   ├── analysis.py           # Funciones de análisis
│   └── visualizations.py     # Funciones de visualización
│
└── Data/                      # Tu carpeta de datos CSV
    ├── df_referencia.csv
    ├── df_procesado.csv
    ├── df_delantero.csv
    ├── df_defensa.csv
    ├── df_mediocampista.csv
    └── df_portero.csv
```

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

### 2. Configurar la ruta de datos

En `utils/data_loader.py`, ajusta la variable `base_path`:

```python
base_path = r'C:/Users/TU_USUARIO/Documents/Proyecto Julio/Data/'
```

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

## 🎨 Personalizar CSS

Edita `styles.css` para cambiar colores, fuentes y estilos:

```css
:root {
	--primary-color: #0f172a; /* Color principal */
	--accent-color: #3b82f6; /* Color de acento */
	--background-color: #f6f8fa; /* Fondo */
}
```

## 📄 Crear Nuevas Páginas

1. Crea un archivo en la carpeta `pages/`
2. Nómbralo con el formato: `N_EMOJI_Nombre.py`
   - Ejemplo: `3_📊_Estadisticas.py`

```python
# pages/3_📊_Estadisticas.py
import streamlit as st

st.set_page_config(page_title="Estadísticas", page_icon="📊", layout="wide")

# Cargar CSS
from pathlib import Path
css_path = Path(__file__).parent.parent / "styles.css"
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📊 Estadísticas Globales")
# Tu código aquí...
```

## 🔧 Funciones Disponibles

### Data Loader (`utils/data_loader.py`)

```python
from utils.data_loader import load_data, get_absolute_metrics_for

# Cargar todos los datos
df_ref, matrices, mapas, df_plots = load_data()

# Obtener métricas absolutas de un jugador
df_metrics = get_absolute_metrics_for(player_id, df_procesado)
```

### Analysis (`utils/analysis.py`)

```python
from utils.analysis import obtener_jugadores_similares, strengths_and_weaknesses

# Buscar jugadores similares
similares = obtener_jugadores_similares(player_id, posicion, matrices, mapas, df_plots, k=5)

# Analizar fortalezas y debilidades
strengths, weaknesses = strengths_and_weaknesses(player_id, df_plot, top_n=5)
```

### Visualizations (`utils/visualizations.py`)

```python
from utils.visualizations import crear_grafico_radar, draw_player_card

# Dibujar tarjeta de jugador
draw_player_card(player_data, title="Mi Jugador")

# Crear gráfico de radar
fig = crear_grafico_radar([player_id], df_reporte, df_plot, "Delantero", df_ref)
st.pyplot(fig)
```

## 🎯 Mejores Prácticas

1. **Usa `@st.cache_data`** para funciones que cargan datos:

   ```python
   @st.cache_data
   def cargar_datos():
       return load_data()
   ```

2. **Organiza el código por funcionalidad** - mantén separadas:

   - Carga de datos
   - Análisis y cálculos
   - Visualización

3. **CSS global** - define estilos comunes en `styles.css`

4. **Componentes reutilizables** - crea funciones para UI repetitiva

## 🐛 Solución de Problemas

### Error: "FileNotFoundError"

→ Verifica la ruta en `utils/data_loader.py` línea `base_path`

### Error: "ModuleNotFoundError: No module named 'utils'"

→ Asegúrate de tener `utils/__init__.py` creado

### El CSS no se aplica

→ Verifica que `styles.css` esté en la raíz del proyecto

## 📝 Notas Adicionales

- **Multipágina**: Streamlit detecta automáticamente archivos en `pages/`
- **Orden**: Los números al inicio determinan el orden en el sidebar
- **Emojis**: Streamlit soporta emojis en nombres de páginas
- **Estado**: Usa `st.session_state` para compartir datos entre páginas

## 🤝 Contribuir

Para añadir nuevas funcionalidades:

1. Crea la función en el módulo correspondiente de `utils/`
2. Añádela a `__init__.py`
3. Importa donde la necesites

---

**Desarrollado con ❤️ usando Streamlit**
