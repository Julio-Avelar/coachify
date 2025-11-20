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
│   ├── 1_Coachify_Match.py    # Dashboard de análisis
│   ├── 2_Face_to_face.py      # Dashboard de Face To Face
│
├── utils.py                   # Archivo de utilidades
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

### 2. Configurar la ruta de datos

En `utils.py`, ajusta la variable `base_path`:

```python
base_path = r'C:/Users/TU_USUARIO/Documents/coachify/Data/'
```

### 3. Ejecutar la aplicación

```bash
streamlit run Inicio.py
```

## 🤝 Contribuir

Para añadir nuevas funcionalidades:

1. Crea la función en el módulo correspondiente de `utils/`
2. Importa donde la necesites

---

**Desarrollado con ❤️ usando Streamlit por Julio Avelar**
