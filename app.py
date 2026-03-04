import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="GEO_PCI - Calculadora", layout="wide")

# --- UTILIDAD DE BÚSQUEDA DE ARCHIVOS (Evita errores de nombres) ---
def find_file(pattern):
    """Busca un archivo en el directorio actual que contenga el patrón, ignorando mayúsculas."""
    for f in os.listdir('.'):
        if pattern.lower().replace(" ", "") in f.lower().replace(" ", ""):
            return f
    return None

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    vd_curves = {"FLEXIBLE": {}, "RIGIDO": {}}
    
    # Mapeo de claves a patrones de búsqueda
    mapping = {
        "FLEXIBLE": {
            "piel_cocodrilo": "piel_cocodrilo",
            "baches": "baches",
            "ahuellamiento": "ahuellamiento",
            "grietas_longitudinales_transver": "longitudinales",
            "parcheo_mal_estado": "parcheo_mal",
            "corrugacion": "corrugacion",
            "pulimiento_agregado": "pulimiento_agregado",
            "agrietamiento_bloque": "bloque",
            "grieta_reflexion_junto": "reflexion",
            "abultamientos_hundimientos": "abultamientos"
        },
        "RIGIDO": {
            "blowup_buckling": "blowup",
            "grieta_lineal": "grieta_lineal",
            "sello_junta_bombeo": "sello_junta",
            "pulimiento_agregado_s_popouts_d": "popouts",
            "grieta_esquina": "esquina",
            "escalonamiento": "escalonamiento",
            "parcheo": "parcheo",
            "retraccion": "retraccion",
            "losa_dividida": "losa_dividida",
            "Descascaramiento": "Descascaramiento"
        }
    }

    for p_type, damages in mapping.items():
        for key, pattern in damages.items():
            path = find_file(pattern)
            if path:
                vd_curves[p_type][key] = pd.read_csv(path)

    # Cargar Tablas de Corrección
    path_c_flex = find_file("Flexible.csv")
    path_c_rig = find_file("rigido.csv")
    
    c_flex = pd.read_csv(path_c_flex) if path_c_flex else None
    c_rig = pd.read_csv(path_c_rig) if path_c_rig else None
    
    return vd_curves, c_flex, c_rig

VD_CURVES, CDV_FLEX, CDV_RIG = load_data()

# --- LÓGICA DE CÁLCULO ---
def get_dv(p_type, key, sev, density):
    if key not in VD_CURVES[p_type]: return 0.0
    df = VD_CURVES[p_type][key]
    col = {"Baja": "Baja", "Media": "Media", "Alta": "Alta"}.get(sev, "Media")
    # Interpolación
    x, y = df['Densidad'].values, df[col].values
    return float(interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))(density))

def get_cdv(p_type, q, tdv):
    df = CDV_FLEX if p_type == "FLEXIBLE" else CDV_RIG
    if df is None or q <= 1: return tdv
    qs = df['q'].unique()
    q_use = qs[np.abs(qs - q).argmin()]
    sub = df[df['q'] == q_use].sort_values('tdv')
    return float(interp1d(sub['tdv'], sub['cdv'], bounds_error=False, fill_value="extrapolate")(tdv))

# --- INTERFAZ ---
st.title("🚜 Sistema PCI - Análisis de Pavimentos")
st.markdown("---")

# Sidebar para datos globales
with st.sidebar:
    st.header("Datos del Tramo")
    pav_type = st.radio("Tipo de Pavimento", ["FLEXIBLE", "RIGIDO"])
    area_total = st.number_input("Área Total (m²)", value=250.0, min_value=1.0)
    if st.button("🗑️ Reiniciar Cálculo"):
        st.session_state.list_danos = []
        st.rerun()

if 'list_danos' not in st.session_state:
    st.session_state.list_danos = []

col_in, col_out = st.columns([1, 1.5])

with col_in:
    st.subheader("➕ Nuevo Deterioro")
    with st.form("form_danos", clear_on_submit=True):
        tipo = st.selectbox("Deterioro", options=list(VD_CURVES[pav_type].keys()))
        sev = st.select_slider("Severidad", options=["Baja", "Media", "Alta"])
        cant = st.number_input("Cantidad (m, m² o losas)", min_value=0.0, step=0.1)
        if st.form_submit_button("Agregar"):
            st.session_state.list_danos.append({"key": tipo, "sev": sev, "qty": cant})
            st.rerun()

with col_out:
    st.subheader("📊 Resultados")
    if st.session_state.list_danos:
        # Calcular DVs
        dvs = []
        for d in st.session_state.list_danos:
            # Cálculo de Densidad
            if pav_type == "FLEXIBLE":
                dens = (d['qty'] / area_total) * 100
                if d['key'] in ["ahuellamiento", "abultamientos_hundimientos"]: dens = min(dens, 31.0)
            else:
                # Rígido: cantidad / (Area/18)
                dens = (d['qty'] / (area_total / 18.0)) * 100
            
            val_d = get_dv(pav_type, d['key'], d['sev'], dens)
            dvs.append(val_d)

        # Proceso CDV Máximo
        vals_sorted = sorted([v for v in dvs if v > 2.0], reverse=True)
        tdv_inicial = sum(dvs)
        
        # Iteración simplificada (ASTM)
        max_cdv = 0
        current_vals = sorted(dvs, reverse=True)
        # m = 1 + (9/98)*(100 - HDV)
        m = 1 + (9/98)*(100 - current_vals[0]) if current_vals else 1
        current_vals = current_vals[:int(np.ceil(m))]

        while True:
            q = sum(1 for v in current_vals if v > 2.0)
            tdv = sum(current_vals)
            cdv = get_cdv(pav_type, q, tdv)
            max_cdv = max(max_cdv, cdv)
            if q <= 1: break
            # Bajar el menor > 2 a 2.0
            idx = [i for i, v in enumerate(current_vals) if v > 2.0]
            current_vals[idx[-1]] = 2.0
        
        pci = 100 - max_cdv
        
        # Mostrar métricas
        m1, m2 = st.columns(2)
        m1.metric("PCI FINAL", f"{round(pci, 1)}")
        
        # Color del Rating
        rating = "Excelente" if pci > 85 else "Muy Bueno" if pci > 70 else "Bueno" if pci > 55 else "Regular" if pci > 40 else "Malo"
        m2.metric("Calificación", rating)

        st.table(pd.DataFrame(st.session_state.list_danos))
    else:
        st.info("Ingresa los deterioros para ver el cálculo.")