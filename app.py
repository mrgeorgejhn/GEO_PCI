import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Calculadora PCI - GEO_PCI", layout="wide")

# Parámetros técnicos
P = {
    "HUECO_AREA_EQ": 0.45,
    "HUECO_UMBRAL_LARGO": 0.75,
    "LOSA_AREA_M2": 18.0,
    "CAP_DENS_AHUELL_PCT": 31.0
}

# Mapeo de nombres de archivos CSV (Deben coincidir con los que subiste a GitHub)
MAP_FILES = {
    "FLEXIBLE": {
        "piel_cocodrilo": "VD FLEXIBLE.xlsx - piel_cocodrilo.csv",
        "baches": "VD FLEXIBLE.xlsx - baches.csv",
        "ahuellamiento": "VD FLEXIBLE.xlsx - ahuellamiento.csv",
        "grietas_longitudinales_transver": "VD FLEXIBLE.xlsx - grietas_longitudinales_transver.csv",
        "parcheo_mal_estado": "VD FLEXIBLE.xlsx - parcheo_mal_estado.csv",
        "corrugacion": "VD FLEXIBLE.xlsx - corrugacion.csv",
        "pulimiento_agregado": "VD FLEXIBLE.xlsx - pulimiento_agregado.csv",
        "agrietamiento_bloque": "VD FLEXIBLE.xlsx - agrietamiento_bloque.csv",
        "grieta_reflexion_junto": "VD FLEXIBLE.xlsx - grieta_reflexion_junto.csv",
        "abultamientos_hundimientos": "VD FLEXIBLE.xlsx - abultamientos_hundimientos.csv",
    },
    "RIGIDO": {
        "blowup_buckling": "VD RIGIDO.xlsx - blowup_buckling.csv",
        "grieta_lineal": "VD RIGIDO.xlsx - grieta_lineal.csv",
        "sello_junta_bombeo": "VD RIGIDO.xlsx - sello_junta_bombeo.csv",
        "pulimiento_agregado_s_popouts_d": "VD RIGIDO.xlsx - pulimiento_agregado_s_popouts_d.csv",
        "grieta_esquina": "VD RIGIDO.xlsx - grieta_esquina.csv",
        "escalonamiento": "VD RIGIDO.xlsx - escalonamiento.csv",
        "parcheo": "VD RIGIDO.xlsx - parcheo.csv",
        "retraccion": "VD RIGIDO.xlsx - retraccion.csv",
        "losa_dividida": "VD RIGIDO.xlsx - losa_dividida.csv",
        "Descascaramiento": "VD RIGIDO.xlsx - Descascaramiento.csv",
    }
}

# --- FUNCIONES DE CARGA ---
@st.cache_data
def load_all_curves():
    vd_curves = {"FLEXIBLE": {}, "RIGIDO": {}}
    
    # Intentar cargar cada archivo
    for p_type in ["FLEXIBLE", "RIGIDO"]:
        for key, filename in MAP_FILES[p_type].items():
            if os.path.exists(filename):
                vd_curves[p_type][key] = pd.read_csv(filename)
            else:
                st.warning(f"Archivo no encontrado: {filename}")
                
    # Cargar Corrección
    c_flex = pd.read_csv("CORRECCION DV.xlsx - Flexible.csv") if os.path.exists("CORRECCION DV.xlsx - Flexible.csv") else None
    c_rig = pd.read_csv("CORRECCION DV.xlsx - rigido.csv") if os.path.exists("CORRECCION DV.xlsx - rigido.csv") else None
    
    return vd_curves, c_flex, c_rig

VD_CURVES, CDV_FLEX, CDV_RIG = load_all_curves()

# --- LÓGICA DE CÁLCULO ---
def get_dv(pavement_type, damage_key, severity, density):
    if damage_key not in VD_CURVES[pavement_type]: return 0.0
    df = VD_CURVES[pavement_type][damage_key]
    col = {"Baja": "Baja", "Media": "Media", "Alta": "Alta"}.get(severity, "Media")
    
    # Asegurar que la densidad no exceda los límites de la tabla
    dens_idx = df['Densidad'].values
    dv_vals = df[col].values
    f_interp = interp1d(dens_idx, dv_vals, bounds_error=False, fill_value=(dv_vals[0], dv_vals[-1]))
    return float(f_interp(density))

def get_cdv(pavement_type, q, tdv):
    df = CDV_FLEX if pavement_type == "FLEXIBLE" else CDV_RIG
    if df is None: return tdv
    
    qs = df['q'].unique()
    q_use = qs[np.abs(qs - q).argmin()]
    sub = df[df['q'] == q_use].sort_values('tdv')
    
    f_interp = interp1d(sub['tdv'], sub['cdv'], bounds_error=False, fill_value="extrapolate")
    return float(f_interp(tdv))

# --- INTERFAZ ---
st.title("📊 GEO_PCI: Calculadora de Pavimentos")
st.sidebar.header("Configuración de Unidad")
pav_type = st.sidebar.radio("Tipo de Pavimento", ["FLEXIBLE", "RIGIDO"])
section_area = st.sidebar.number_input("Área total (m²)", min_value=1.0, value=250.0)

if 'damages' not in st.session_state:
    st.session_state.damages = []

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 Ingreso de Datos")
    with st.form("add_damage", clear_on_submit=True):
        d_key = st.selectbox("Tipo de Daño", options=list(MAP_FILES[pav_type].keys()))
        d_sev = st.selectbox("Severidad", ["Baja", "Media", "Alta"])
        d_qty = st.number_input("Cantidad (m2, m o Ud)", min_value=0.0, step=0.1)
        if st.form_submit_button("Añadir Daño"):
            st.session_state.damages.append({"key": d_key, "sev": d_sev, "qty": d_qty})
            st.rerun()

    if st.button("🗑️ Limpiar Todo"):
        st.session_state.damages = []
        st.rerun()

with col2:
    st.subheader("📉 Resultados del Tramo")
    if st.session_state.damages:
        # Procesar
        deduct_values = []
        for d in st.session_state.damages:
            if pav_type == "FLEXIBLE":
                dens = (d['qty'] / section_area) * 100
                # Aplicar caps
                if d['key'] in ["abultamientos_hundimientos", "ahuellamiento"]: dens = min(dens, 31.0)
                if d['key'] == "grietas_longitudinales_transver": dens = min(dens, 50.0)
            else:
                dens = (d['qty'] / (section_area / 18.0)) * 100
            
            dv = get_dv(pav_type, d['key'], d['sev'], dens)
            deduct_values.append(dv)

        # Cálculo PCI
        dvs = sorted([v for v in deduct_values if v > 0], reverse=True)
        if dvs:
            hdv = dvs[0]
            m = 1 + (9/98)*(100-hdv)
            vals = dvs[:int(np.ceil(m))]
            
            # Iteración CDV
            current = list(vals)
            max_cdv = 0
            while True:
                q = sum(1 for v in current if v > 2.0)
                tdv = sum(current)
                cdv = get_cdv(pav_type, q, tdv) if q > 1 else tdv
                max_cdv = max(max_cdv, cdv)
                if q <= 1: break
                # Reducir el menor > 2 a 2
                idx_min = [i for i, v in enumerate(current) if v > 2.0]
                current[idx_min[np.argmin([current[i] for i in idx_min])]] = 2.0
            
            pci = max(0.0, 100.0 - max_cdv)
            
            # Mostrar
            st.metric("Índice PCI", f"{pci:.1f}")
            
            # Tabla de resumen
            res_df = pd.DataFrame(st.session_state.damages)
            res_df['Deducido'] = deduct_values
            st.dataframe(res_df)
    else:
        st.info("Esperando ingreso de datos...")