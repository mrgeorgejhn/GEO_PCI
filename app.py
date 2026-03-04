import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="GEO_PCI - Análisis de Pavimentos", layout="wide", page_icon="🚜")

# --- PARÁMETROS TÉCNICOS ---
P = {
    "HUECO_AREA_EQ": 0.45,
    "HUECO_UMBRAL_LARGO": 0.75,
    "LOSA_AREA_M2": 18.0,
    "CAP_DENS_AHUELL_PCT": 31.0
}

# --- UTILIDAD DE BÚSQUEDA DE ARCHIVOS ---
def find_file(pattern):
    """Busca un archivo en la carpeta 'data' que contenga el patrón."""
    data_folder = 'data'
    if not os.path.exists(data_folder):
        data_folder = '.' # Fallback a raíz si no hay carpeta data
    
    try:
        for f in os.listdir(data_folder):
            f_clean = f.lower().replace(" ", "").replace("_", "")
            p_clean = pattern.lower().replace(" ", "").replace("_", "")
            if p_clean in f_clean and f.endswith('.csv'):
                return os.path.join(data_folder, f)
    except Exception:
        return None
    return None

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    vd_curves = {"FLEXIBLE": {}, "RIGIDO": {}}
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
                try:
                    vd_curves[p_type][key] = pd.read_csv(path)
                except:
                    continue

    path_c_flex = find_file("Flexible")
    path_c_rig = find_file("rigido")
    
    c_flex = pd.read_csv(path_c_flex) if path_c_flex else None
    c_rig = pd.read_csv(path_c_rig) if path_c_rig else None
    
    return vd_curves, c_flex, c_rig

VD_CURVES, CDV_FLEX, CDV_RIG = load_data()

# --- LÓGICA DE CÁLCULO ---
def get_dv(p_type, key, sev, density):
    if key not in VD_CURVES[p_type]: return 0.0
    df = VD_CURVES[p_type][key]
    col = {"Baja": "Baja", "Media": "Media", "Alta": "Alta"}.get(sev, "Media")
    
    if 'Densidad' not in df.columns or col not in df.columns: return 0.0
    
    x, y = df['Densidad'].values, df[col].values
    f_interp = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    return float(f_interp(density))

def get_cdv(p_type, q, tdv):
    df = CDV_FLEX if p_type == "FLEXIBLE" else CDV_RIG
    if df is None or q <= 1: return tdv
    
    qs = df['q'].unique()
    q_use = qs[np.abs(qs - q).argmin()]
    sub = df[df['q'] == q_use].sort_values('tdv')
    
    x_tdv, y_cdv = sub['tdv'].values, sub['cdv'].values
    f_interp = interp1d(x_tdv, y_cdv, bounds_error=False, fill_value=(y[0], y[-1]))
    return float(f_interp(tdv))

# --- INICIALIZACIÓN DE ESTADO ---
if 'list_danos' not in st.session_state:
    st.session_state.list_danos = []

# --- INTERFAZ ---
st.title("🚜 Sistema de Evaluación PCI")
st.caption("Herramienta avanzada para Ingeniería de Pavimentos - Cali, Colombia")

with st.sidebar:
    st.header("⚙️ Configuración Unidad")
    pav_type = st.radio("Pavimento", ["FLEXIBLE", "RIGIDO"])
    area_total = st.number_input("Área Unidad (m²)", value=250.0, min_value=1.0)
    
    st.divider()
    if st.button("🗑️ Reiniciar Todo", use_container_width=True):
        st.session_state.list_danos = []
        st.rerun()

col_in, col_out = st.columns([1, 1.5])

with col_in:
    st.subheader("📝 Nuevo Registro")
    with st.form("form_danos", clear_on_submit=True):
        opciones = list(VD_CURVES[pav_type].keys())
        tipo = st.selectbox("Deterioro", options=opciones if opciones else ["Sin datos en /data"])
        sev = st.select_slider("Severidad", options=["Baja", "Media", "Alta"], value="Media")
        cant = st.number_input("Cantidad", min_value=0.0, step=0.1)
        
        if st.form_submit_button("Añadir Deterioro", use_container_width=True):
            if opciones:
                st.session_state.list_danos.append({"Deterioro": tipo, "Severidad": sev, "Cantidad": cant})
                st.rerun()

with col_out:
    st.subheader("📊 Análisis de Resultados")
    if st.session_state.list_danos:
        dvs = []
        for d in st.session_state.list_danos:
            # Densidad según tipo
            if pav_type == "FLEXIBLE":
                dens = (d['Cantidad'] / area_total) * 100
                if d['Deterioro'] in ["ahuellamiento", "abultamientos_hundimientos"]:
                    dens = min(dens, P["CAP_DENS_AHUELL_PCT"])
            else:
                dens = (d['Cantidad'] / (area_total / P["LOSA_AREA_M2"])) * 100
            
            dvs.append(get_dv(pav_type, d['Deterioro'], d['Severidad'], dens))

        # Cálculo PCI (ASTM D6433)
        current_vals = sorted([v for v in dvs if v > 0], reverse=True)
        if current_vals:
            hdv = current_vals[0]
            m = 1 + (9/98)*(100 - hdv)
            current_vals = current_vals[:int(np.ceil(m))]
            
            vals_iter = list(current_vals)
            max_cdv = 0
            while True:
                q = sum(1 for v in vals_iter if v > 2.0)
                tdv = sum(vals_iter)
                cdv = get_cdv(pav_type, q, tdv)
                max_cdv = max(max_cdv, cdv)
                if q <= 1: break
                # Reducción iterativa
                idx_to_reduce = [i for i, v in enumerate(vals_iter) if v > 2.0]
                vals_iter[idx_to_reduce[-1]] = 2.0
            
            pci = max(0.0, 100.0 - max_cdv)
            
            # --- Métricas Principales ---
            m1, m2 = st.columns(2)
            m1.metric("PCI Resultante", f"{round(pci, 1)}")
            
            rating = "Excelente" if pci > 85 else "Muy Bueno" if pci > 70 else "Bueno" if pci > 55 else "Regular" if pci > 40 else "Malo" if pci > 25 else "Muy Malo" if pci > 10 else "Fallado"
            m2.metric("Calificación", rating)

            # --- Detalle de Deterioros ---
            st.markdown("#### Detalle de Valores Deducidos")
            df_res = pd.DataFrame(st.session_state.list_danos)
            df_res['Deducido'] = [round(v, 2) for v in dvs]
            st.table(df_res)
    else:
        st.info("Registre los daños observados para calcular la condición del tramo.")