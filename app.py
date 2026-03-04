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

# --- INICIALIZACIÓN DEL ESTADO (Crucial para evitar errores al añadir) ---
if 'list_danos' not in st.session_state:
    st.session_state.list_danos = []

# --- UTILIDAD DE BÚSQUEDA DE ARCHIVOS ---
def find_file(pattern):
    data_folder = 'data'
    # Si no existe la carpeta 'data', buscar en raíz
    search_path = data_folder if os.path.exists(data_folder) else '.'
    
    try:
        for f in os.listdir(search_path):
            f_clean = f.lower().replace(" ", "").replace("_", "")
            p_clean = pattern.lower().replace(" ", "").replace("_", "")
            if p_clean in f_clean and f.endswith('.csv'):
                return os.path.join(search_path, f)
    except:
        return None
    return None

# --- CARGA DE DATOS ---
@st.cache_data
def load_all_data():
    vd_curves = {"FLEXIBLE": {}, "RIGIDO": {}}
    mapping = {
        "FLEXIBLE": {
            "piel_cocodrilo": "piel_cocodrilo", "baches": "baches", 
            "ahuellamiento": "ahuellamiento", "grietas_longitudinales_transver": "longitudinales",
            "parcheo_mal_estado": "parcheo_mal", "corrugacion": "corrugacion",
            "pulimiento_agregado": "pulimiento_agregado", "agrietamiento_bloque": "bloque",
            "grieta_reflexion_junto": "reflexion", "abultamientos_hundimientos": "abultamientos"
        },
        "RIGIDO": {
            "blowup_buckling": "blowup", "grieta_lineal": "grieta_lineal",
            "sello_junta_bombeo": "sello_junta", "pulimiento_agregado_s_popouts_d": "popouts",
            "grieta_esquina": "esquina", "escalonamiento": "escalonamiento",
            "parcheo": "parcheo", "retraccion": "retraccion",
            "losa_dividida": "losa_dividida", "Descascaramiento": "Descascaramiento"
        }
    }
    for p_type, damages in mapping.items():
        for key, pattern in damages.items():
            path = find_file(pattern)
            if path: vd_curves[p_type][key] = pd.read_csv(path)

    c_flex = pd.read_csv(find_file("Flexible")) if find_file("Flexible") else None
    c_rig = pd.read_csv(find_file("rigido")) if find_file("rigido") else None
    return vd_curves, c_flex, c_rig

VD_CURVES, CDV_FLEX, CDV_RIG = load_all_data()

# --- FUNCIONES DE CÁLCULO ---
def get_dv(p_type, key, sev, density):
    if key not in VD_CURVES[p_type]: return 0.0
    df = VD_CURVES[p_type][key]
    col = {"Baja": "Baja", "Media": "Media", "Alta": "Alta"}.get(sev, "Media")
    if col not in df.columns: return 0.0
    x, y = df['Densidad'].values, df[col].values
    f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    return float(f(density))

def get_cdv(p_type, q, tdv):
    df = CDV_FLEX if p_type == "FLEXIBLE" else CDV_RIG
    if df is None or q <= 1: return tdv
    qs = df['q'].unique()
    q_use = qs[np.abs(qs - q).argmin()]
    sub = df[df['q'] == q_use].sort_values('tdv')
    return float(interp1d(sub['tdv'], sub['cdv'], bounds_error=False, fill_value=(sub['cdv'].iloc[0], sub['cdv'].iloc[-1]))(tdv))

# --- INTERFAZ ---
st.title("🚜 GEO_PCI - Sistema de Evaluación")

with st.sidebar:
    st.header("⚙️ Configuración")
    pav_type = st.radio("Tipo de Pavimento", ["FLEXIBLE", "RIGIDO"], key="pav_type")
    area_total = st.number_input("Área de la Unidad (m²)", value=250.0, min_value=1.0)
    if st.button("🗑️ Reiniciar Todo"):
        st.session_state.list_danos = []
        st.rerun()

col_in, col_out = st.columns([1, 1.5])

with col_in:
    st.subheader("📝 Nuevo Deterioro")
    # Usamos un formulario para evitar recargas parciales
    with st.form("form_danos", clear_on_submit=True):
        opciones = list(VD_CURVES[pav_type].keys())
        tipo = st.selectbox("Tipo de Deterioro", options=opciones if opciones else ["Error: Datos no encontrados"])
        sev = st.select_slider("Severidad", options=["Baja", "Media", "Alta"], value="Media")
        cant = st.number_input("Cantidad", min_value=0.0, step=0.1)
        
        if st.form_submit_button("Añadir a la lista"):
            if opciones:
                # Guardamos el daño en la sesión
                st.session_state.list_danos.append({
                    "Deterioro": tipo, 
                    "Severidad": sev, 
                    "Cantidad": cant,
                    "Pav": pav_type # Guardamos el tipo para validar consistencia
                })
                st.rerun()

with col_out:
    st.subheader("📊 Resultados")
    if st.session_state.list_danos:
        # 1. Calcular DVs Individuales
        dvs_finales = []
        for d in st.session_state.list_danos:
            # Validamos que el daño pertenezca al tipo de pavimento actual
            if d["Pav"] != pav_type: continue
            
            if pav_type == "FLEXIBLE":
                dens = (d['Cantidad'] / area_total) * 100
                if d['Deterioro'] in ["ahuellamiento", "abultamientos_hundimientos"]:
                    dens = min(dens, P["CAP_DENS_AHUELL_PCT"])
            else:
                dens = (d['Cantidad'] / (area_total / P["LOSA_AREA_M2"])) * 100
            
            val_dv = get_dv(pav_type, d['Deterioro'], d['Severidad'], dens)
            dvs_finales.append(val_dv)

        # 2. Lógica ASTM D6433
        if dvs_finales:
            current_vals = sorted([v for v in dvs_finales if v > 0], reverse=True)
            hdv = current_vals[0]
            m = 1 + (9/98)*(100 - hdv)
            
            # Proceso iterativo para CDV
            vals_iter = current_vals[:int(np.ceil(m))]
            max_cdv = 0
            while True:
                q = sum(1 for v in vals_iter if v > 2.0)
                tdv = sum(vals_iter)
                cdv = get_cdv(pav_type, q, tdv)
                max_cdv = max(max_cdv, cdv)
                if q <= 1: break
                idx_gt2 = [i for i, v in enumerate(vals_iter) if v > 2.0]
                vals_iter[idx_gt2[-1]] = 2.0
            
            pci = max(0.0, 100.0 - max_cdv)
            
            # 3. Mostrar Métricas
            c1, c2 = st.columns(2)
            c1.metric("PCI", f"{round(pci, 1)}")
            
            # Clasificación
            rating = "Excelente" if pci > 85 else "Muy Bueno" if pci > 70 else "Bueno" if pci > 55 else "Regular" if pci > 40 else "Malo" if pci > 25 else "Muy Malo" if pci > 10 else "Fallado"
            c2.metric("Condición", rating)

            # 4. Tabla de detalles
            st.markdown("---")
            df_mostrar = pd.DataFrame([d for d in st.session_state.list_danos if d["Pav"] == pav_type])
            df_mostrar['Deducido'] = [round(v, 2) for v in dvs_finales]
            st.table(df_mostrar[['Deterioro', 'Severidad', 'Cantidad', 'Deducido']])
    else:
        st.info("Agregue deterioros para ver el cálculo.")