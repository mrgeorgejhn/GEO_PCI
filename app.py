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
        # Si no existe la carpeta, intenta en el directorio raíz (fallback)
        data_folder = '.'
    
    for f in os.listdir(data_folder):
        # Limpieza de nombres para comparación
        f_clean = f.lower().replace(" ", "").replace("_", "")
        p_clean = pattern.lower().replace(" ", "").replace("_", "")
        if p_clean in f_clean and f.endswith('.csv'):
            return os.path.join(data_folder, f)
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
                vd_curves[p_type][key] = pd.read_csv(path)

    # Cargar Tablas de Corrección
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
    
    # Asegurar que las columnas existan
    if 'Densidad' not in df.columns or col not in df.columns: return 0.0
    
    x, y = df['Densidad'].values, df[col].values
    # Interpolación lineal con límites seguros
    f_interp = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    return float(f_interp(density))

def get_cdv(p_type, q, tdv):
    df = CDV_FLEX if p_type == "FLEXIBLE" else CDV_RIG
    if df is None or q <= 1: return tdv
    
    # Filtrar por q y asegurar que tdv esté en rango
    qs = df['q'].unique()
    q_use = qs[np.abs(qs - q).argmin()]
    sub = df[df['q'] == q_use].sort_values('tdv')
    
    x_tdv, y_cdv = sub['tdv'].values, sub['cdv'].values
    f_interp = interp1d(x_tdv, y_cdv, bounds_error=False, fill_value="extrapolate")
    return float(f_interp(tdv))

# --- INTERFAZ DE USUARIO ---
st.title("🚜 Sistema de Evaluación PCI")
st.subheader("Ingeniería de Pavimentos - GEO_PCI")

with st.sidebar:
    st.header("⚙️ Configuración")
    pav_type = st.radio("Tipo de Pavimento", ["FLEXIBLE", "RIGIDO"])
    area_total = st.number_input("Área de la Unidad (m²)", value=250.0, min_value=1.0)
    
    st.divider()
    if st.button("🗑️ Reiniciar Todo"):
        st.session_state.list_danos = []
        st.rerun()

if 'list_danos' not in st.session_state:
    st.session_state.list_danos = []

col_in, col_out = st.columns([1, 1.5])

with col_in:
    st.markdown("### 📝 Registrar Deterioro")
    with st.form("form_danos", clear_on_submit=True):
        # Solo mostrar deterioros cargados correctamente
        opciones = list(VD_CURVES[pav_type].keys())
        tipo = st.selectbox("Tipo de Deterioro", options=opciones if opciones else ["No se encontraron datos"])
        sev = st.select_slider("Severidad", options=["Baja", "Media", "Alta"], value="Media")
        cant = st.number_input("Cantidad (m, m²)", min_value=0.0, step=0.1)
        
        submitted = st.form_submit_button("Añadir a la lista")
        if submitted and tipo != "No se encontraron datos":
            st.session_state.list_danos.append({"Deterioro": tipo, "Severidad": sev, "Cantidad": cant})
            st.rerun()

with col_out:
    st.markdown("### 📊 Análisis de Condición")
    if st.session_state.list_danos:
        # Calcular DVs Individuales
        dvs = []
        for d in st.session_state.list_danos:
            if pav_type == "FLEXIBLE":
                # Regla de Densidad Flexible
                dens = (d['Cantidad'] / area_total) * 100
                # Aplicar Caps específicos del manual
                if d['Deterioro'] in ["ahuellamiento", "abultamientos_hundimientos"]:
                    dens = min(dens, P["CAP_DENS_AHUELL_PCT"])
                elif d['Deterioro'] == "grietas_longitudinales_transver":
                    dens = min(dens, 50.0)
            else:
                # Regla de Densidad Rígido (Basado en Losas de 18m2)
                dens = (d['Cantidad'] / (area_total / P["LOSA_AREA_M2"])) * 100
            
            val_d = get_dv(pav_type, d['Deterioro'], d['Severidad'], dens)
            dvs.append(val_d)

        # Proceso Iterativo CDV (ASTM D6433)
        current_vals = sorted([v for v in dvs if v > 0], reverse=True)
        
        if current_vals:
            # Cálculo de m (Número máximo admisible de deducibles)
            hdv = current_vals[0]
            m = 1 + (9/98)*(100 - hdv)
            # Solo trabajamos con los valores significativos según m
            current_vals = current_vals[:int(np.ceil(m))]
            
            # Guardamos copia para iterar
            vals_iter = list(current_vals)
            max_cdv = 0
            
            while True:
                q = sum(1 for v in vals_iter if v > 2.0)
                tdv = sum(vals_iter)
                cdv = get_cdv(pav_type, q, tdv)
                max_cdv = max(max_cdv, cdv)
                
                if q <= 1: break
                
                # Reducir el menor valor mayor a 2.0 a exactamente 2.0
                idx_to_reduce = [i for i, v in enumerate(vals_iter) if v > 2.0]
                vals_iter[idx_to_reduce[-1]] = 2.0
            
            pci = max(0.0, 100.0 - max_cdv)
            
            # --- MOSTRAR RESULTADOS ---
            m1, m2, m3 = st.columns(3)
            m1.metric("PCI", f"{round(pci, 1)}")
            
            # Clasificación ASTM
            if pci > 85: rating, color = "EXCELENTE", "#00b050"
            elif pci > 70: rating, color = "MUY BUENO", "#92d050"
            elif pci > 55: rating, color = "BUENO", "#ffff00"
            elif pci > 40: rating, color = "REGULAR", "#ffc000"
            elif pci > 25: rating, color = "MALO", "#ff0000"
            elif pci > 10: rating, color = "MUY MALO", "#c00000"
            else: rating, color = "FALLADO", "#333333"
            
            m2.markdown(f"**Calificación:** <span style='color:{color}; font-weight:bold;'>{rating}</span>", unsafe_allow_html=True)
            m3.metric("Deducido Máx", f"{round(max_cdv, 1)}")

            # Tabla de detalles
            df_res = pd.DataFrame(st.session_state.list_danos)
            df_res['Valor Deducido'] = [round(v, 2) for v in dvs]
            st.table(df_res)
    else:
        st.info("💡 Agregue deterioros en el panel de la izquierda para iniciar el cálculo.")