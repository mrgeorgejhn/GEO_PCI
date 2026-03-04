import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Calculadora PCI - ASTM D6433", layout="wide")

# --- CONSTANTES Y MAPEO ---
P = {
    "HUECO_AREA_EQ": 0.45,
    "HUECO_UMBRAL_LARGO": 0.75,
    "LOSA_AREA_M2": 18.0,
    "CAP_DENS_AHUELL_PCT": 31.0
}

# Mapeo de daños (Basado en los archivos suministrados)
MAP_DANOS = {
    "FLEXIBLE": {
        "piel_cocodrilo": {"name": "Piel de Cocodrilo", "unit": "m2", "mode": "AREA"},
        "baches": {"name": "Baches (Huecos)", "unit": "Ud", "mode": "COUNT"},
        "ahuellamiento": {"name": "Ahuellamiento", "unit": "m2", "mode": "AREA", "cap": 31},
        "grietas_longitudinales_transver": {"name": "Grietas Long. y Trans.", "unit": "m", "mode": "LINEAL", "cap": 50},
        "parcheo_mal_estado": {"name": "Parcheo Mal Estado", "unit": "m2", "mode": "AREA"},
        "corrugacion": {"name": "Corrugación", "unit": "m2", "mode": "AREA"},
        "pulimiento_agregado": {"name": "Pulimiento Agregado", "unit": "m2", "mode": "AREA"},
        "agrietamiento_bloque": {"name": "Agrietamiento en Bloque", "unit": "m2", "mode": "AREA"},
        "grieta_reflexion_junto": {"name": "Grieta Reflexión Junta", "unit": "m", "mode": "LINEAL"},
        "abultamientos_hundimientos": {"name": "Abultamientos y Hundimientos", "unit": "m2", "mode": "AREA", "cap": 31},
    },
    "RIGIDO": {
        "blowup_buckling": {"name": "Soplado/Pandeo", "unit": "losa", "mode": "LOSA"},
        "grieta_lineal": {"name": "Grieta Lineal", "unit": "losa", "mode": "LOSA"},
        "sello_junta_bombeo": {"name": "Sello Junta/Bombeo", "unit": "losa", "mode": "LOSA"},
        "pulimiento_agregado_s_popouts_d": {"name": "Pulimiento/Descascaramiento", "unit": "losa", "mode": "LOSA"},
        "grieta_esquina": {"name": "Grieta de Esquina", "unit": "losa", "mode": "LOSA"},
        "escalonamiento": {"name": "Escalonamiento", "unit": "losa", "mode": "LOSA"},
        "parcheo": {"name": "Parcheo", "unit": "losa", "mode": "LOSA"},
        "retraccion": {"name": "Retracción (Grietas)", "unit": "losa", "mode": "LOSA"},
        "losa_dividida": {"name": "Losa Dividida", "unit": "losa", "mode": "LOSA"},
        "Descascaramiento": {"name": "Descascaramiento Junta/Superficie", "unit": "losa", "mode": "LOSA"},
    }
}

# --- FUNCIONES DE CARGA ---
@st.cache_data
def load_all_curves():
    # Cargar curvas VD y CDV desde archivos locales
    vd_curves = {"FLEXIBLE": {}, "RIGIDO": {}}
    
    # Flexible
    for key in MAP_DANOS["FLEXIBLE"].keys():
        fname = f"VD FLEXIBLE.xlsx - {key}.csv"
        if os.path.exists(fname):
            vd_curves["FLEXIBLE"][key] = pd.read_csv(fname)
            
    # Rigido
    for key in MAP_DANOS["RIGIDO"].keys():
        fname = f"VD RIGIDO.xlsx - {key}.csv"
        if os.path.exists(fname):
            vd_curves["RIGIDO"][key] = pd.read_csv(fname)
            
    # CDV
    cdv_flex = pd.read_csv("CORRECCION DV.xlsx - Flexible.csv")
    cdv_rig = pd.read_csv("CORRECCION DV.xlsx - rigido.csv")
    
    return vd_curves, cdv_flex, cdv_rig

VD_CURVES, CDV_FLEX, CDV_RIG = load_all_curves()

# --- LÓGICA DE CÁLCULO ---
def get_dv(pavement_type, damage_key, severity, density):
    if damage_key not in VD_CURVES[pavement_type]:
        return 0.0
    df = VD_CURVES[pavement_type][damage_key]
    col = {"Baja": "Baja", "Media": "Media", "Alta": "Alta"}.get(severity, "Media")
    
    f_interp = interp1d(df['Densidad'], df[col], bounds_error=False, fill_value="extrapolate")
    dv = float(f_interp(density))
    return max(0.0, dv)

def get_cdv(pavement_type, q, tdv):
    df = CDV_FLEX if pavement_type == "FLEXIBLE" else CDV_RIG
    # Buscar q exacto o el más cercano
    qs = df['q'].unique()
    q_use = qs[np.abs(qs - q).argmin()]
    
    sub = df[df['q'] == q_use].sort_values('tdv')
    f_interp = interp1d(sub['tdv'], sub['cdv'], bounds_error=False, fill_value="extrapolate")
    return float(f_interp(tdv))

def calculate_pci(pavement_type, damages, section_area):
    if not damages:
        return 100.0, [], []

    # 1. Calcular DV para cada daño
    deduct_values = []
    for d in damages:
        # Densidad
        if pavement_type == "FLEXIBLE":
            meta = MAP_DANOS["FLEXIBLE"].get(d['key'], {})
            density = (d['qty'] / section_area) * 100
            if "cap" in meta:
                density = min(density, meta['cap'])
        else:
            # Rígido: cantidad ingresada ya es número de losas dañadas
            total_slabs = section_area / P["LOSA_AREA_M2"]
            density = (d['qty'] / total_slabs) * 100
        
        dv = get_dv(pavement_type, d['key'], d['sev'], density)
        deduct_values.append(dv)

    # 2. Proceso iterativo para Max CDV
    dvs = sorted(deduct_values, reverse=True)
    hdv = dvs[0]
    m = 1 + (9/98) * (100 - hdv)
    
    # Solo tomar los top m valores
    vals = dvs[:int(np.ceil(m))]
    
    iterations = []
    max_cdv = 0
    current_vals = list(vals)
    
    while True:
        q = sum(1 for v in current_vals if v > 2.0)
        tdv = sum(current_vals)
        
        if q <= 1:
            cdv = tdv
        else:
            cdv = get_cdv(pavement_type, q, tdv)
        
        iterations.append({"q": q, "vals": list(current_vals), "tdv": tdv, "cdv": cdv})
        max_cdv = max(max_cdv, cdv)
        
        if q <= 1:
            break
            
        # Reemplazar el menor valor > 2 por 2.0
        indices_gt2 = [i for i, v in enumerate(current_vals) if v > 2.0]
        idx_min = indices_gt2[np.argmin([current_vals[i] for i in indices_gt2])]
        current_vals[idx_min] = 2.0

    pci = max(0.0, 100.0 - max_cdv)
    return pci, iterations, deduct_values

def get_rating(pci):
    if pci > 85: return "EXCELENTE"
    if pci > 70: return "MUY BUENO"
    if pci > 55: return "BUENO"
    if pci > 40: return "REGULAR"
    if pci > 25: return "MALO"
    if pci > 10: return "MUY MALO"
    return "FALLADO"

# --- INTERFAZ STREAMLIT ---
st.title("🧮 Calculadora PCI (Manual)")
st.markdown("Cálculo punto a punto del Índice de Condición del Pavimento.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuración de Unidad")
    pav_type = st.radio("Tipo de Pavimento", ["FLEXIBLE", "RIGIDO"])
    section_area = st.number_input("Área total de la sección (m2)", min_value=1.0, value=200.0)
    
    if pav_type == "RIGIDO":
        st.info(f"Equivale a aprox. {section_area/18:.1f} losas (18m2 c/u)")

    st.divider()
    st.subheader("Agregar Daño")
    with st.form("damage_form", clear_on_submit=True):
        d_key = st.selectbox("Tipo de Daño", options=list(MAP_DANOS[pav_type].keys()), 
                             format_func=lambda x: MAP_DANOS[pav_type][x]['name'])
        d_sev = st.selectbox("Severidad", ["Baja", "Media", "Alta"])
        
        unit = MAP_DANOS[pav_type][d_key]['unit']
        label_qty = f"Cantidad ({unit})"
        if pav_type == "RIGIDO": label_qty = "Número de losas dañadas"
            
        d_qty = st.number_input(label_qty, min_value=0.01, step=0.1)
        
        add_btn = st.form_submit_button("Añadir Daño")
        if add_btn:
            if 'damages' not in st.session_state: st.session_state.damages = []
            st.session_state.damages.append({"key": d_key, "sev": d_sev, "qty": d_qty})

with col2:
    st.header("Resultados")
    
    if 'damages' in st.session_state and st.session_state.damages:
        # Mostrar tabla de daños ingresados
        df_input = pd.DataFrame(st.session_state.damages)
        df_input['Nombre'] = df_input['key'].apply(lambda x: MAP_DANOS[pav_type][x]['name'])
        st.table(df_input[['Nombre', 'sev', 'qty']])
        
        if st.button("Limpiar Datos"):
            st.session_state.damages = []
            st.rerun()
            
        # Calcular
        pci, iters, all_dvs = calculate_pci(pav_type, st.session_state.damages, section_area)
        
        # Mostrar Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("PCI", f"{pci:.1f}")
        c2.metric("Calificación", get_rating(pci))
        c3.metric("Deducido Máx (CDV)", f"{100-pci:.1f}")
        
        # Mostrar Iteraciones
        with st.expander("Ver proceso de iteración (CDV)"):
            it_df = pd.DataFrame(iters)
            st.write(it_df[['q', 'tdv', 'cdv']])
            
    else:
        st.warning("Agregue daños en la izquierda para ver el cálculo.")