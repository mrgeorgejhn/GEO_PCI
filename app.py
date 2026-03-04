import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="GEO_PCI - Análisis de Pavimentos",
    layout="wide",
    page_icon="🚜"
)

P = {
    "HUECO_AREA_EQ": 0.45,
    "HUECO_UMBRAL_LARGO": 0.75,
    "LOSA_AREA_M2": 18.0,
    "CAP_DENS_AHUELL_PCT": 31.0
}

DATA_DIR = "data"

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "list_danos" not in st.session_state:
    st.session_state.list_danos = []

# ------------------------------------------------------------
# HELPERS: FILE SEARCH + DATA CLEANING
# ------------------------------------------------------------
def _clean_name(x: str) -> str:
    return (
        str(x).lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

def list_csv_files() -> list[str]:
    search_path = DATA_DIR if os.path.exists(DATA_DIR) else "."
    try:
        return [os.path.join(search_path, f) for f in os.listdir(search_path) if f.lower().endswith(".csv")]
    except Exception:
        return []

def find_file_best(pattern: str) -> str | None:
    """
    Devuelve el mejor match para `pattern` dentro de la carpeta data (o .).
    Prioriza:
      1) match exacto por nombre (sin extensión)
      2) match por 'contains'
    """
    pat = _clean_name(pattern)
    candidates = list_csv_files()
    if not candidates:
        return None

    scored = []
    for path in candidates:
        fname = os.path.basename(path)
        base = os.path.splitext(fname)[0]
        base_c = _clean_name(base)
        fname_c = _clean_name(fname)

        score = -1
        if base_c == pat or fname_c == pat:
            score = 100
        elif pat in base_c:
            score = 50
        elif pat in fname_c:
            score = 40

        if score >= 0:
            # desempate: archivos más “específicos” tienden a ser más largos
            score += min(len(base_c), 30) / 100.0
            scored.append((score, path))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Lee CSV intentando inferir separador si hace falta.
    """
    # Primero intento estándar
    try:
        df = pd.read_csv(path)
        # Si queda 1 sola columna y hay ';' en el header, es separador equivocado
        if df.shape[1] == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        # fallback: inferencia de separador
        return pd.read_csv(path, sep=None, engine="python")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
    )
    return df

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_all_data():
    # VD curves (DV vs densidad por severidad)
    vd_curves = {"FLEXIBLE": {}, "RIGIDO": {}}

    # Map de "clave interna" -> "suffix de archivo"
    # OJO: aquí los sufijos son los de tus CSV (sin 'vd_flexible_' / 'vd_rigido_')
    mapping = {
        "FLEXIBLE": {
            "piel_cocodrilo": "piel_cocodrilo",
            "baches": "baches",
            "ahuellamiento": "ahuellamiento",
            "grietas_longitudinales_transver": "grietas_longitudinales_transver",
            "parcheo_mal_estado": "parcheo_mal_estado",
            "corrugacion": "corrugacion",
            "pulimiento_agregado": "pulimiento_agregado",
            "agrietamiento_bloque": "agrietamiento_bloque",
            "grieta_reflexion_junto": "grieta_reflexion_junto",
            "abultamientos_hundimientos": "abultamientos_hundimientos",
        },
        "RIGIDO": {
            "blowup_buckling": "blowup_buckling",
            "grieta_lineal": "grieta_lineal",
            "sello_junta_bombeo": "sello_junta_bombeo",
            "pulimiento_agregado_s_popouts_d": "pulimiento_agregado_s_popouts_d",
            "grieta_esquina": "grieta_esquina",
            "escalonamiento": "escalonamiento",
            "parcheo": "parcheo",
            "retraccion": "retraccion",
            "losa_dividida": "losa_dividida",
            "descascaramiento": "descascaramiento",
        },
    }

    # Cargar VD con patrón estricto: vd_{tipo}_{suffix}.csv
    for p_type, damages in mapping.items():
        prefix = "vd_flexible_" if p_type == "FLEXIBLE" else "vd_rigido_"
        for key, suffix in damages.items():
            path = find_file_best(prefix + suffix)
            if path:
                df = read_csv_robust(path)
                # VD: columnas esperadas: Densidad,Baja,Media,Alta
                # normalizamos para tolerar variaciones
                df = standardize_columns(df)
                # renombrar si vinieran como "densidad" etc.
                # (en tu ejemplo ya son correctas)
                vd_curves[p_type][key] = df

    # Cargar CDV (corrección): archivos específicos
    # IMPORTANTÍSIMO: NO usar "Flexible" / "rigido" porque colisiona con vd_*.csv
    cflex_path = find_file_best("correccion_flexible")
    crig_path  = find_file_best("correccion_rigido")

    c_flex = read_csv_robust(cflex_path) if cflex_path else None
    c_rig  = read_csv_robust(crig_path)  if crig_path else None

    # Normalizar CDV
    if c_flex is not None:
        c_flex = standardize_columns(c_flex)
        c_flex = coerce_numeric(c_flex, ["q", "tdv", "cdv"])

    if c_rig is not None:
        c_rig = standardize_columns(c_rig)
        c_rig = coerce_numeric(c_rig, ["q", "tdv", "cdv"])

    return vd_curves, c_flex, c_rig

VD_CURVES, CDV_FLEX, CDV_RIG = load_all_data()

# ------------------------------------------------------------
# CORE CALCS
# ------------------------------------------------------------
def get_dv(p_type: str, key: str, sev: str, density: float) -> float:
    if key not in VD_CURVES.get(p_type, {}):
        return 0.0

    df = VD_CURVES[p_type][key].copy()
    # Normalizar columnas mínimas
    df = standardize_columns(df)

    # Columnas esperadas (en minúscula)
    # densidad, baja, media, alta
    if "densidad" not in df.columns:
        return 0.0

    col = {"Baja": "baja", "Media": "media", "Alta": "alta"}.get(sev, "media")
    if col not in df.columns:
        return 0.0

    x = pd.to_numeric(df["densidad"], errors="coerce").values
    y = pd.to_numeric(df[col], errors="coerce").values

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return 0.0

    f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    return float(f(density))

def get_cdv(p_type: str, q: int, tdv: float) -> float:
    df = CDV_FLEX if p_type == "FLEXIBLE" else CDV_RIG

    if df is None or q <= 1:
        return float(tdv)

    required = {"q", "tdv", "cdv"}
    if not required.issubset(df.columns):
        st.error(
            f"Tabla CDV ({p_type}) sin columnas requeridas {required}. "
            f"Columnas actuales: {list(df.columns)}"
        )
        return float(tdv)

    dff = df.dropna(subset=["q", "tdv", "cdv"]).copy()
    if dff.empty:
        return float(tdv)

    qs = dff["q"].unique()
    if len(qs) == 0:
        return float(tdv)

    q_use = qs[np.abs(qs - q).argmin()]
    sub = dff[dff["q"] == q_use].sort_values("tdv")

    if len(sub) < 2:
        return float(tdv)

    f = interp1d(
        sub["tdv"].values,
        sub["cdv"].values,
        bounds_error=False,
        fill_value=(sub["cdv"].iloc[0], sub["cdv"].iloc[-1]),
    )
    return float(f(tdv))

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🚜 GEO_PCI - Sistema de Evaluación")

with st.sidebar:
    st.header("⚙️ Configuración")
    pav_type = st.radio("Tipo de Pavimento", ["FLEXIBLE", "RIGIDO"], key="pav_type")
    area_total = st.number_input("Área de la Unidad (m²)", value=250.0, min_value=1.0)

    if st.button("🗑️ Reiniciar Todo"):
        st.session_state.list_danos = []
        st.rerun()

    # Debug opcional
    if st.checkbox("Debug (CDV/VD)"):
        st.write("Archivos CSV detectados:")
        st.write([os.path.basename(p) for p in list_csv_files()])

        df_dbg = CDV_FLEX if pav_type == "FLEXIBLE" else CDV_RIG
        st.write("CDV columns:", None if df_dbg is None else list(df_dbg.columns))
        if df_dbg is not None:
            st.dataframe(df_dbg.head(5))

        st.write("VD keys:", list(VD_CURVES[pav_type].keys()))

col_in, col_out = st.columns([1, 1.5])

with col_in:
    st.subheader("📝 Nuevo Deterioro")
    with st.form("form_danos", clear_on_submit=True):
        opciones = list(VD_CURVES[pav_type].keys())
        tipo = st.selectbox(
            "Tipo de Deterioro",
            options=opciones if opciones else ["Error: Datos no encontrados"]
        )
        sev = st.select_slider("Severidad", options=["Baja", "Media", "Alta"], value="Media")
        cant = st.number_input("Cantidad", min_value=0.0, step=0.1)

        if st.form_submit_button("Añadir a la lista"):
            if opciones:
                st.session_state.list_danos.append({
                    "Deterioro": tipo,
                    "Severidad": sev,
                    "Cantidad": cant,
                    "Pav": pav_type
                })
                st.rerun()

with col_out:
    st.subheader("📊 Resultados")

    if st.session_state.list_danos:
        dvs_finales = []
        danos_filtrados = [d for d in st.session_state.list_danos if d["Pav"] == pav_type]

        for d in danos_filtrados:
            if pav_type == "FLEXIBLE":
                dens = (d["Cantidad"] / area_total) * 100.0
                if d["Deterioro"] in ["ahuellamiento", "abultamientos_hundimientos"]:
                    dens = min(dens, P["CAP_DENS_AHUELL_PCT"])
            else:
                # densidad en rígido por # losas equivalentes
                dens = (d["Cantidad"] / (area_total / P["LOSA_AREA_M2"])) * 100.0

            val_dv = get_dv(pav_type, d["Deterioro"], d["Severidad"], dens)
            dvs_finales.append(val_dv)

        if dvs_finales:
            current_vals = sorted([v for v in dvs_finales if v > 0], reverse=True)
            if not current_vals:
                st.warning("Todos los DV resultaron 0. Verifica archivos VD y rangos.")
            else:
                hdv = current_vals[0]
                m = 1 + (9 / 98) * (100 - hdv)

                vals_iter = current_vals[: int(np.ceil(m))]
                max_cdv = 0.0

                while True:
                    q = sum(1 for v in vals_iter if v > 2.0)
                    tdv = float(np.sum(vals_iter))
                    cdv = get_cdv(pav_type, q, tdv)
                    max_cdv = max(max_cdv, cdv)

                    if q <= 1:
                        break

                    idx_gt2 = [i for i, v in enumerate(vals_iter) if v > 2.0]
                    if not idx_gt2:
                        break
                    vals_iter[idx_gt2[-1]] = 2.0

                pci = max(0.0, 100.0 - max_cdv)

                c1, c2 = st.columns(2)
                c1.metric("PCI", f"{round(pci, 1)}")

                rating = (
                    "Excelente" if pci > 85 else
                    "Muy Bueno" if pci > 70 else
                    "Bueno" if pci > 55 else
                    "Regular" if pci > 40 else
                    "Malo" if pci > 25 else
                    "Muy Malo" if pci > 10 else
                    "Fallado"
                )
                c2.metric("Condición", rating)

                st.markdown("---")
                df_mostrar = pd.DataFrame(danos_filtrados)
                df_mostrar["Deducido"] = [round(v, 2) for v in dvs_finales]
                st.table(df_mostrar[["Deterioro", "Severidad", "Cantidad", "Deducido"]])
    else:
        st.info("Agregue deterioros para ver el cálculo.")