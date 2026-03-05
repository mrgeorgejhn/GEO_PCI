import os
import math
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
    "CAP_DENS_AHUELL_PCT": 31.0,
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
    pat = _clean_name(pattern)
    candidates = list_csv_files()
    if not candidates:
        return None

    scored: list[tuple[float, str]] = []
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
            score += min(len(base_c), 30) / 100.0
            scored.append((float(score), path))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
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
    vd_curves = {"FLEXIBLE": {}, "RIGIDO": {}}

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

    # VD: vd_{tipo}_{suffix}.csv
    for p_type, damages in mapping.items():
        prefix = "vd_flexible_" if p_type == "FLEXIBLE" else "vd_rigido_"
        for key, suffix in damages.items():
            path = find_file_best(prefix + suffix)
            if path:
                df = read_csv_robust(path)
                df = standardize_columns(df)  # densidad/baja/media/alta
                vd_curves[p_type][key] = df

    # CDV (corrección)
    cflex_path = find_file_best("correccion_flexible")
    crig_path  = find_file_best("correccion_rigido")

    c_flex = read_csv_robust(cflex_path) if cflex_path else None
    c_rig  = read_csv_robust(crig_path)  if crig_path else None

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

    df = standardize_columns(VD_CURVES[p_type][key])

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

    # Regla ASTM/MATLAB: si q <= 1 => CDV = TDV
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

def baches_qty_equiv(largo_m: float, area_m2: float) -> float:
    """
    Regla MATLAB:
      - si largo > 0.75 m => qty = area/0.45
      - si no => qty = 1
    Si no hay datos (largo=0 y area=0), qty=0.
    """
    largo_m = float(largo_m or 0.0)
    area_m2 = float(area_m2 or 0.0)

    if largo_m <= 0.0 and area_m2 <= 0.0:
        return 0.0

    if largo_m > P["HUECO_UMBRAL_LARGO"]:
        return max(0.0, area_m2) / P["HUECO_AREA_EQ"]

    return 1.0

def losas_totales(area_total_m2: float) -> int:
    # MATLAB: losasTot = ceil(areaU / 18)
    return int(math.ceil(float(area_total_m2) / float(P["LOSA_AREA_M2"])))

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🚜 GEO_PCI - Sistema de Evaluación (Manual)")

with st.sidebar:
    st.header("⚙️ Configuración")
    pav_type = st.radio("Tipo de Pavimento", ["FLEXIBLE", "RIGIDO"], key="pav_type")
    area_total = st.number_input("Área de la Unidad (m²)", value=250.0, min_value=1.0)

    if st.button("🗑️ Reiniciar Todo"):
        st.session_state.list_danos = []
        st.rerun()

    if st.checkbox("Debug (CDV/VD)"):
        st.write("Archivos CSV detectados:")
        st.write([os.path.basename(p) for p in list_csv_files()])

        df_dbg = CDV_FLEX if pav_type == "FLEXIBLE" else CDV_RIG
        st.write("CDV columns:", None if df_dbg is None else list(df_dbg.columns))
        if df_dbg is not None:
            st.dataframe(df_dbg.head(5))

        st.write("VD keys:", list(VD_CURVES[pav_type].keys()))

col_in, col_out = st.columns([1, 1.5])

# ------------------------------------------------------------
# INPUT PANEL
# ------------------------------------------------------------
with col_in:
    st.subheader("📝 Nuevo Deterioro (evento manual)")

    opciones = list(VD_CURVES[pav_type].keys())
    tipo = st.selectbox(
        "Tipo de Deterioro",
        options=opciones if opciones else ["Error: Datos no encontrados"],
        key="tipo_deterioro_sel",
    )
    sev = st.select_slider(
        "Severidad",
        options=["Baja", "Media", "Alta"],
        value="Media",
        key="sev_sel",
    )

    with st.form("form_danos", clear_on_submit=True):
        cant = 0.0
        largo_bache = 0.0
        area_bache_m2 = 0.0
        area_dano_m2_rig = 0.0

        if pav_type == "FLEXIBLE" and str(tipo).strip().lower() == "baches":
            st.caption("Baches: si largo > 0.75 m ⇒ huecos_equiv = area/0.45; si no ⇒ 1 hueco.")
            largo_bache = st.number_input(
                "Largo del bache (m)",
                min_value=0.0,
                step=0.01,
                value=0.0,
                key="largo_bache_in",
            )
            area_bache_m2 = st.number_input(
                "Área del bache (m²)",
                min_value=0.0,
                step=0.01,
                value=0.0,
                key="area_bache_in",
            )

        elif pav_type == "RIGIDO":
            st.caption("Rígido (MATLAB): se ingresa área dañada (m²) y se convierte a losas equivalentes (área/18).")
            area_dano_m2_rig = st.number_input(
                "Área del daño (m²)",
                min_value=0.0,
                step=0.01,
                value=0.0,
                key="area_dano_rig_in",
            )

        else:
            cant = st.number_input(
                "Cantidad",
                min_value=0.0,
                step=0.1,
                key="cant_in",
            )

        if st.form_submit_button("Añadir a la lista"):
            if opciones and "Error: Datos no encontrados" not in opciones:
                st.session_state.list_danos.append({
                    "Deterioro": tipo,
                    "Severidad": sev,
                    "Cantidad": float(cant),
                    "Largo_m": float(largo_bache),
                    "Area_m2": float(area_bache_m2),
                    "Area_rig_m2": float(area_dano_m2_rig),
                    "Pav": pav_type
                })
                st.rerun()

# ------------------------------------------------------------
# OUTPUT PANEL
# ------------------------------------------------------------
with col_out:
    st.subheader("📊 Resultados")

    if st.session_state.list_danos:
        # Filtra por tipo de pavimento actual
        danos_filtrados = [d for d in st.session_state.list_danos if d["Pav"] == pav_type]

        if not danos_filtrados:
            st.info("No hay eventos para este tipo de pavimento.")
        else:
            # -------------------------
            # 1) Construir tabla "tipo MATLAB": cantidad_equiv por evento
            # -------------------------
            rows = []
            for d in danos_filtrados:
                det = str(d["Deterioro"])
                sev_i = str(d["Severidad"])

                if pav_type == "FLEXIBLE":
                    if det == "baches":
                        qty_equiv = baches_qty_equiv(d.get("Largo_m", 0.0), d.get("Area_m2", 0.0))
                        dens = (qty_equiv / float(area_total)) * 100.0
                    else:
                        qty_equiv = float(d.get("Cantidad", 0.0) or 0.0)
                        dens = (qty_equiv / float(area_total)) * 100.0
                        if det in ["ahuellamiento", "abultamientos_hundimientos"]:
                            dens = min(dens, P["CAP_DENS_AHUELL_PCT"])

                else:
                    # RÍGIDO (MATLAB): qty_equiv = area_dano_m2 / 18; dens = 100 * qty / ceil(area/18)
                    area_dano_m2 = float(d.get("Area_rig_m2", 0.0) or 0.0)
                    qty_equiv = max(0.0, area_dano_m2) / float(P["LOSA_AREA_M2"])
                    lt = losas_totales(area_total)
                    dens = 0.0 if lt <= 0 else (qty_equiv / float(lt)) * 100.0

                rows.append({
                    "Deterioro": det,
                    "Severidad": sev_i,
                    "Cantidad": float(d.get("Cantidad", 0.0) or 0.0),
                    "Cantidad_equiv": float(qty_equiv),
                    "Densidad_%_evento": float(dens),
                })

            df_evt = pd.DataFrame(rows)

            # -------------------------
            # 2) Agrupar por Deterioro + Severidad (MATLAB: construir_tabla_reporte)
            # -------------------------
            grp = (
                df_evt
                .groupby(["Deterioro", "Severidad"], as_index=False)
                .agg({"Cantidad_equiv": "sum"})
            )

            # Recalcular densidad por grupo con reglas de pavimento
            dens_list = []
            dv_list = []
            for _, r in grp.iterrows():
                det = r["Deterioro"]
                sev_i = r["Severidad"]
                qty = float(r["Cantidad_equiv"] or 0.0)

                if pav_type == "FLEXIBLE":
                    dens_g = 0.0 if area_total <= 0 else (qty / float(area_total)) * 100.0
                    if det in ["ahuellamiento", "abultamientos_hundimientos"]:
                        dens_g = min(dens_g, P["CAP_DENS_AHUELL_PCT"])
                else:
                    lt = losas_totales(area_total)
                    dens_g = 0.0 if lt <= 0 else (qty / float(lt)) * 100.0

                dens_list.append(float(dens_g))
                dv_list.append(float(get_dv(pav_type, det, sev_i, float(dens_g))))

            grp["Densidad_%"] = np.round(dens_list, 3)
            grp["Deducido_DV"] = np.round(dv_list, 2)

            # -------------------------
            # 3) ASTM: HDV -> m -> iteración CDV (con DVs agrupados)
            # -------------------------
            dvs_finales = sorted([float(v) for v in grp["Deducido_DV"].values if v and v > 0], reverse=True)

            if not dvs_finales:
                st.warning("Todos los DV resultaron 0. Verifica VD/curvas y rangos de densidad.")
            else:
                hdv = dvs_finales[0]

                m = 1 + (9 / 98) * (100 - hdv)
                m = min(max(m, 1.0), 10.0)

                vals_iter = dvs_finales[: int(np.ceil(m))]
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
                    i_min = min(idx_gt2, key=lambda i: vals_iter[i])
                    vals_iter[i_min] = 2.0

                pci = max(0.0, 100.0 - max_cdv)

                # -------------------------
                # 4) Mostrar
                # -------------------------
                c1, c2 = st.columns(2)
                c1.metric("PCI", f"{round(pci, 1)}")

                rating = (
                    "BUENO" if pci > 85 else
                    "SATISFACTORIO" if pci > 70 else
                    "REGULAR" if pci > 55 else
                    "MALO" if pci > 40 else
                    "MUY MALO" if pci > 25 else
                    "CRITICO" if pci > 10 else
                    "FALLADO"
                )
                c2.metric("Condición", rating)

                st.markdown("---")

                if pav_type == "RIGIDO":
                    st.caption(f"Rígido: losas totales (ceil(Área/18)) = {losas_totales(area_total)}")

                st.subheader("Tabla agrupada (equivalente al reporte MATLAB)")
                st.table(grp[["Deterioro", "Severidad", "Cantidad_equiv", "Densidad_%", "Deducido_DV"]])

                st.subheader("Eventos ingresados (auditoría)")
                st.table(df_evt)

    else:
        st.info("Agregue deterioros para ver el cálculo.")