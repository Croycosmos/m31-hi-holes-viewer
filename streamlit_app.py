from __future__ import annotations

from pathlib import Path
import json
import base64
import math

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / 'data'
HI_CATALOG_PATH = DATA_DIR / 'holes_catalog_streamlit.csv'
OBJECTS_CATALOG_PATH = DATA_DIR / 'objects_catalog_streamlit.csv'
NEW_CANDIDATES_CATALOG_PATH = DATA_DIR / 'new_hi_candidates_catalog_streamlit.csv'
BACKGROUND_PNG = DATA_DIR / 'm31_background.png'
BACKGROUND_META = DATA_DIR / 'm31_background_meta.json'
CANDIDATE_SUMMARY_DIR = APP_DIR / 'figures' / 'candidate_summary_pdfs'
CANDIDATE_SUMMARY_PDFS = [
    {
        'title': 'BB86 141 cavités - géométrie 2DCG / joint refit',
        'caption': 'Distributions des paramètres géométriques utilisés pour les 141 cavités BB86.',
        'filename': 'bb86_2dcg_joint_histograms_main.pdf',
    },
    {
        'title': 'BB86 table.fits vs géométrie 2DCG',
        'caption': 'Comparaison entre les valeurs BB86 historiques et la géométrie 2DCG / refit utilisée sur le site.',
        'filename': 'bb86_table_vs_2dcg_refit_histograms.pdf',
    },
    {
        'title': 'Candidats HI 247 objets - distributions globales',
        'caption': 'Histogrammes globaux pour les 72 nouveaux candidats retenus et les 175 candidats potentiels.',
        'filename': 'candidates_247_histograms_main.pdf',
    },
    {
        'title': '72 candidats retenus — distributions des paramètres',
        'caption': 'Histogrammes globaux pour les 72 candidats H I retenus.',
        'filename': 'new_hi_candidates_v9_histograms_main.pdf',
    },
    {
        'title': '175 candidats potentiels — distributions des paramètres',
        'caption': 'Histogrammes globaux pour les 175 candidats potentiels du filtre v9.',
        'filename': 'v9_potential_175_histograms_main.pdf',
    },
    {
        'title': 'Potentiels 175 candidats - critères du statut potentiel',
        'caption': 'Décompte des critères ayant conduit au statut potentiel dans le filtre v9.',
        'filename': 'v9_potential_reason_counts_split.pdf',
    },
]

V11_POPULATION_DIR = APP_DIR / 'figures' / 'intermediate' / 'RefineV9Candidates2DCG_v11_merged_population'
V11_POPULATION_DIAGNOSTIC_IMAGES = [
    {
        'title': '2DCG refit population - counts',
        'caption': 'Comptages globaux des classes et statuts v11 pour les 247 candidats.',
        'filename': 'v11_population_merged247_counts.png',
    },
    {
        'title': '2DCG refit population - histograms',
        'caption': 'Histogrammes globaux de la population v11 des 247 candidats.',
        'filename': 'v11_population_merged247_histograms.png',
    },
    {
        'title': '2DCG refit population - geometry scatter',
        'caption': 'Plans de dispersion géométriques pour les paramètres v11.',
        'filename': 'v11_population_merged247_geometry_scatter.png',
    },
    {
        'title': '2DCG refit population - core/control scatter',
        'caption': 'Comparaison core/control pour les scores et déficits v11.',
        'filename': 'v11_population_merged247_core_control_scatter.png',
    },
    {
        'title': '2DCG refit population - trial velocity scores',
        'caption': 'Scores selon les essais en vitesse.',
        'filename': 'v11_population_merged247_trial_velocity_scores.png',
    },
]
DISPLAY_FLIP_X = True

ARCSEC_PER_PC = 206265.0 / 690000.0
PC_PER_ARCMIN = 60.0 / ARCSEC_PER_PC

TRACER_OPTIONS = ['HI', 'Koch25 new HI candidate', 'Potential new HI Candidate', 'UV', 'Hα', 'Dust - HELGA', 'CO', 'IR', 'X-ray']
TRACER_COLORS = {
    'HI': 'deepskyblue',
    'Koch25 new HI candidate': 'lime',
    'Potential new HI Candidate': 'orange',
    'UV': 'violet',
    'Hα': 'red',
    'Dust - HELGA': 'gold',
    'CO': 'cyan',
    'IR': 'gold',
    'X-ray': 'lime',
}
TRACER_SYMBOLS = {
    'HI': 'circle',
    'Koch25 new HI candidate': 'circle',
    'Potential new HI Candidate': 'circle',
    'UV': 'diamond',
    'Hα': 'diamond',
    'Dust - HELGA': 'diamond',
    'CO': 'diamond',
    'IR': 'diamond',
    'X-ray': 'diamond',
}

st.set_page_config(page_title='M31 multi-tracer structures viewer', layout='wide')


def as_bool(value) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def value_to_text(value) -> str:
    if pd.isna(value):
        return ''
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ''
        return f'{float(value):.4g}'
    return str(value)


def candidate_selector_sort_key(df: pd.DataFrame) -> pd.Series:
    """Numeric candidate id used only to sort selection menus."""
    if df.empty:
        return pd.Series(dtype=float)

    for col in [
        'new_candidate_id',
        'candidate_id',
        'new_hi_candidate_id',
        'id',
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            if vals.notna().any():
                return vals

    labels = df.get('display_label', pd.Series('', index=df.index)).astype(str)
    return pd.to_numeric(labels.str.extract(r'(\d+)')[0], errors='coerce')


def sort_candidate_selector_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort new/potential H I candidates numerically for selectbox menus."""
    if df.empty:
        return df

    out = df.copy()
    out['_selector_sort_id'] = candidate_selector_sort_key(out)

    sort_cols = ['_selector_sort_id']
    if 'display_label' in out.columns:
        sort_cols.append('display_label')
    elif 'object_uid' in out.columns:
        sort_cols.append('object_uid')

    out = out.sort_values(sort_cols, na_position='last')
    return out.drop(columns=['_selector_sort_id']).reset_index(drop=True)


def numeric(row: pd.Series, col: str | None) -> float:
    if col is None or col not in row.index:
        return float('nan')
    try:
        return float(row[col])
    except Exception:
        return float('nan')



@st.cache_data(show_spinner=False)
def load_hi_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Seq'] = df['Seq'].astype(int)
    df['Seq_str'] = df['Seq'].map(lambda x: f'{x:03d}')
    if 'has_video' in df.columns:
        df['has_video'] = df['has_video'].map(as_bool)
    else:
        df['has_video'] = df.get('video_path', '').astype(str).str.len() > 0
    if 'video_url' in df.columns:
        df['has_video'] = df['has_video'] | df['video_url'].fillna('').astype(str).str.strip().str.len().gt(0)
    if 'tracer' not in df.columns:
        df['tracer'] = 'HI'
    if 'source_catalog' not in df.columns:
        df['source_catalog'] = 'BB86 + LGLBS 2DCG/joint contrast'
    if 'object_label' not in df.columns:
        df['object_label'] = df['Seq'].map(lambda x: f'HI {int(x):03d}')
    df['x_arcmin'] = pd.to_numeric(df['x_arcmin'], errors='coerce')
    df['y_arcmin'] = pd.to_numeric(df['y_arcmin'], errors='coerce')
    df['x_plot'] = -df['x_arcmin'] if DISPLAY_FLIP_X else df['x_arcmin']
    df['y_plot'] = df['y_arcmin']
    return df


@st.cache_data(show_spinner=False)
def load_objects_catalog(path: Path, hi_df: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        obj = pd.read_csv(path)
    else:
        rows = []
        for _, r in hi_df.iterrows():
            rows.append({
                'object_uid': f'HI_BB86_{int(r["Seq"]):03d}',
                'tracer': 'HI',
                'source_catalog': 'BB86 + LGLBS 2DCG/joint contrast',
                'display_label': f'HI {int(r["Seq"]):03d}',
                'hi_seq': int(r['Seq']),
                'x_arcmin': r['x_arcmin'],
                'y_arcmin': r['y_arcmin'],
                'marker_size': 8.0,
            })
        obj = pd.DataFrame(rows)

    for col in ['object_uid', 'tracer', 'source_catalog', 'display_label']:
        if col not in obj.columns:
            obj[col] = ''
        obj[col] = obj[col].fillna('').astype(str)
    for col in ['x_arcmin', 'y_arcmin', 'major_arcsec', 'minor_arcsec', 'pa_deg', 'marker_size']:
        if col not in obj.columns:
            obj[col] = np.nan
        obj[col] = pd.to_numeric(obj[col], errors='coerce')
    if 'hi_seq' not in obj.columns:
        obj['hi_seq'] = np.nan
    obj['hi_seq'] = pd.to_numeric(obj['hi_seq'], errors='coerce')
    if 'has_video' in obj.columns:
        obj['has_video'] = obj['has_video'].map(as_bool)
    else:
        obj['has_video'] = False
    obj['marker_size'] = obj['marker_size'].fillna(8.0).clip(3.0, 22.0)
    obj['x_plot'] = -obj['x_arcmin'] if DISPLAY_FLIP_X else obj['x_arcmin']
    obj['y_plot'] = obj['y_arcmin']
    return obj


@st.cache_data(show_spinner=False)
def load_background_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def image_file_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = 'image/jpeg' if suffix in {'.jpg', '.jpeg'} else 'image/png'
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode('utf-8')
    return f'data:{mime};base64,{b64}'


@st.cache_data(show_spinner=False)
def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode('utf-8')


def render_pdf_inline(path: Path, height: int = 700) -> None:
    if not path.exists():
        st.info(f'PDF not found: `{path.relative_to(APP_DIR) if path.is_relative_to(APP_DIR) else path}`')
        return
    b64 = file_to_base64(path)
    html = f"""
    <iframe
        src="data:application/pdf;base64,{b64}#toolbar=1&navpanes=0&scrollbar=1"
        width="100%"
        height="{int(height)}"
        style="border: 1px solid #ddd; border-radius: 6px; background: white;"
    ></iframe>
    """
    components.html(html, height=int(height) + 20, scrolling=False)



def render_v11_population_diagnostics() -> None:
    st.markdown('### 2DCG refit population diagnostics')
    st.caption('Global diagnostic figures for the 247-candidate population.')
    for i, item in enumerate(V11_POPULATION_DIAGNOSTIC_IMAGES):
        path = V11_POPULATION_DIR / item['filename']
        with st.expander(item['title'], expanded=False):
            st.caption(item['caption'])
            if path.exists():
                size_mb = path.stat().st_size / 1024**2
                st.write(f'File available — `{size_mb:.1f} MB`')
                st.image(str(path), width='stretch')
                with path.open('rb') as fh:
                    st.download_button(
                        'Download PNG',
                        data=fh.read(),
                        file_name=path.name,
                        mime='image/png',
                        key=f'v11_population_{i}_{path.name}',
                    )
            else:
                st.warning(f'Missing file: `{path.relative_to(APP_DIR)}`')


def render_candidate_summary_pdfs() -> None:
    st.markdown('### Candidate summary documents')
    st.caption('Global diagnostic PDF panels for the v9 new-candidate catalogues.')
    for i, item in enumerate(CANDIDATE_SUMMARY_PDFS):
        path = CANDIDATE_SUMMARY_DIR / item['filename']
        with st.expander(item['title'], expanded=(i == 0)):
            st.caption(item['caption'])
            if path.exists():
                size_mb = path.stat().st_size / 1024**2
                st.write(f'File available — `{size_mb:.1f} MB`')
                with path.open('rb') as fh:
                    st.download_button(
                        'Download PDF',
                        data=fh.read(),
                        file_name=path.name,
                        mime='application/pdf',
                        key=f'download_summary_pdf_{i}',
                    )
                render_pdf_inline(path, height=680)
            else:
                st.info(f'PDF not found: `{path.relative_to(APP_DIR)}`')


def transform_background_extent(meta: dict) -> dict:
    x_min = float(meta['x_min'])
    x_max = float(meta['x_max'])
    y_min = float(meta['y_min'])
    y_max = float(meta['y_max'])
    if DISPLAY_FLIP_X:
        x0 = -x_max
        x1 = -x_min
    else:
        x0 = x_min
        x1 = x_max
    out = {'x_min': float(min(x0, x1)), 'x_max': float(max(x0, x1)), 'y_min': y_min, 'y_max': y_max}
    out['sizex'] = out['x_max'] - out['x_min']
    out['sizey'] = out['y_max'] - out['y_min']
    return out


def resolve_video_path(row: pd.Series) -> Path | None:
    raw = str(row.get('video_path', '')).strip()
    if not raw or raw.lower() in {'nan', 'none'}:
        return None
    path = Path(raw)
    if path.is_absolute():
        path = Path('videos') / path.name
    return APP_DIR / path


def resolve_video_url(row: pd.Series) -> str:
    raw = str(row.get('video_url', '')).strip()
    if not raw or raw.lower() in {'nan', 'none'}:
        return ''
    return raw


def resolve_optional_path(row: pd.Series, column: str) -> Path | None:
    raw = str(row.get(column, '')).strip()
    if not raw or raw.lower() in {'nan', 'none'}:
        return None
    path = Path(raw)
    if path.is_absolute():
        path = Path(path.name)
    return APP_DIR / path


def render_optional_png(row: pd.Series, title: str, column: str, missing_message: str) -> None:
    st.markdown(f'### {title}')
    path = resolve_optional_path(row, column)
    if path is not None and path.exists():
        st.image(str(path), width='stretch')
    else:
        st.info(missing_message)


def _row_value(row: pd.Series, column: str | None) -> str:
    if column is None or column not in row.index:
        return ''
    return value_to_text(row[column])


def build_refit_table(row: pd.Series) -> pd.DataFrame:
    specs = [
        ('Maj [pc]', 'joint__Maj_best_pc', 'cg2d__Maj_growth_pc'),
        ('Min [pc]', 'joint__Min_best_pc', 'cg2d__Min_growth_pc'),
        ('PA [deg]', 'joint__PA_astro_best_deg', 'cg2d__PA_geometry_deg'),
        ('HRV [km/s]', 'joint__HRV_used_kms', 'cg2d__hrv_kms'),
        ('N channels', 'joint__nch_used', 'cg2d__nch_best'),
        ('v min [km/s]', None, 'cg2d__vel_lo_kms'),
        ('v max [km/s]', None, 'cg2d__vel_hi_kms'),
        ('dx [arcsec]', 'joint__dx_best_arcsec', None),
        ('dy [arcsec]', 'joint__dy_best_arcsec', None),
        ('dr [arcsec]', 'joint__dr_best_arcsec', None),
        ('contrast base', 'joint__contrast_base', None),
        ('contrast best', 'joint__contrast_best', None),
        ('ΔI/σ base', 'joint__delta_I_sn_base', None),
        ('ΔI/σ best', 'joint__delta_I_sn_best', 'cg2d__delta_I_sn'),
        ('median Nbeam center', None, 'cg2d__med_Nbeam_center'),
        ('median Nbeam ring', None, 'cg2d__med_Nbeam_ring'),
        ('major r50 [arcsec]', None, 'cg2d__major_r50_deficit_arcsec'),
        ('major r80 [arcsec]', None, 'cg2d__major_r80_deficit_arcsec'),
        ('minor r50 [arcsec]', None, 'cg2d__minor_r50_deficit_arcsec'),
        ('minor r80 [arcsec]', None, 'cg2d__minor_r80_deficit_arcsec'),
        ('major growth score', None, 'cg2d__major_growth_score'),
        ('minor growth score', None, 'cg2d__minor_growth_score'),
        ('status / source', 'joint__status', 'cg2d__major_source'),
    ]
    rows = []
    for parameter, joint_col, cg2d_col in specs:
        joint_val = _row_value(row, joint_col)
        cg2d_val = _row_value(row, cg2d_col)
        if joint_val or cg2d_val:
            rows.append({'parameter': parameter, 'joint_contrast_refit': joint_val, '2DCG': cg2d_val})
    return pd.DataFrame(rows)


def hi_size_pc(row: pd.Series) -> tuple[float, float]:
    maj_candidates = ['cg2d__Maj_growth_pc', 'joint__Maj_best_pc', 'Maj']
    min_candidates = ['cg2d__Min_growth_pc', 'joint__Min_best_pc', 'Min']
    maj = next((numeric(row, c) for c in maj_candidates if np.isfinite(numeric(row, c)) and numeric(row, c) > 0), float('nan'))
    minu = next((numeric(row, c) for c in min_candidates if np.isfinite(numeric(row, c)) and numeric(row, c) > 0), float('nan'))
    return maj, minu


def hi_radius_arcmin(row: pd.Series) -> float:
    maj, minu = hi_size_pc(row)
    if np.isfinite(maj) and np.isfinite(minu) and maj > 0 and minu > 0:
        return 0.5 * math.sqrt(maj * minu) / PC_PER_ARCMIN
    if np.isfinite(maj) and maj > 0:
        return 0.5 * maj / PC_PER_ARCMIN
    return 1.0


def selected_hi_geometry(row: pd.Series) -> dict:
    # BB86 rows store sizes in pc. New-candidate rows may store angular sizes directly.
    major_arcsec = numeric(row, 'major_arcsec')
    minor_arcsec = numeric(row, 'minor_arcsec')
    if np.isfinite(major_arcsec) and major_arcsec > 0:
        a = 0.5 * major_arcsec / 60.0
        b = 0.5 * minor_arcsec / 60.0 if np.isfinite(minor_arcsec) and minor_arcsec > 0 else a
        r_eq = math.sqrt(max(a, 1e-9) * max(b, 1e-9))
    else:
        maj, minu = hi_size_pc(row)
        if not np.isfinite(maj) or maj <= 0:
            maj = numeric(row, 'Maj')
        if not np.isfinite(minu) or minu <= 0:
            minu = numeric(row, 'Min')
        a = 0.5 * maj / PC_PER_ARCMIN if np.isfinite(maj) and maj > 0 else hi_radius_arcmin(row)
        b = 0.5 * minu / PC_PER_ARCMIN if np.isfinite(minu) and minu > 0 else hi_radius_arcmin(row)
        r_eq = hi_radius_arcmin(row)
    pa = numeric(row, 'cg2d__PA_geometry_deg')
    if not np.isfinite(pa):
        pa = numeric(row, 'joint__PA_astro_best_deg')
    if not np.isfinite(pa):
        pa = numeric(row, 'PA_astro_deg')
    if not np.isfinite(pa):
        pa = numeric(row, 'pa_deg')
    if not np.isfinite(pa):
        pa = numeric(row, 'PA')
    if not np.isfinite(pa):
        pa = 0.0
    return {'a': a, 'b': b, 'pa': pa, 'r_eq': r_eq}


def ellipse_points(cx: float, cy: float, a: float, b: float, pa_deg: float, n: int = 181) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    pa = np.deg2rad(pa_deg)
    # PA measured from North toward East in the native x=East, y=North plane.
    x = cx + a * np.cos(t) * np.sin(pa) + b * np.sin(t) * np.sin(pa + np.pi / 2.0)
    y = cy + a * np.cos(t) * np.cos(pa) + b * np.sin(t) * np.cos(pa + np.pi / 2.0)
    if DISPLAY_FLIP_X:
        x = -x
    return x, y


def circle_points(cx: float, cy: float, radius: float, n: int = 181) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    x = cx + radius * np.cos(t)
    y = cy + radius * np.sin(t)
    if DISPLAY_FLIP_X:
        x = -x
    return x, y


def nearest_context_tables(row: pd.Series, objects_df: pd.DataFrame, max_radius_factor: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    cx = numeric(row, 'x_arcmin')
    cy = numeric(row, 'y_arcmin')
    geom = selected_hi_geometry(row)
    r = geom.get('r_eq', float('nan'))
    if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(r) or r <= 0:
        return pd.DataFrame(), pd.DataFrame()

    # Context layers only. H I holes and H I candidates are the selected structures;
    # UV/Hα/CO/IR/X-ray remain environmental tracers around them.
    context_tracers = ['UV', 'Hα', 'Dust - HELGA', 'CO', 'IR', 'X-ray']
    obj = objects_df.loc[objects_df['tracer'].isin(context_tracers)].copy()
    if obj.empty:
        return pd.DataFrame(), pd.DataFrame()

    obj['distance_arcmin'] = np.sqrt((obj['x_arcmin'] - cx) ** 2 + (obj['y_arcmin'] - cy) ** 2)
    obj['distance_R'] = obj['distance_arcmin'] / r
    near = obj.loc[obj['distance_R'] <= max_radius_factor].copy().sort_values(['distance_R', 'tracer'])

    rows = []
    for tracer in context_tracers:
        sub = obj.loc[obj['tracer'] == tracer].copy()
        if sub.empty:
            rows.append({'tracer': tracer, 'N ≤ 1R': 0, 'N ≤ 2R': 0, 'N ≤ 3R': 0, 'nearest': '', 'nearest distance [arcmin]': '', 'nearest distance [R]': ''})
            continue
        sub = sub.sort_values('distance_arcmin')
        nearest = sub.iloc[0]
        rows.append({
            'tracer': tracer,
            'N ≤ 1R': int((sub['distance_R'] <= 1.0).sum()),
            'N ≤ 2R': int((sub['distance_R'] <= 2.0).sum()),
            'N ≤ 3R': int((sub['distance_R'] <= 3.0).sum()),
            'nearest': str(nearest.get('display_label', '')),
            'nearest distance [arcmin]': f'{nearest["distance_arcmin"]:.2f}',
            'nearest distance [R]': f'{nearest["distance_R"]:.2f}',
        })
    summary = pd.DataFrame(rows)
    keep_cols = ['display_label', 'tracer', 'source_catalog', 'distance_arcmin', 'distance_R', 'x_arcmin', 'y_arcmin', 'age_myr', 'mass_msun', 'luminosity', 'notes']
    keep_cols = [c for c in keep_cols if c in near.columns]
    near_table = near[keep_cols].head(80).copy()
    for _txt_col in ['display_label', 'source_catalog', 'notes', 'tracer']:
        if _txt_col in near_table.columns:
            near_table[_txt_col] = (
                near_table[_txt_col]
                .fillna('')
                .astype(str)
                .str.replace('not a cavity', 'context tracer', case=False, regex=False)
                .str.replace('Not a cavity', 'context tracer', regex=False)
            )
    for _txt_col in ['display_label', 'source_catalog', 'notes', 'tracer']:
        if _txt_col in near_table.columns:
            near_table[_txt_col] = (
                near_table[_txt_col]
                .fillna('')
                .astype(str)
                .str.replace('not a cavity', 'context tracer', case=False, regex=False)
                .str.replace('Not a cavity', 'context tracer', regex=False)
            )
    for _txt_col in ['display_label', 'source_catalog', 'notes', 'tracer']:
        if _txt_col in near_table.columns:
            near_table[_txt_col] = (
                near_table[_txt_col]
                .fillna('')
                .astype(str)
                .str.replace('not a cavity', 'context tracer', case=False, regex=False)
                .str.replace('Not a cavity', 'context tracer', regex=False)
            )
    for col in ['distance_arcmin', 'distance_R']:
        if col in near_table.columns:
            near_table[col] = near_table[col].map(lambda v: f'{float(v):.3g}' if pd.notna(v) else '')
    return summary, near_table


def video_player_from_file(path: Path, height: int = 520) -> None:
    if not path.exists():
        st.error(f'Video file not found: {path}')
        return
    video_bytes = path.read_bytes()
    video_b64 = base64.b64encode(video_bytes).decode('utf-8')
    html = f"""
    <video width="100%" controls preload="metadata" style="background-color: black;">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser cannot play this MP4 video.
    </video>
    """
    components.html(html, height=height)
    st.download_button(label='Download MP4', data=video_bytes, file_name=path.name, mime='video/mp4', width='stretch')


def video_player_from_url(url: str, height: int = 520) -> None:
    url = str(url).strip()
    if not url:
        st.error('Empty video URL.')
        return
    html = f"""
    <video width="100%" controls preload="metadata" style="background-color: black;">
        <source src="{url}" type="video/mp4">
        Your browser cannot play this MP4 video.
    </video>
    <p style="font-size: 0.85rem;">
        <a href="{url}" target="_blank" rel="noopener noreferrer">Open / download MP4</a>
    </p>
    """
    components.html(html, height=height)


def build_m31_figure(
    df_visible: pd.DataFrame,
    df_extent: pd.DataFrame,
    background_png: Path,
    meta: dict,
    selected_hi_row: pd.Series | None,
    show_search_rings: bool,
) -> go.Figure:
    fig = go.Figure()
    has_bg = background_png.exists() and bool(meta)
    if has_bg:
        bg = transform_background_extent(meta)
        fig.add_layout_image(dict(
            source=image_file_to_data_uri(background_png), xref='x', yref='y', x=bg['x_min'], y=bg['y_max'],
            sizex=bg['sizex'], sizey=bg['sizey'], sizing='stretch', opacity=1.0, layer='below'))
        fig.update_xaxes(range=[bg['x_min'], bg['x_max']])
        fig.update_yaxes(range=[bg['y_min'], bg['y_max']])
    else:
        xpad = 10.0
        ypad = 10.0
        fig.update_xaxes(range=[float(df_extent['x_plot'].min()) - xpad, float(df_extent['x_plot'].max()) + xpad])
        fig.update_yaxes(range=[float(df_extent['y_plot'].min()) - ypad, float(df_extent['y_plot'].max()) + ypad])

    for tracer in TRACER_OPTIONS:
        sub = df_visible.loc[df_visible['tracer'] == tracer].copy()
        if sub.empty:
            continue
        marker = dict(
            size=sub['marker_size'],
            color=TRACER_COLORS.get(tracer, 'white'),
            opacity=0.82 if tracer != 'HI' else 0.9,
            line=dict(width=0.6, color='white'),
            symbol=TRACER_SYMBOLS.get(tracer, 'circle'),
        )
        if tracer == 'HI':
            marker['opacity'] = 0.88
        fig.add_trace(go.Scatter(
            x=sub['x_plot'], y=sub['y_plot'], mode='markers', name=tracer,
            marker=marker,
            text=sub['display_label'],
            customdata=sub[['object_uid', 'tracer', 'source_catalog', 'hi_seq', 'x_arcmin', 'y_arcmin', 'major_arcsec', 'minor_arcsec', 'pa_deg']].fillna('').to_numpy(),
            hovertemplate=(
                '%{text}<br>'
                'Tracer=%{customdata[1]}<br>'
                'Catalog=%{customdata[2]}<br>'
                'X=%{customdata[4]:.2f} arcmin<br>'
                'Y=%{customdata[5]:.2f} arcmin<br>'
                'Major=%{customdata[6]} arcsec<br>'
                'Minor=%{customdata[7]} arcsec<br>'
                'PA=%{customdata[8]} deg<br>'
                '<extra></extra>'
            ),
        ))

    if selected_hi_row is not None:
        cx = numeric(selected_hi_row, 'x_arcmin')
        cy = numeric(selected_hi_row, 'y_arcmin')
        if np.isfinite(cx) and np.isfinite(cy):
            tracer = str(selected_hi_row.get('tracer', 'HI'))
            is_bb86_hi = 'Seq' in selected_hi_row.index and tracer in {'HI', '', 'nan'}
            geom = selected_hi_geometry(selected_hi_row)

            if is_bb86_hi:
                seq = int(selected_hi_row['Seq'])
                label = f'HI {seq:03d}'
                trace_name = 'selected H I ellipse'
                line_color = 'red'
                marker_color = 'red'
                marker_symbol = 'circle-open'
            else:
                label = str(selected_hi_row.get('display_label', selected_hi_row.get('object_uid', 'selected object')))
                trace_name = f'selected {tracer} ellipse'
                line_color = TRACER_COLORS.get(tracer, 'red')
                marker_color = line_color
                marker_symbol = 'circle-open'

            xe, ye = ellipse_points(cx, cy, geom['a'], geom['b'], geom['pa'])
            fig.add_trace(go.Scatter(
                x=xe, y=ye, mode='lines', name=trace_name,
                line=dict(color=line_color, width=3), hoverinfo='skip'
            ))

            # 1R/2R/3R are drawn for any selected H I structure: BB86 holes,
            # Koch25 new candidates, and low-confidence/potential candidates.
            if show_search_rings:
                for factor, dash in [(1, 'dot'), (2, 'dash'), (3, 'longdash')]:
                    xc, yc = circle_points(cx, cy, geom['r_eq'] * factor)
                    fig.add_trace(go.Scatter(
                        x=xc, y=yc, mode='lines', name=f'{factor}R search ring',
                        line=dict(color='white', width=1, dash=dash), opacity=0.65, hoverinfo='skip'
                    ))

            fig.add_trace(go.Scatter(
                x=[-cx if DISPLAY_FLIP_X else cx], y=[cy], mode='markers+text', name='selected object', showlegend=False, marker=dict(size=21, color=marker_color, symbol=marker_symbol, line=dict(width=3.0, color=marker_color)),
                text=[label], textposition='top center', textfont=dict(color=marker_color, size=13), hoverinfo='skip'
            ))

    x_title = 'Displayed X [arcmin] — East left / West right' if DISPLAY_FLIP_X else 'X [arcmin]'
    fig.update_layout(
        title='M31 — multi-tracer context map', template='plotly_dark', height=820,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0.35)', bordercolor='rgba(255,255,255,0.25)', borderwidth=1),
        xaxis_title=x_title, yaxis_title='Y [arcmin] — North/South', clickmode='event+select')
    fig.update_yaxes(scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig


def build_local_context_figure(
    selected_row: pd.Series,
    objects_df: pd.DataFrame,
    background_png: Path,
    meta: dict,
    show_search_rings: bool,
    selected_tracers: list[str] | None = None,
) -> go.Figure | None:
    """Build an interactive 3R local zoom around the selected H I structure.

    The zoom is derived from the same x/y offset system as the global M31 map.
    It uses the same background image and marker conventions, then restricts the
    axes to a region slightly larger than 3R.
    """
    cx = numeric(selected_row, 'x_arcmin')
    cy = numeric(selected_row, 'y_arcmin')
    geom = selected_hi_geometry(selected_row)
    r_eq = float(geom.get('r_eq', np.nan))
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r_eq) and r_eq > 0):
        return None

    x0 = -cx if DISPLAY_FLIP_X else cx
    y0 = cy
    half_size = 3.6 * r_eq

    # Local zoom always shows environmental tracers, plus H I structures if the
    # user has them enabled in the global map. This keeps the plot aligned with
    # the near-object table without forcing every global layer to be visible.
    context_tracers = ['UV', 'Hα', 'Dust - HELGA', 'CO', 'IR', 'X-ray']
    hi_structure_tracers = ['HI', 'Koch25 new HI candidate', 'Potential new HI Candidate']
    chosen = list(selected_tracers or [])
    local_tracers = []
    for tracer in context_tracers + hi_structure_tracers + chosen:
        if tracer not in local_tracers:
            local_tracers.append(tracer)

    local = objects_df.loc[objects_df['tracer'].isin(local_tracers)].copy()
    if not local.empty:
        local = local.loc[
            np.isfinite(local['x_plot'])
            & np.isfinite(local['y_plot'])
            & (local['x_plot'] >= x0 - half_size)
            & (local['x_plot'] <= x0 + half_size)
            & (local['y_plot'] >= y0 - half_size)
            & (local['y_plot'] <= y0 + half_size)
        ].copy()

    title = str(selected_row.get('display_label', selected_row.get('object_label', selected_row.get('Seq_str', 'selected object'))))
    if not title or title.lower() in {'nan', 'none'}:
        title = 'selected object'

    fig = build_m31_figure(
        df_visible=local,
        df_extent=objects_df,
        background_png=background_png,
        meta=meta,
        selected_hi_row=selected_row,
        show_search_rings=show_search_rings,
    )
    fig.update_xaxes(range=[x0 - half_size, x0 + half_size])
    fig.update_yaxes(range=[y0 - half_size, y0 + half_size])
    # The Streamlit section title above the chart carries the label.
    # Keep the Plotly title empty so it cannot overlap the horizontal legend.
    fig.update_layout(
        title=None,
        height=560,
        margin=dict(l=20, r=20, t=82, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
    )
    return fig


def get_clicked_object(event) -> dict | None:
    points = []
    try:
        points = event.selection.points
    except Exception:
        try:
            points = event['selection']['points']
        except Exception:
            points = []
    if not points:
        return None
    p = points[0]
    custom = p.get('customdata') if hasattr(p, 'get') else None
    if custom is None or len(custom) < 4:
        return None
    return {'object_uid': str(custom[0]), 'tracer': str(custom[1]), 'source_catalog': str(custom[2]), 'hi_seq': custom[3]}


def build_external_object_table(obj_row: pd.Series) -> pd.DataFrame:
    cols = [
        ('tracer', 'tracer'), ('catalog', 'source_catalog'), ('label', 'display_label'),
        ('RA [deg]', 'ra_deg'), ('Dec [deg]', 'dec_deg'), ('X [arcmin]', 'x_arcmin'), ('Y [arcmin]', 'y_arcmin'),
        ('major [arcsec]', 'major_arcsec'), ('minor [arcsec]', 'minor_arcsec'), ('PA [deg]', 'pa_deg'),
        ('flux', 'flux'), ('luminosity', 'luminosity'), ('area [pc²]', 'area_pc2'),
        ('age [Myr]', 'age_myr'), ('mass [M☉]', 'mass_msun'), ('notes', 'notes'),
    ]
    rows = []
    for label, col in cols:
        if col in obj_row.index:
            val = value_to_text(obj_row[col])
            if val:
                rows.append({'parameter': label, 'value': val})
    return pd.DataFrame(rows)



@st.cache_data(show_spinner=False)
def load_new_candidates_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ['object_uid', 'tracer', 'source_catalog', 'display_label', 'validation_png', 'video_url', 'video_release_url', 'video_path', 'video_name', 'video_release_tag', 'v11_diagnostic_png', 'v11_plot_class', 'v11_passes_2dcg']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)
    for col in ['candidate_id', 'new_candidate_id', 'ra_deg', 'dec_deg', 'x_arcmin', 'y_arcmin', 'major_arcsec', 'minor_arcsec', 'pa_deg', 'v_center_kms', 'Maj_pc', 'Min_pc', 'PA_astro_deg', 'v9_final_score', 'v11_source_candidate_id', 'v11_match_sep_arcsec', 'v11_joint_2dcg_score', 'v11_final_Maj_pc', 'v11_final_Min_pc', 'v11_final_PA_astro_deg', 'v11_final_v_center_kms']:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['x_plot'] = -df['x_arcmin'] if DISPLAY_FLIP_X else df['x_arcmin']
    df['y_plot'] = df['y_arcmin']
    return df


def resolve_catalog_path(row: pd.Series, column: str) -> Path | None:
    raw = str(row.get(column, '')).strip()
    if not raw or raw.lower() in {'nan', 'none'}:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return APP_DIR / path


def build_candidate_table(row: pd.Series) -> pd.DataFrame:
    cols = [
        ('candidate ID', 'candidate_id'),
        ('RA [deg]', 'ra_deg'), ('Dec [deg]', 'dec_deg'),
        ('offset East/West from M31 centre [arcmin]', 'x_arcmin'),
        ('offset North/South from M31 centre [arcmin]', 'y_arcmin'),
        ('v centre [km/s]', 'v_center_kms'),
        ('Maj [pc]', 'Maj_pc'), ('Min [pc]', 'Min_pc'), ('PA [deg]', 'PA_astro_deg'),
        ('major [arcsec]', 'major_arcsec'), ('minor [arcsec]', 'minor_arcsec'),
        ('2DCG source ID', 'v11_source_candidate_id'),
        ('2DCG match separation [arcsec]', 'v11_match_sep_arcsec'),
        ('2DCG refit Maj [pc]', 'v11_final_Maj_pc'),
        ('2DCG refit Min [pc]', 'v11_final_Min_pc'),
        ('2DCG refit PA [deg]', 'v11_final_PA_astro_deg'),
        ('2DCG refit v centre [km/s]', 'v11_final_v_center_kms'),
        ('2DCG class', 'v11_plot_class'),
        ('passes 2DCG', 'v11_passes_2dcg'),
    ]
    rows = []
    for label, col in cols:
        if col in row.index:
            val = value_to_text(row[col])
            if val:
                rows.append({'parameter': label, 'value': val})
    return pd.DataFrame(rows)







def render_stat_figure_card(title: str, caption: str, png_rel: str | None = None, pdf_rel: str | None = None, csv_rel: str | None = None, key_prefix: str = '') -> None:
    st.markdown(f'#### {title}')
    if caption:
        st.caption(caption)

    png_path = APP_DIR / png_rel if png_rel else None
    pdf_path = APP_DIR / pdf_rel if pdf_rel else None
    csv_path = APP_DIR / csv_rel if csv_rel else None

    if png_path is not None and png_path.exists():
        st.image(str(png_path), width='stretch')
        st.caption('PNG available for this figure.')
    elif png_rel:
        st.warning(f'Missing PNG: `{png_rel}`')

    buttons = st.columns(2)

    with buttons[0]:
        if pdf_path is not None and pdf_path.exists():
            with pdf_path.open('rb') as fh:
                st.download_button(
                    'Download PDF',
                    data=fh.read(),
                    file_name=pdf_path.name,
                    mime='application/pdf',
                    key=f'{key_prefix}_pdf_{pdf_path.name}',
                )

    with buttons[1]:
        if csv_path is not None and csv_path.exists():
            with csv_path.open('rb') as fh:
                st.download_button(
                    'Download CSV',
                    data=fh.read(),
                    file_name=csv_path.name,
                    mime='text/csv',
                    key=f'{key_prefix}_csv_{csv_path.name}',
                )



def _first_numeric_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index, dtype=float)


def render_bb86_vs_new247_statistics(hi_df: pd.DataFrame, candidates_df: pd.DataFrame) -> None:
    st.subheader('BB86 2DCG refit vs new candidate 2DCG refit')
    st.caption(
        'Comparison based on the 2DCG refit geometry: BB86 known H I holes '
        'versus the full 247 blind-detection candidate population.'
    )

    png_rel = 'figures/final/bb86_vs_new247_2dcg/bb86_2dcg_vs_new247_2dcg_histograms_samebins_orange.png'
    pdf_rel = 'figures/final/bb86_vs_new247_2dcg/bb86_2dcg_vs_new247_2dcg_histograms_samebins_orange.pdf'

    png_path = APP_DIR / png_rel
    pdf_path = APP_DIR / pdf_rel

    with st.container(border=True):
        if png_path.exists():
            st.image(str(png_path), width='stretch')
        else:
            st.warning(f'Missing PNG: `{png_rel}`')

        if pdf_path.exists():
            with pdf_path.open('rb') as fh:
                st.download_button(
                    'Download PDF',
                    data=fh.read(),
                    file_name=pdf_path.name,
                    mime='application/pdf',
                    key='bb86_vs_new247_2dcg_samebins_pdf',
                )
        else:
            st.info(f'PDF not found: `{pdf_rel}`')


def render_global_statistics() -> None:
    st.subheader('Statistiques globales')
    st.caption('Histogrammes et figures de population pour BB86, les 72 nouvelles cavités retenues, les 175 cavités candidates et la population complète.')

    groups = {
        'BB86': [
            {
                'title': 'BB86 — catalogue original vs refit 2DCG',
                'caption': 'Comparaison entre les paramètres du catalogue BB86 et les paramètres refittés avec 2DCG.',
                'png': 'figures/final/bb86_2dcg_histograms/bb86_table_vs_2dcg_refit_histograms.png',
                'pdf': 'figures/final/bb86_2dcg_histograms/bb86_table_vs_2dcg_refit_histograms.pdf',
                'csv': 'figures/final/bb86_2dcg_histograms/bb86_2dcg_joint_histogram_input_with_derived_columns.csv',
            },
        ],
        '72 retenues': [
            {
                'title': '72 nouvelles cavités — histogrammes',
                'caption': 'Distributions principales des nouvelles cavités retenues.',
                'png': 'figures/final/new_candidates_v9_histograms/new_hi_candidates_v9_histograms_main.png',
                'pdf': 'figures/final/new_candidates_v9_histograms/new_hi_candidates_v9_histograms_main.pdf',
                'csv': 'figures/final/new_candidates_v9_histograms/new_hi_candidates_2dcg_v9_histogram_summary_stats.csv',
            },
        ],
        '175 candidates': [
            {
                'title': '175 cavités candidates — histogrammes',
                'caption': 'Distributions des cavités candidates.',
                'png': 'figures/final/potential_candidates_v9_histograms/v9_potential_175_histograms_main.png',
                'pdf': 'figures/final/potential_candidates_v9_histograms/v9_potential_175_histograms_main.pdf',
                'csv': 'figures/final/potential_candidates_v9_histograms/v9_potential_175_summary_stats.csv',
            },
            {
                'title': '175 cavités candidates — critères de classement',
                'caption': 'Décompte des critères qui placent ces objets dans la classe candidate.',
                'png': 'figures/final/potential_candidates_v9_histograms/v9_potential_reason_counts_split.png',
                'pdf': 'figures/final/potential_candidates_v9_histograms/v9_potential_reason_counts_split.pdf',
                'csv': 'figures/final/potential_candidates_v9_histograms/v9_potential_reason_counts_split.csv',
            },
            {
                'title': '175 cavités candidates — critères dominants',
                'caption': 'Critères les plus fréquents dans la population candidate.',
                'png': 'figures/final/potential_candidates_v9_histograms/v9_potential_reason_counts_combined_top20.png',
                'pdf': 'figures/final/potential_candidates_v9_histograms/v9_potential_reason_counts_combined_top20.pdf',
                'csv': 'figures/final/potential_candidates_v9_histograms/v9_potential_reason_counts_combined.csv',
            },
        ],
        '247 total': [
            {
                'title': '247 cavités — retenues vs candidates',
                'caption': 'Comparaison des distributions entre les 72 retenues et les 175 candidates.',
                'png': 'figures/final/potential_candidates_v9_histograms/v9_kept72_vs_potential175_histograms.png',
                'pdf': 'figures/final/potential_candidates_v9_histograms/v9_kept72_vs_potential175_histograms.pdf',
                'csv': 'figures/final/potential_candidates_v9_histograms/v9_all_247_summary_stats.csv',
            },
        ],
        'Population refittée 2DCG': [
            {
                'title': 'Population refittée 2DCG — counts',
                'caption': 'Comptages globaux des classes et statuts.',
                'png': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_counts.png',
                'csv': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_derived.csv',
            },
            {
                'title': 'Population refittée 2DCG — histogrammes',
                'caption': 'Histogrammes de la population complète.',
                'png': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_histograms.png',
                'csv': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_derived.csv',
            },
            {
                'title': 'Population refittée 2DCG — géométrie',
                'caption': 'Plans de dispersion géométriques.',
                'png': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_geometry_scatter.png',
                'csv': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_derived.csv',
            },
            {
                'title': 'Population refittée 2DCG — core/control',
                'caption': 'Comparaison core/control.',
                'png': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_core_control_scatter.png',
                'csv': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_derived.csv',
            },
            {
                'title': 'Population refittée 2DCG — vitesses testées',
                'caption': 'Scores selon les essais en vitesse.',
                'png': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_trial_velocity_scores.png',
                'csv': 'figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_derived.csv',
            },
        ],
    }

    options = list(groups.keys())

    if hasattr(st, 'segmented_control'):
        chosen = st.segmented_control('Population', options, default='BB86', key='stats_population_group', width='stretch')
    else:
        chosen = st.radio('Population', options, index=0, horizontal=True, key='stats_population_group_radio')

    if not chosen:
        chosen = 'BB86'

    st.markdown(f'### {chosen}')

    cards = groups[chosen]
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for j, card in enumerate(cards[i:i + 2]):
            with cols[j]:
                with st.container(border=True):
                    render_stat_figure_card(
                        title=card.get('title', ''),
                        caption=card.get('caption', ''),
                        png_rel=card.get('png'),
                        pdf_rel=card.get('pdf'),
                        csv_rel=card.get('csv'),
                        key_prefix=f'stats_{chosen}_{i}_{j}'.replace(' ', '_'),
                    )



def render_multitracer_object_selector(hi_df: pd.DataFrame, candidates_df: pd.DataFrame):
    st.markdown('### Object selection')

    source_options = ['H I BB86', 'Koch25 new HI candidate', 'Potential new HI Candidate']
    current_source = st.session_state.get('selected_source_type', 'H I BB86')
    source_index = source_options.index(current_source) if current_source in source_options else 0

    source_type = st.selectbox(
        'Selected object type',
        source_options,
        index=source_index,
        key='multi_selected_object_type',
    )
    st.session_state['selected_source_type'] = source_type

    selected_hi_row = None
    selected_candidate_row = None

    if source_type == 'H I BB86':
        seq_options = hi_df['Seq_str'].tolist()
        current = f'{int(st.session_state.get("selected_hi_seq", int(hi_df["Seq"].iloc[0]))):03d}'
        idx = seq_options.index(current) if current in seq_options else 0
        seq_str = st.selectbox(
            'Selected object',
            seq_options,
            index=idx,
            key='multi_selected_bb86_seq',
            format_func=lambda s: f'BB86 {int(s):03d}',
        )
        st.session_state['selected_hi_seq'] = int(seq_str)
        selected_hi_row = hi_df.loc[hi_df['Seq'].eq(int(seq_str))].iloc[0]
        return source_type, selected_hi_row, None, selected_hi_row

    sub = candidates_df.loc[candidates_df['tracer'].eq(source_type)].copy() if not candidates_df.empty else pd.DataFrame()
    sub = sort_candidate_selector_rows(sub)
    if sub.empty:
        st.warning(f'No row found for {source_type}.')
        return source_type, None, None, None

    def _cand_id(row):
        for c in ['new_candidate_id', 'candidate_id']:
            if c in row.index:
                try:
                    return int(float(row[c]))
                except Exception:
                    pass
        return -1

    label_prefix = 'New cavity' if source_type == 'Koch25 new HI candidate' else 'Candidate cavity'
    sub['_selector_label'] = sub.apply(lambda r: f'{label_prefix} {_cand_id(r):04d}', axis=1)
    uid_options = sub['object_uid'].astype(str).tolist()
    label_map = dict(zip(sub['object_uid'].astype(str), sub['_selector_label'].astype(str)))

    current_uid = str(st.session_state.get('selected_candidate_uid', ''))
    idx = uid_options.index(current_uid) if current_uid in uid_options else 0

    uid = st.selectbox(
        'Selected object',
        uid_options,
        index=idx,
        key=f'multi_selected_candidate_uid_{source_type}',
        format_func=lambda u: label_map.get(str(u), str(u)),
    )

    st.session_state['selected_candidate_uid'] = str(uid)
    selected_candidate_row = sub.loc[sub['object_uid'].astype(str).eq(str(uid))].iloc[0]
    return source_type, None, selected_candidate_row, selected_candidate_row


def render_multitracer_panel(selected_source_type: str, selected_hi_row, selected_candidate_row) -> None:
    st.subheader('Multi-tracer context')

    row = selected_hi_row if selected_source_type == 'H I BB86' else selected_candidate_row
    if row is None:
        st.info('Select an H I object first.')
        return

    if selected_source_type == 'H I BB86':
        seq = numeric(row, 'Seq')
        if not np.isfinite(seq):
            st.warning('No BB86 sequence number found for this object.')
            return
        seq = int(seq)
        label = f'BB86 H I hole {seq:03d}'
        base = APP_DIR / 'figures' / 'final' / 'multitracer' / 'bb86_zoom_v4'
        stem = f'BB86_{seq:03d}_multitracer_zoom_v4'
    else:
        sid = numeric(row, 'v11_source_candidate_id')
        label = str(row.get('display_label', row.get('object_uid', 'selected candidate')))
        if not np.isfinite(sid):
            st.warning('No v11 source_candidate_id found for this candidate.')
            return
        sid = int(sid)
        base = APP_DIR / 'figures' / 'final' / 'multitracer' / 'new247_zoom_v4'
        stem = f'NEW247_v11_best_{sid:04d}_multitracer_zoom_v4'

    png_path = base / f'{stem}.png'
    pdf_path = base / f'{stem}.pdf'

    st.markdown(f'### {label}')
    st.caption('Panels show local context around the selected H I structure. Dust - HELGA is the dust/FIR-sub-mm tracer.')

    if png_path.exists():
        st.image(str(png_path), width='stretch')
        st.caption('Multi-tracer PNG available for this object.')
    else:
        st.warning(f'No multi-tracer PNG found for this selected object: `{png_path.relative_to(APP_DIR)}`')

    if pdf_path.exists():
        with pdf_path.open('rb') as fh:
            st.download_button(
                'Download multi-tracer PDF',
                data=fh.read(),
                file_name=pdf_path.name,
                mime='application/pdf',
                key=f'multitracer_pdf_{selected_source_type}_{stem}',
            )
    else:
        st.caption('No multi-tracer PDF found for this selected object.')



NO_2DCG_BB86 = {15, 23, 29, 52, 79, 83, 90, 128}


def clean_video_label(raw: str) -> str:
    raw = str(raw).strip()
    if not raw:
        return ''
    name = raw.split('?')[0].rstrip('/').split('/')[-1]
    for token in ['_dark', 'dark_', 'dark']:
        name = name.replace(token, '')
    name = name.replace('__', '_')
    return name


def format_seq_list(values, prefix='BB86 '):
    vals = []
    for v in values:
        try:
            vals.append(f'{prefix}{int(v):03d}')
        except Exception:
            pass
    return ', '.join(vals)





def render_coverage_overview() -> None:
    st.markdown('### Coverage overview')

    bb86_v4 = APP_DIR / 'figures' / 'final' / 'multitracer' / 'bb86_zoom_v4'
    new_v4 = APP_DIR / 'figures' / 'final' / 'multitracer' / 'new247_zoom_v4'
    bb86_pngs = sorted(bb86_v4.glob('BB86_*_multitracer_zoom_v4.png'))
    new_pngs = sorted(new_v4.glob('NEW247_*_multitracer_zoom_v4.png'))

    st.caption(f'Multi-tracer panels available: BB86 = {len(bb86_pngs)}, new/potential candidates = {len(new_pngs)}.')

    summary_candidates = [
        APP_DIR / 'data' / 'multitracer_coverage_master.csv',
        APP_DIR / 'figures' / 'final' / 'multitracer' / 'bb86_zoom_v4' / 'bb86_multitracer_zoom_v4_summary.csv',
        APP_DIR / 'figures' / 'final' / 'multitracer' / 'bb86_zoom_v2' / 'bb86_multitracer_zoom_v2_summary.csv',
    ]

    summary_path = next((x for x in summary_candidates if x.exists()), None)
    if summary_path is None:
        st.caption('No exported PHAST/SITELLE coverage table found yet. Per-object coverage is shown directly in each multi-tracer panel.')
        return

    try:
        cov = pd.read_csv(summary_path)
    except Exception:
        st.caption('A coverage table exists, but it could not be read. Per-object coverage is shown directly in each multi-tracer panel.')
        return

    seq_col = next((c for c in cov.columns if c.lower() in {'seq', 'bb86_seq', 'hi_seq'}), None)
    sitelle_col = next((c for c in cov.columns if 'sitelle' in c.lower()), None)
    phast_col = next((c for c in cov.columns if 'phast' in c.lower() or 'phat' in c.lower()), None)

    if seq_col is None or sitelle_col is None or phast_col is None:
        st.caption('Joint PHAST/SITELLE coverage table not exported yet. Per-object coverage is shown directly in each multi-tracer panel.')
        return

    def is_ok(s):
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors='coerce').fillna(0).gt(0)
        return s.fillna('').astype(str).str.lower().isin({'true', '1', 'yes', 'ok', 'covered'})

    seq = pd.to_numeric(cov[seq_col], errors='coerce')
    sit = is_ok(cov[sitelle_col])
    pha = is_ok(cov[phast_col])

    both = seq.loc[sit & pha].dropna().astype(int).tolist()
    one = seq.loc[sit ^ pha].dropna().astype(int).tolist()

    both_txt = ', '.join(f'{x:03d}' for x in both) if both else 'none listed'
    one_txt = ', '.join(f'{x:03d}' for x in one) if one else 'none listed'

    st.write(f'BB86 holes covered by SITELLE and PHAST/PHAT: {both_txt}')
    st.write(f'BB86 holes covered by only one of SITELLE or PHAST/PHAT: {one_txt}')


def render_sitelle_footprint_panel() -> None:
    foot_dir = APP_DIR / 'figures' / 'final' / 'multitracer' / 'sitelle_footprints'
    options = [
        ('SITELLE SN3 — BB86 + new candidates', foot_dir / 'sitelle_sn3_footprints_bb86_vs_new247_v1.png'),
        ('SITELLE SN3 — BB86', foot_dir / 'sitelle_sn3_footprints_bb86_v1.png'),
        ('SITELLE SN3 — new candidates', foot_dir / 'sitelle_sn3_footprints_new247_v1.png'),
    ]
    available = [(label, path) for label, path in options if path.exists()]
    if not available:
        st.caption('No SITELLE footprint PNG found in figures/final/multitracer/sitelle_footprints/.')
        return

    st.markdown('### SITELLE SN3 coverage on M31')
    labels = [x[0] for x in available]
    chosen = st.selectbox('SITELLE footprint map', labels, index=0, key='sitelle_footprint_choice')
    path = dict(available)[chosen]
    st.image(str(path), width='stretch')

    pdf_path = path.with_suffix('.pdf')
    if pdf_path.exists():
        with pdf_path.open('rb') as fh:
            st.download_button(
                'Download SITELLE coverage PDF',
                data=fh.read(),
                file_name=pdf_path.name,
                mime='application/pdf',
                key=f'download_{pdf_path.name}',
            )

def build_bb86_metadata_table(row: pd.Series) -> pd.DataFrame:
    specs = [
        ('Tracer', 'tracer'),
        ('Catalogue', 'source_catalog'),
        ('central velocity HRV [km/s]', 'HRV'),
        ('major diameter [pc]', 'Maj'),
        ('minor diameter [pc]', 'Min'),
        ('PA [deg]', 'PA'),
        ('offset East/West from M31 centre [arcmin]', 'x_arcmin'),
        ('offset North/South from M31 centre [arcmin]', 'y_arcmin'),
        ('video file', 'video_name'),
    ]
    rows = []
    for label, col in specs:
        if col not in row.index:
            continue
        val = value_to_text(row[col])
        if col == 'video_name':
            val = clean_video_label(val)
        if val:
            rows.append({'parameter': label, 'value': val})
    geom = selected_hi_geometry(row)
    rows.append({'parameter': 'context radius R [arcmin]', 'value': f'{geom["r_eq"]:.3g}'})
    rows.append({'parameter': 'context radius R [pc]', 'value': f'{geom["r_eq"] * PC_PER_ARCMIN:.3g}'})
    return pd.DataFrame(rows)


def selected_object_image_path(row: pd.Series, source_type: str) -> Path | None:
    if source_type == 'H I BB86':
        for col in ['summary_2dcg_png', 'contrast_joint_refit_png']:
            path = resolve_optional_path(row, col)
            if path is not None and path.exists():
                return path
        return None
    for col in ['v11_diagnostic_png', 'validation_png']:
        path = resolve_catalog_path(row, col)
        if path is not None and path.exists():
            return path
    return None


def selected_object_title(row: pd.Series, source_type: str) -> str:
    if source_type == 'H I BB86':
        try:
            return f'BB86 H I hole {int(row["Seq"]):03d}'
        except Exception:
            return 'BB86 H I hole'
    return str(row.get('display_label', row.get('object_uid', source_type)))



def resolve_spitzer_mips_panel_path(row: pd.Series) -> Path | None:
    panel_dir = APP_DIR / 'figures/final/multitracer/IR/Spitzer_MIPS/object_panels'
    tracer = str(row.get('tracer', '')).strip()

    if tracer == 'HI':
        seq = numeric(row, 'hi_seq')
        if np.isfinite(seq):
            return panel_dir / f'BB86_{int(seq):03d}_spitzer_mips_panel.png'

    if tracer == 'Koch25 new HI candidate':
        cid = numeric(row, 'new_candidate_id')
        if not np.isfinite(cid):
            cid = numeric(row, 'candidate_id')
        if np.isfinite(cid):
            return panel_dir / f'NEW247_v11_best_{int(cid):04d}_spitzer_mips_panel.png'

    if tracer == 'Potential new HI Candidate':
        cid = numeric(row, 'new_candidate_id')
        if not np.isfinite(cid):
            cid = numeric(row, 'candidate_id')
        if np.isfinite(cid):
            return panel_dir / f'POTENTIAL_NEW247_{int(cid):04d}_spitzer_mips_panel.png'

    uid = str(row.get('object_uid', '')).strip()
    if uid:
        candidates = sorted(panel_dir.glob(f'{uid}*_spitzer_mips_panel.png'))
        if candidates:
            return candidates[0]

    return None


def render_spitzer_mips_panel(row: pd.Series) -> None:
    path = resolve_spitzer_mips_panel_path(row)
    st.markdown('### Spitzer/MIPS IR')
    if path is not None and path.exists():
        st.image(str(path), width='stretch')
    else:
        st.info('No Spitzer/MIPS panel found for this selected object.')

def render_selected_video(row: pd.Series, source_type: str) -> None:
    st.markdown('### PPV video')

    if source_type == 'H I BB86':
        video_url = resolve_video_url(row)
        video_path = resolve_video_path(row)
        label = clean_video_label(video_url or (video_path.name if video_path else ''))
        if label:
            st.caption(f'Video: `{label}`')
        if video_url:
            st.success('Video served from GitHub Release asset.')
            video_player_from_url(video_url)
        elif video_path is not None and video_path.exists():
            video_player_from_file(video_path)
        else:
            st.warning('No video associated with this selected object.')
        return

    candidate_video_url = ''
    for video_col in ['video_url', 'video_release_url', 'video_path']:
        if video_col in row.index:
            raw = str(row.get(video_col, '')).strip()
            if raw and raw.lower() not in {'nan', 'none'}:
                candidate_video_url = raw
                break

    if not candidate_video_url:
        cand_id = None
        for id_col in ['new_candidate_id', 'candidate_id']:
            if id_col in row.index:
                try:
                    cand_id = int(float(row.get(id_col)))
                    break
                except Exception:
                    pass
        if cand_id is not None:
            repo = 'Croycosmos/m31-hi-holes-viewer'
            tag = 'ppv-new-candidates-v1'
            tracer_txt = str(row.get('tracer', source_type)).lower()
            if 'potential' in tracer_txt:
                fname = f'potential_candidate_{cand_id:04d}_ppv_flythrough_dark_web.mp4'
            else:
                fname = f'new_candidate_{cand_id:04d}_ppv_flythrough_dark_web.mp4'
            candidate_video_url = f'https://github.com/{repo}/releases/download/{tag}/{fname}'

    label = clean_video_label(candidate_video_url)
    if label:
        st.caption(f'Video: `{label}`')

    if candidate_video_url.startswith('http://') or candidate_video_url.startswith('https://'):
        st.success('Video served from GitHub Release asset.')
        video_player_from_url(candidate_video_url)
    elif candidate_video_url:
        candidate_video_path = Path(candidate_video_url)
        if candidate_video_path.is_absolute():
            candidate_video_path = Path('videos') / candidate_video_path.name
        candidate_video_path = APP_DIR / candidate_video_path
        if candidate_video_path.exists():
            video_player_from_file(candidate_video_path)
        else:
            st.error(f'Candidate video path is set but file is missing: {candidate_video_path}')
    else:
        st.warning('No video associated with this selected object.')


def render_context_tables_and_zoom(row: pd.Series, objects_df: pd.DataFrame, selected_tracers: list[str], show_search_rings: bool, key_suffix: str) -> None:
    st.markdown('### Nearby objects within 3R')
    context_summary, near_table = nearest_context_tables(row, objects_df, max_radius_factor=3.0)

    if context_summary.empty:
        st.info('No nearby-object table could be computed for this selected object.')
    else:
        st.dataframe(context_summary, width='stretch', hide_index=True)
        with st.expander('Nearby objects within 3R', expanded=False):
            if near_table.empty:
                st.info('No contextual object within 3R.')
            else:
                st.dataframe(near_table, width='stretch', hide_index=True)

    local_fig = build_local_context_figure(
        row, objects_df, BACKGROUND_PNG, meta,
        show_search_rings=show_search_rings, selected_tracers=selected_tracers,
    )
    if local_fig is not None:
        st.markdown(f'### Local 3R zoom — {selected_object_title(row, str(row.get("tracer", "")))}')
        st.plotly_chart(
            local_fig,
            key=f'local_3r_zoom_{key_suffix}',
            config={'responsive': True},
            width='stretch',
        )
    else:
        st.info('Local 3R zoom could not be generated for this selected object.')

def render_object_browser(hi_df: pd.DataFrame, candidates_df: pd.DataFrame) -> None:
    st.subheader('Object browser')

    category_options = ['BB86', 'New candidates', 'Potential new candidates']
    if 'browser_category' not in st.session_state:
        st.session_state['browser_category'] = 'BB86'
    if 'browser_index' not in st.session_state:
        st.session_state['browser_index'] = 0

    category = st.selectbox(
        'Object family',
        category_options,
        index=category_options.index(st.session_state['browser_category']) if st.session_state['browser_category'] in category_options else 0,
        key='browser_category_select',
    )
    if category != st.session_state.get('browser_category'):
        st.session_state['browser_category'] = category
        st.session_state['browser_index'] = 0

    if category == 'BB86':
        df = hi_df.copy()
        df['_browser_label'] = df['Seq'].map(lambda x: f'BB86 H I {int(x):03d}')
        source_type = 'H I BB86'
    elif category == 'New candidates':
        df = candidates_df.loc[candidates_df['tracer'] == 'Koch25 new HI candidate'].copy()
        df['_browser_label'] = df['display_label'].astype(str)
        source_type = 'Koch25 new HI candidate'
    else:
        df = candidates_df.loc[candidates_df['tracer'] == 'Potential new HI Candidate'].copy()
        df['_browser_label'] = df['display_label'].astype(str)
        source_type = 'Potential new HI Candidate'

    if df.empty:
        st.warning('No object in this category.')
        return

    n = len(df)
    st.session_state['browser_index'] = int(st.session_state.get('browser_index', 0)) % n

    left_img, right_info = st.columns([1.35, 1.0])
    row = df.iloc[st.session_state['browser_index']]

    with left_img:
        st.markdown(f'### {selected_object_title(row, source_type)}')
        img = selected_object_image_path(row, source_type)
        if img is not None:
            st.image(str(img), width='stretch')
        else:
            st.info('No PNG found for this selected object.')

    with right_info:
        st.markdown('### 2DCG characteristics')
        if source_type == 'H I BB86':
            table = build_refit_table(row)
            if table.empty:
                table = build_bb86_metadata_table(row)
        else:
            table = build_candidate_table(row)
        st.dataframe(table, width='stretch', hide_index=True)

    nav_left, nav_mid, nav_right = st.columns([0.18, 0.64, 0.18])
    with nav_left:
        if st.button('← Previous', width='stretch'):
            st.session_state['browser_index'] = (st.session_state['browser_index'] - 1) % n
            st.rerun()
    with nav_mid:
        labels = df['_browser_label'].tolist()
        current_label = labels[st.session_state['browser_index']]
        chosen_label = st.selectbox(
            'Selected object',
            labels,
            index=labels.index(current_label),
            key=f'browser_object_select_{category}',
        )
        new_index = labels.index(chosen_label)
        if new_index != st.session_state['browser_index']:
            st.session_state['browser_index'] = new_index
            st.rerun()
    with nav_right:
        if st.button('Next →', width='stretch'):
            st.session_state['browser_index'] = (st.session_state['browser_index'] + 1) % n
            st.rerun()



st.title('M31 multi-tracer structures viewer')
st.caption(
    'H I objects are cavities/holes. Koch25 new H I candidates and potential candidates come from the blind 2DCG pipeline. UV, Hα, CO, Dust-HELGA and other layers are contextual tracers.'
)
st.write(
    'First step : refit the characteristics of the known holes of HI of Brinks and Bajaja of 1986 (BB86) in the new dataset of Koch of 2025. '
    'Then, find a new way of study and detect holes (2D Cumulative Growth). '
    'Finally, launch a blind detection in the data of Koch with 2DCG and Dassa-Terrier core seeds (2022).'
)

if not HI_CATALOG_PATH.exists():
    st.error(f'Missing catalogue: {HI_CATALOG_PATH}')
    st.stop()

hi_df = load_hi_catalog(HI_CATALOG_PATH)
objects_df = load_objects_catalog(OBJECTS_CATALOG_PATH, hi_df)
candidates_df = load_new_candidates_catalog(NEW_CANDIDATES_CATALOG_PATH)
objects_df = objects_df.loc[objects_df['x_plot'].notna() & objects_df['y_plot'].notna()].copy()
meta = load_background_meta(BACKGROUND_META)

valid_hi = set(hi_df['Seq'].astype(int))
if 'selected_source_type' not in st.session_state:
    st.session_state['selected_source_type'] = 'H I BB86'
if 'selected_hi_seq' not in st.session_state or int(st.session_state.get('selected_hi_seq', -1)) not in valid_hi:
    st.session_state['selected_hi_seq'] = int(hi_df['Seq'].iloc[0])
if 'selected_candidate_uid' not in st.session_state:
    st.session_state['selected_candidate_uid'] = ''
if 'clicked_object_uid' not in st.session_state:
    st.session_state['clicked_object_uid'] = ''

tab_detection, tab_multi, tab_stats, tab_browser = st.tabs([
    'Détection / validation H I',
    'Multi-traceurs',
    'Statistiques globales',
    'Object browser',
])

selected_hi_row = None
selected_candidate_row = None
selected_object_row = None
selected_source_type = st.session_state.get('selected_source_type', 'H I BB86')

with tab_detection:
    st.subheader('Object selection')

    c0, c1, c2, c3 = st.columns([1.1, 1.0, 1.0, 1.0])
    with c0:
        source_options = ['H I BB86', 'Koch25 new HI candidate', 'Potential new HI Candidate']
        current_source = st.session_state.get('selected_source_type', 'H I BB86')
        source_index = source_options.index(current_source) if current_source in source_options else 0
        selected_source_type = st.selectbox('Selected object type', source_options, index=source_index)
        st.session_state['selected_source_type'] = selected_source_type

    with c1:
        st.metric('BB86 H I holes', len(hi_df))
    with c2:
        n_clean = int((candidates_df['tracer'] == 'Koch25 new HI candidate').sum()) if not candidates_df.empty else 0
        st.metric('New H I cavities', n_clean)
    with c3:
        n_potential = int((candidates_df['tracer'] == 'Potential new HI Candidate').sum()) if not candidates_df.empty else 0
        st.metric('Potential H I cavities', n_potential)

    if selected_source_type == 'H I BB86':
        seq_options = hi_df['Seq_str'].tolist()
        selected_seq_str = f'{int(st.session_state["selected_hi_seq"]):03d}'
        index = seq_options.index(selected_seq_str) if selected_seq_str in seq_options else 0
        selected = st.selectbox('BB86 H I hole', seq_options, index=index)
        st.session_state['selected_hi_seq'] = int(selected)
        selected_hi_seq = int(st.session_state['selected_hi_seq'])
        selected_hi_row = hi_df.loc[hi_df['Seq'] == selected_hi_seq].iloc[0]
        selected_object_row = selected_hi_row
    else:
        sub = candidates_df.loc[candidates_df['tracer'] == selected_source_type].copy() if not candidates_df.empty else pd.DataFrame()
        sub = sort_candidate_selector_rows(sub)
        if sub.empty:
            st.warning(f'No rows for {selected_source_type}.')
        else:
            uid_options = sub['object_uid'].astype(str).tolist()
            label_map = dict(zip(sub['object_uid'].astype(str), sub['display_label'].astype(str)))
            current_uid = str(st.session_state.get('selected_candidate_uid', ''))
            index = uid_options.index(current_uid) if current_uid in uid_options else 0
            selected_uid = st.selectbox(
                selected_source_type,
                uid_options,
                index=index,
                format_func=lambda uid: label_map.get(uid, uid),
            )
            selected_candidate_row = sub.loc[sub['object_uid'].astype(str) == str(selected_uid)].iloc[0]
            st.session_state['selected_candidate_uid'] = str(selected_candidate_row['object_uid'])
            selected_object_row = selected_candidate_row

    left, right = st.columns([1.45, 1.0])

    with left:
        st.subheader('M31 H I cavity map')
        simple_tracers = ['HI', 'Koch25 new HI candidate', 'Potential new HI Candidate']
        df_map = objects_df.loc[objects_df['tracer'].isin(simple_tracers)].copy()
        counts = df_map.groupby('tracer').size().to_dict() if not df_map.empty else {}
        counts_text = ' | '.join([f'{k}: {v}' for k, v in counts.items()]) if counts else 'aucun objet'
        st.caption(f'Objets affichés : {len(df_map)} — {counts_text}')

        if not BACKGROUND_PNG.exists():
            st.warning('Missing background image: data/m31_background.png')

        fig = build_m31_figure(
            df_visible=df_map,
            df_extent=objects_df,
            background_png=BACKGROUND_PNG,
            meta=meta,
            selected_hi_row=selected_object_row,
            show_search_rings=False,
        )
        fig.update_layout(title='M31 — H I cavities and candidates')
        event = st.plotly_chart(fig, key='m31_detection_map_click', on_select='rerun', selection_mode='points', config={'responsive': True}, width='stretch')

        clicked = get_clicked_object(event)
        if clicked:
            st.session_state['clicked_object_uid'] = clicked['object_uid']
            if clicked['tracer'] == 'HI' and str(clicked['hi_seq']).strip() not in {'', 'nan', 'None'}:
                try:
                    new_seq = int(float(clicked['hi_seq']))
                    st.session_state['selected_source_type'] = 'H I BB86'
                    st.session_state['selected_hi_seq'] = new_seq
                    st.rerun()
                except Exception:
                    pass
            elif clicked['tracer'] in {'Koch25 new HI candidate', 'Potential new HI Candidate'}:
                st.session_state['selected_source_type'] = clicked['tracer']
                st.session_state['selected_candidate_uid'] = clicked['object_uid']
                st.rerun()

        if selected_source_type == 'H I BB86' and selected_hi_row is not None:
            render_optional_png(selected_hi_row, 'contrast_joint_refit', 'contrast_joint_refit_png', 'No contrast_joint_refit PNG found for this H I hole.')
            render_optional_png(selected_hi_row, '2DCG summary', 'summary_2dcg_png', 'No 2DCG summary PNG found for this H I hole.')
        elif selected_candidate_row is not None:
            png_path = resolve_catalog_path(selected_candidate_row, 'validation_png')
            if png_path is not None and png_path.exists():
                st.markdown('### 2DCG validation')
                st.image(str(png_path), width='stretch')

            st.markdown('### 2DCG refit')
            v11_png_path = resolve_catalog_path(selected_candidate_row, 'v11_diagnostic_png')
            if v11_png_path is not None and v11_png_path.exists():
                st.image(str(v11_png_path), width='stretch')
            else:
                st.caption('No 2DCG refit PNG available for this selected candidate.')

    with right:
        if selected_source_type == 'H I BB86' and selected_hi_row is not None:
            selected_hi_seq = int(selected_hi_row['Seq'])
            st.markdown(f'### Selected BB86 H I hole {selected_hi_seq:03d}')
            st.dataframe(build_bb86_metadata_table(selected_hi_row), width='stretch', hide_index=True)

            no_2dcg_txt = ', '.join(f'{seq:03d}' for seq in sorted(NO_2DCG_BB86))
            if selected_hi_seq in NO_2DCG_BB86:
                st.warning(f'This BB86 hole is in the no-2DCG list: {no_2dcg_txt}.')
            else:
                st.info(f'BB86 holes without 2DCG figure: {no_2dcg_txt}.')

            refit_table = build_refit_table(selected_hi_row)
            if not refit_table.empty:
                st.markdown('### Refit parameters')
                st.dataframe(refit_table, width='stretch', hide_index=True)

            render_selected_video(selected_hi_row, selected_source_type)

        elif selected_candidate_row is not None:
            st.markdown(f'### Selected {selected_source_type}')
            geom = selected_hi_geometry(selected_candidate_row)
            cand_meta = build_candidate_table(selected_candidate_row)
            extra = pd.DataFrame([
                {'parameter': 'context radius R [arcmin]', 'value': f'{geom["r_eq"]:.3g}'},
                {'parameter': 'context radius R [pc]', 'value': f'{geom["r_eq"] * PC_PER_ARCMIN:.3g}'},
            ])
            st.dataframe(pd.concat([cand_meta, extra], ignore_index=True), width='stretch', hide_index=True)
            render_selected_video(selected_candidate_row, selected_source_type)

with tab_multi:
    selected_source_type, selected_hi_row, selected_candidate_row, selected_row_for_context = render_multitracer_object_selector(hi_df, candidates_df)

    render_coverage_overview()
    render_sitelle_footprint_panel()

    st.subheader('M31 multi-tracer map')

    selected_row_for_context = selected_hi_row if selected_source_type == 'H I BB86' else selected_candidate_row

    ctrl1, ctrl2 = st.columns([1.0, 1.0])
    with ctrl1:
        available_tracers = [t for t in TRACER_OPTIONS if t in set(objects_df['tracer'].astype(str))]
        default_tracers = [t for t in TRACER_OPTIONS if t in available_tracers]
        selected_tracers_multi = st.multiselect(
            'Traceurs affichés',
            TRACER_OPTIONS,
            default=default_tracers,
            help='H I = cavités. Les autres traceurs servent au contexte physique.',
        )
        show_search_rings_multi = st.checkbox('Afficher les anneaux 1R/2R/3R de l’objet H I sélectionné', value=True)
    with ctrl2:
        possible_sources = sorted(objects_df.loc[objects_df['tracer'].isin(selected_tracers_multi), 'source_catalog'].dropna().astype(str).unique()) if selected_tracers_multi else []
        selected_sources_multi = st.multiselect('Catalogues affichés', possible_sources, default=possible_sources)

    df_multi = objects_df.loc[objects_df['tracer'].isin(selected_tracers_multi)].copy() if selected_tracers_multi else objects_df.iloc[0:0].copy()
    if selected_sources_multi:
        df_multi = df_multi.loc[df_multi['source_catalog'].isin(selected_sources_multi)].copy()

    counts = df_multi.groupby('tracer').size().to_dict() if not df_multi.empty else {}
    counts_text = ' | '.join([f'{k}: {v}' for k, v in counts.items()]) if counts else 'aucun objet'
    st.caption(f'Objets affichés : {len(df_multi)} / {len(objects_df)} — {counts_text}')

    fig_multi = build_m31_figure(
        df_visible=df_multi,
        df_extent=objects_df,
        background_png=BACKGROUND_PNG,
        meta=meta,
        selected_hi_row=selected_row_for_context,
        show_search_rings=show_search_rings_multi,
    )
    st.plotly_chart(fig_multi, key='m31_multitracer_map', config={'responsive': True}, width='stretch')

    render_multitracer_panel(selected_source_type, selected_hi_row, selected_candidate_row)

    if selected_row_for_context is not None:
        render_context_tables_and_zoom(
            selected_row_for_context,
            objects_df,
            selected_tracers_multi,
            show_search_rings_multi,
            key_suffix='multi',
        )
        render_spitzer_mips_panel(selected_row_for_context)

with tab_stats:
    stats_figures_tab, stats_compare_tab = st.tabs([
        'Figures de population',
        'BB86 vs 247 candidats',
    ])

    with stats_figures_tab:
        render_global_statistics()

    with stats_compare_tab:
        render_bb86_vs_new247_statistics(hi_df, candidates_df)

with tab_browser:
    render_object_browser(hi_df, candidates_df)
