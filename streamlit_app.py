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
        'title': 'Nouveaux 72 candidats - distributions des paramètres',
        'caption': 'Histogrammes globaux pour les 72 candidats H I retenus par le filtre v9.',
        'filename': 'new_hi_candidates_v9_histograms_main.pdf',
    },
    {
        'title': 'Potentiels 175 candidats - distributions des paramètres',
        'caption': 'Histogrammes globaux pour les 175 candidats potentiels / rejetés par le filtre v9.',
        'filename': 'v9_rejected_175_histograms_main.pdf',
    },
    {
        'title': 'Potentiels 175 candidats - raisons de rejet',
        'caption': 'Décompte des raisons de rejet du filtre v9, séparées par famille de critère.',
        'filename': 'v9_rejected_reason_counts_split.pdf',
    },
]
DISPLAY_FLIP_X = True

ARCSEC_PER_PC = 206265.0 / 690000.0
PC_PER_ARCMIN = 60.0 / ARCSEC_PER_PC

TRACER_OPTIONS = ['HI', 'Koch25 new HI candidate', 'Potential new HI Candidate', 'UV', 'Hα', 'CO', 'IR', 'X-ray']
TRACER_COLORS = {
    'HI': 'deepskyblue',
    'Koch25 new HI candidate': 'lime',
    'Potential new HI Candidate': 'gray',
    'UV': 'violet',
    'Hα': 'red',
    'CO': 'orange',
    'IR': 'gold',
    'X-ray': 'lime',
}
TRACER_SYMBOLS = {
    'HI': 'circle',
    'Koch25 new HI candidate': 'cross',
    'Potential new HI Candidate': 'x',
    'UV': 'square',
    'Hα': 'triangle-up',
    'CO': 'diamond',
    'IR': 'hexagon',
    'X-ray': 'star',
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


def render_candidate_summary_pdfs() -> None:
    st.markdown('### Candidate summary documents')
    st.caption('Global diagnostic PDF panels for the v9 new-candidate catalogues.')
    for i, item in enumerate(CANDIDATE_SUMMARY_PDFS):
        path = CANDIDATE_SUMMARY_DIR / item['filename']
        with st.expander(item['title'], expanded=(i == 0)):
            st.caption(item['caption'])
            if path.exists():
                size_mb = path.stat().st_size / 1024**2
                st.write(f'File: `{path.relative_to(APP_DIR)}` - `{size_mb:.1f} MB`')
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
    context_tracers = ['UV', 'Hα', 'CO', 'IR', 'X-ray']
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
                marker_symbol = 'cross-open' if tracer == 'Koch25 new HI candidate' else 'x-open'

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
                x=[-cx if DISPLAY_FLIP_X else cx], y=[cy], mode='markers+text', name='selected object',
                marker=dict(size=21, color=marker_color, symbol=marker_symbol, line=dict(width=3.0, color=marker_color)),
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
    context_tracers = ['UV', 'Hα', 'CO', 'IR', 'X-ray']
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
    for col in ['object_uid', 'tracer', 'source_catalog', 'display_label', 'validation_png']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)
    for col in ['candidate_id', 'ra_deg', 'dec_deg', 'x_arcmin', 'y_arcmin', 'major_arcsec', 'minor_arcsec', 'pa_deg', 'v_center_kms', 'Maj_pc', 'Min_pc', 'PA_astro_deg', 'v9_final_score']:
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
        ('catalog status', 'tracer'), ('candidate ID', 'candidate_id'), ('source catalog', 'source_catalog'),
        ('RA [deg]', 'ra_deg'), ('Dec [deg]', 'dec_deg'), ('X [arcmin]', 'x_arcmin'), ('Y [arcmin]', 'y_arcmin'),
        ('v center [km/s]', 'v_center_kms'), ('Maj [pc]', 'Maj_pc'), ('Min [pc]', 'Min_pc'), ('PA [deg]', 'PA_astro_deg'),
        ('major [arcsec]', 'major_arcsec'), ('minor [arcsec]', 'minor_arcsec'), ('v9 score', 'v9_final_score'),
        ('reject reason', 'v9_reject_reason'), ('notes', 'notes'),
    ]
    rows = []
    for label, col in cols:
        if col in row.index:
            val = value_to_text(row[col])
            if val:
                rows.append({'parameter': label, 'value': val})
    return pd.DataFrame(rows)


st.title('M31 multi-tracer structures viewer')
st.caption(
    'H I objects are cavities/holes. Koch25 new H I candidates and potential candidates come from the blind 2DCG/v9 pipeline. UV, Hα and CO layers remain contextual tracers.'
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
    first_with_video = hi_df.loc[hi_df['has_video'], 'Seq']
    st.session_state['selected_hi_seq'] = int(first_with_video.iloc[0]) if not first_with_video.empty else int(hi_df['Seq'].iloc[0])
if 'selected_candidate_uid' not in st.session_state:
    st.session_state['selected_candidate_uid'] = ''
if 'clicked_object_uid' not in st.session_state:
    st.session_state['clicked_object_uid'] = ''

left, right = st.columns([1.45, 1.0])

with right:
    st.subheader('Object selection')
    n_hi = len(hi_df)
    n_video = int(hi_df['has_video'].sum())
    n_clean = int((candidates_df['tracer'] == 'Koch25 new HI candidate').sum()) if not candidates_df.empty else 0
    n_rej = int((candidates_df['tracer'] == 'Potential new HI Candidate').sum()) if not candidates_df.empty else 0
    st.write(f'BB86 H I videos found: `{n_video}/{n_hi}`')
    st.write(f'Koch25 new H I candidates: `{n_clean}`')
    st.write(f'Potential new H I candidates: `{n_rej}`')

    with st.expander('Global candidate summary PDFs', expanded=False):
        render_candidate_summary_pdfs()

    source_options = ['H I BB86', 'Koch25 new HI candidate', 'Potential new HI Candidate']
    current_source = st.session_state.get('selected_source_type', 'H I BB86')
    source_index = source_options.index(current_source) if current_source in source_options else 0
    selected_source_type = st.selectbox('Selected object type', source_options, index=source_index)
    st.session_state['selected_source_type'] = selected_source_type

    selected_hi_row = None
    selected_candidate_row = None
    selected_object_row = None
    if selected_source_type == 'H I BB86':
        show_only_with_video = st.checkbox('Show only BB86 H I holes with video in selector', value=True)
        hi_select = hi_df.loc[hi_df['has_video']].copy() if show_only_with_video else hi_df.copy()
        seq_options = hi_select['Seq_str'].tolist()
        selected_seq_str = f'{int(st.session_state["selected_hi_seq"]):03d}'
        index = seq_options.index(selected_seq_str) if selected_seq_str in seq_options else 0
        selected = st.selectbox('BB86 H I hole', seq_options, index=index)
        st.session_state['selected_hi_seq'] = int(selected)
        selected_hi_seq = int(st.session_state['selected_hi_seq'])
        selected_hi_row = hi_df.loc[hi_df['Seq'] == selected_hi_seq].iloc[0]
        selected_object_row = selected_hi_row
    else:
        sub = candidates_df.loc[candidates_df['tracer'] == selected_source_type].copy() if not candidates_df.empty else pd.DataFrame()
        if sub.empty:
            st.warning(f'No rows for {selected_source_type}. Run the V4 export script first.')
        else:
            labels = sub['display_label'].fillna(sub['object_uid']).astype(str).tolist()
            uids = sub['object_uid'].astype(str).tolist()
            current_uid = str(st.session_state.get('selected_candidate_uid', ''))
            index = uids.index(current_uid) if current_uid in uids else 0
            label = st.selectbox(selected_source_type, labels, index=index)
            selected_candidate_row = sub.loc[sub['display_label'].astype(str) == str(label)].iloc[0]
            st.session_state['selected_candidate_uid'] = str(selected_candidate_row['object_uid'])
            selected_object_row = selected_candidate_row

with left:
    st.subheader('M31 map')
    ctrl1, ctrl2 = st.columns([1.0, 1.0])
    with ctrl1:
        available_tracers = [t for t in TRACER_OPTIONS if t in set(objects_df['tracer'].astype(str))]
        default_tracers = [t for t in ['HI', 'UV', 'Hα', 'CO'] if t in available_tracers]
        selected_tracers = st.multiselect(
            'Traceurs affichés', TRACER_OPTIONS, default=default_tracers,
            help='H I = cavités. UV/Hα/CO = contexte physique, pas cavités.'
        )
        show_search_rings = st.checkbox('Afficher les anneaux 1R/2R/3R de l’objet H I sélectionné', value=True)
    with ctrl2:
        possible_sources = sorted(objects_df.loc[objects_df['tracer'].isin(selected_tracers), 'source_catalog'].dropna().astype(str).unique())
        selected_sources = st.multiselect('Catalogues affichés', possible_sources, default=possible_sources)

    df_map = objects_df.loc[objects_df['tracer'].isin(selected_tracers)].copy() if selected_tracers else objects_df.iloc[0:0].copy()
    if selected_sources:
        df_map = df_map.loc[df_map['source_catalog'].isin(selected_sources)].copy()

    missing_tracers = [t for t in selected_tracers if t not in set(objects_df['tracer'].astype(str).unique())]
    if missing_tracers:
        st.caption('Couches non encore peuplées : ' + ', '.join(missing_tracers))

    counts = df_map.groupby('tracer').size().to_dict() if not df_map.empty else {}
    counts_text = ' | '.join([f'{k}: {v}' for k, v in counts.items()]) if counts else 'aucun objet'
    st.caption(f'Objets affichés : {len(df_map)} / {len(objects_df)} — {counts_text}')

    if not BACKGROUND_PNG.exists():
        st.warning('Missing background image: data/m31_background.png')
    fig = build_m31_figure(df_visible=df_map, df_extent=objects_df, background_png=BACKGROUND_PNG, meta=meta, selected_hi_row=selected_object_row, show_search_rings=show_search_rings)
    event = st.plotly_chart(fig, key='m31_map_click', on_select='rerun', selection_mode='points', config={'responsive': True}, width='stretch')

    clicked = get_clicked_object(event)
    if clicked:
        st.session_state['clicked_object_uid'] = clicked['object_uid']
        if clicked['tracer'] == 'HI' and str(clicked['hi_seq']).strip() not in {'', 'nan', 'None'}:
            try:
                new_seq = int(float(clicked['hi_seq']))
                st.session_state['selected_source_type'] = 'H I BB86'
                if new_seq != int(st.session_state['selected_hi_seq']):
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
        st.markdown('### 2DCG/v9 validation PNG')
        png_path = resolve_catalog_path(selected_candidate_row, 'validation_png')
        if png_path is not None and png_path.exists():
            st.image(str(png_path), width='stretch')
        else:
            st.warning('No validation PNG found for this candidate.')

with right:
    if selected_source_type == 'H I BB86' and selected_hi_row is not None:
        selected_hi_seq = int(selected_hi_row['Seq'])
        st.markdown(f'### Selected BB86 H I hole {selected_hi_seq:03d}')
        geom = selected_hi_geometry(selected_hi_row)
        meta_cols = [c for c in ['tracer', 'source_catalog', 'HRV', 'Maj', 'Min', 'PA', 'x_arcmin', 'y_arcmin', 'has_video', 'video_name'] if c in hi_df.columns]
        meta_rows = [{'parameter': col, 'value': value_to_text(selected_hi_row[col])} for col in meta_cols]
        meta_rows.append({'parameter': 'context radius R [arcmin]', 'value': f'{geom["r_eq"]:.3g}'})
        meta_rows.append({'parameter': 'context radius R [pc]', 'value': f'{geom["r_eq"] * PC_PER_ARCMIN:.3g}'})
        st.dataframe(pd.DataFrame(meta_rows), width='stretch', hide_index=True)

        st.markdown('### Multi-tracer context around selected H I hole')
        context_summary, near_table = nearest_context_tables(selected_hi_row, objects_df, max_radius_factor=3.0)
        if context_summary.empty:
            st.info('No context table could be computed for this H I hole.')
        else:
            st.dataframe(context_summary, width='stretch', hide_index=True)
            with st.expander('Nearby objects within 3R'):
                if near_table.empty:
                    st.info('No UV/Hα/CO object within 3R.')
                else:
                    st.dataframe(near_table, width='stretch', hide_index=True)

            local_fig = build_local_context_figure(
                selected_hi_row, objects_df, BACKGROUND_PNG, meta,
                show_search_rings=show_search_rings, selected_tracers=selected_tracers,
            )
            if local_fig is not None:
                st.markdown(f'### Local 3R zoom — {selected_hi_row.get("object_label", selected_hi_row.get("display_label", selected_hi_row.get("Seq_str", "selected object")))}')
                st.plotly_chart(
                    local_fig, key='local_context_map_hi',
                    config={'responsive': True}, width='stretch',
                )

        clicked_uid = str(st.session_state.get('clicked_object_uid', ''))
        if clicked_uid:
            clicked_rows = objects_df.loc[objects_df['object_uid'] == clicked_uid]
            if not clicked_rows.empty:
                obj_row = clicked_rows.iloc[0]
                st.markdown('### Last clicked map object')
                st.dataframe(build_external_object_table(obj_row), width='stretch', hide_index=True)

        st.markdown('### PPV video')
        video_url = resolve_video_url(selected_hi_row)
        video_path = resolve_video_path(selected_hi_row)

        if video_url:
            st.code(video_url, language='text')
            st.success('Video served from GitHub Release asset.')
            try:
                video_player_from_url(video_url)
            except Exception as exc:
                st.error(f'Video URL playback error: {exc}')
        elif video_path is None:
            st.warning('No video associated with this H I hole.')
        else:
            try:
                display_path = str(video_path.relative_to(APP_DIR))
            except Exception:
                display_path = str(video_path)
            st.code(display_path, language='text')
            if video_path.exists():
                size_mb = video_path.stat().st_size / 1024**2
                st.success(f'Local video file found. Size: {size_mb:.1f} MB')
                try:
                    video_player_from_file(video_path)
                except Exception as exc:
                    st.error(f'Video playback error: {exc}')
            else:
                st.error(f'Catalogue points to a missing video: {video_path}')
    elif selected_candidate_row is not None:
        st.markdown(f'### Selected {selected_source_type}')
        geom = selected_hi_geometry(selected_candidate_row)
        cand_meta = build_candidate_table(selected_candidate_row)
        extra = pd.DataFrame([
            {'parameter': 'context radius R [arcmin]', 'value': f'{geom["r_eq"]:.3g}'},
            {'parameter': 'context radius R [pc]', 'value': f'{geom["r_eq"] * PC_PER_ARCMIN:.3g}'},
        ])
        st.dataframe(pd.concat([cand_meta, extra], ignore_index=True), width='stretch', hide_index=True)

        st.markdown('### Multi-tracer context around selected H I candidate')
        context_summary, near_table = nearest_context_tables(selected_candidate_row, objects_df, max_radius_factor=3.0)
        if context_summary.empty:
            st.info('No context table could be computed for this candidate.')
        else:
            st.dataframe(context_summary, width='stretch', hide_index=True)
            with st.expander('Nearby objects within 3R'):
                if near_table.empty:
                    st.info('No UV/Hα/CO object within 3R.')
                else:
                    st.dataframe(near_table, width='stretch', hide_index=True)

            local_fig = build_local_context_figure(
                selected_candidate_row, objects_df, BACKGROUND_PNG, meta,
                show_search_rings=show_search_rings, selected_tracers=selected_tracers,
            )
            if local_fig is not None:
                st.markdown(f'### Local 3R zoom — {selected_candidate_row.get("display_label", selected_candidate_row.get("object_uid", "selected candidate"))}')
                st.plotly_chart(
                    local_fig, key='local_context_map_candidate',
                    config={'responsive': True}, width='stretch',
                )

        st.info('For these new-candidate objects the BB86 2DCG + joint-contrast refit table is intentionally hidden. The diagnostic PNG above is the relevant validation product for now.')

    st.markdown('### Catalogue status')
    st.write(f'H I holes: `{len(hi_df)}`')
    st.write(f'All map objects: `{len(objects_df)}`')
    st.write(f'Objects visible on map: `{len(df_map)}`')
    st.write(f'H I videos found: `{n_video}`')
    st.write(f'H I videos missing: `{n_hi - n_video}`')
