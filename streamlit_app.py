from __future__ import annotations

from pathlib import Path
import json
import base64

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / 'data'
CATALOG_PATH = DATA_DIR / 'holes_catalog_streamlit.csv'
BACKGROUND_PNG = DATA_DIR / 'm31_background.png'
BACKGROUND_META = DATA_DIR / 'm31_background_meta.json'
DISPLAY_FLIP_X = True

TRACER_OPTIONS = ['HI', 'CO', 'UV', 'Hα', 'IR', 'X-ray']
TRACER_COLORS = {
    'HI': 'deepskyblue',
    'CO': 'orange',
    'UV': 'violet',
    'Hα': 'red',
    'IR': 'gold',
    'X-ray': 'lime',
}
TRACER_SYMBOLS = {
    'HI': 'circle',
    'CO': 'diamond',
    'UV': 'square',
    'Hα': 'triangle-up',
    'IR': 'hexagon',
    'X-ray': 'star',
}

st.set_page_config(page_title='M31 multi-tracer cavity viewer', layout='wide')


def as_bool(value) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def value_to_text(value) -> str:
    if pd.isna(value):
        return ''
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)


@st.cache_data(show_spinner=False)
def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Seq'] = df['Seq'].astype(int)
    df['Seq_str'] = df['Seq'].map(lambda x: f'{x:03d}')

    if 'has_video' in df.columns:
        df['has_video'] = df['has_video'].map(as_bool)
    else:
        df['has_video'] = df['video_path'].astype(str).str.len() > 0

    if 'tracer' not in df.columns:
        df['tracer'] = 'HI'
    df['tracer'] = df['tracer'].fillna('HI').astype(str).str.strip().replace({'': 'HI'})

    if 'source_catalog' not in df.columns:
        df['source_catalog'] = 'BB86 + 2DCG/joint contrast'
    df['source_catalog'] = df['source_catalog'].fillna('').astype(str)

    if 'object_label' not in df.columns:
        df['object_label'] = df['tracer'].astype(str) + ' ' + df['Seq_str']
    df['object_label'] = df['object_label'].fillna(df['Seq_str']).astype(str)

    df['x_plot'] = -df['x_arcmin'] if DISPLAY_FLIP_X else df['x_arcmin']
    df['y_plot'] = df['y_arcmin']
    return df


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
    out = {
        'x_min': float(min(x0, x1)),
        'x_max': float(max(x0, x1)),
        'y_min': float(y_min),
        'y_max': float(y_max),
    }
    out['sizex'] = out['x_max'] - out['x_min']
    out['sizey'] = out['y_max'] - out['y_min']
    return out


def resolve_video_path(row: pd.Series) -> Path | None:
    raw = str(row.get('video_path', '')).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        path = Path('videos') / path.name
    return APP_DIR / path


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
        ('major fit amp S/N', None, 'cg2d__major_fit_amp_sn'),
        ('minor fit amp S/N', None, 'cg2d__minor_fit_amp_sn'),
        ('major r50 [arcsec]', None, 'cg2d__major_r50_deficit_arcsec'),
        ('major r80 [arcsec]', None, 'cg2d__major_r80_deficit_arcsec'),
        ('minor r50 [arcsec]', None, 'cg2d__minor_r50_deficit_arcsec'),
        ('minor r80 [arcsec]', None, 'cg2d__minor_r80_deficit_arcsec'),
        ('major growth score', None, 'cg2d__major_growth_score'),
        ('minor growth score', None, 'cg2d__minor_growth_score'),
        ('major combined score', None, 'cg2d__major_combined_score'),
        ('status / source', 'joint__status', 'cg2d__major_source'),
    ]
    rows = []
    for parameter, joint_col, cg2d_col in specs:
        joint_val = _row_value(row, joint_col)
        cg2d_val = _row_value(row, cg2d_col)
        if joint_val or cg2d_val:
            rows.append({
                'parameter': parameter,
                'joint_contrast_refit': joint_val,
                '2DCG': cg2d_val,
            })
    return pd.DataFrame(rows)


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


def add_background(fig: go.Figure, background_png: Path, meta: dict, df_extent: pd.DataFrame) -> None:
    has_bg = background_png.exists() and bool(meta)
    if has_bg:
        bg = transform_background_extent(meta)
        background_source = image_file_to_data_uri(background_png)
        fig.add_layout_image(
            dict(
                source=background_source,
                xref='x', yref='y',
                x=bg['x_min'], y=bg['y_max'],
                sizex=bg['sizex'], sizey=bg['sizey'],
                sizing='stretch', opacity=1.0, layer='below',
            )
        )
        fig.update_xaxes(range=[bg['x_min'], bg['x_max']])
        fig.update_yaxes(range=[bg['y_min'], bg['y_max']])
    else:
        xpad = 10.0
        ypad = 10.0
        if not df_extent.empty:
            fig.update_xaxes(range=[float(df_extent['x_plot'].min()) - xpad, float(df_extent['x_plot'].max()) + xpad])
            fig.update_yaxes(range=[float(df_extent['y_plot'].min()) - ypad, float(df_extent['y_plot'].max()) + ypad])
        else:
            fig.update_xaxes(range=[-120.0, 120.0])
            fig.update_yaxes(range=[-80.0, 80.0])


def build_m31_figure(df_visible: pd.DataFrame, df_extent: pd.DataFrame, background_png: Path, meta: dict, selected_seq: int | None) -> go.Figure:
    fig = go.Figure()
    add_background(fig, background_png, meta, df_extent)

    if not df_visible.empty:
        for tracer in sorted(df_visible['tracer'].dropna().astype(str).unique()):
            df_tracer = df_visible.loc[df_visible['tracer'].astype(str) == tracer].copy()
            for has_video, df_group in df_tracer.groupby('has_video', dropna=False):
                if df_group.empty:
                    continue
                color = TRACER_COLORS.get(tracer, 'white')
                base_symbol = TRACER_SYMBOLS.get(tracer, 'circle')
                symbol = base_symbol if bool(has_video) else f'{base_symbol}-open'
                name = f'{tracer} — with video' if bool(has_video) else f'{tracer} — without video'
                fig.add_trace(go.Scatter(
                    x=df_group['x_plot'],
                    y=df_group['y_plot'],
                    mode='markers',
                    name=name,
                    marker=dict(size=8, color=color, opacity=0.88, line=dict(width=0.8, color='white'), symbol=symbol),
                    text=df_group['object_label'],
                    customdata=df_group[['Seq', 'tracer', 'source_catalog', 'x_arcmin', 'y_arcmin', 'HRV', 'Maj', 'Min', 'PA']],
                    hovertemplate=(
                        '%{text}<br>'
                        'Tracer=%{customdata[1]}<br>'
                        'Source=%{customdata[2]}<br>'
                        'X catalog=%{customdata[3]:.1f} arcmin<br>'
                        'Y catalog=%{customdata[4]:.1f} arcmin<br>'
                        'HRV=%{customdata[5]:.1f} km/s<br>'
                        'Maj=%{customdata[6]:.0f} pc<br>'
                        'Min=%{customdata[7]:.0f} pc<br>'
                        'PA=%{customdata[8]:.1f} deg<br>'
                        '<extra></extra>'
                    ),
                ))

    if selected_seq is not None and not df_visible.empty:
        sel = df_visible.loc[df_visible['Seq'] == int(selected_seq)]
        if not sel.empty:
            r = sel.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r['x_plot']], y=[r['y_plot']], mode='markers+text', name='selected',
                marker=dict(size=18, color='red', symbol='circle-open', line=dict(width=3.0, color='red')),
                text=[f"{int(r['Seq']):03d}"], textposition='top center', textfont=dict(color='red', size=13), hoverinfo='skip'
            ))

    x_title = 'Displayed X [arcmin] — East left / West right' if DISPLAY_FLIP_X else 'X [arcmin]'
    fig.update_layout(
        title='M31 — multi-tracer cavity map', template='plotly_dark', height=820,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0.35)', bordercolor='rgba(255,255,255,0.25)', borderwidth=1),
        xaxis_title=x_title, yaxis_title='Y [arcmin] — North/South',
        clickmode='event+select',
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig


def nearest_seq_from_click(df: pd.DataFrame, event: dict) -> int | None:
    try:
        x = float(event.get('x'))
        y = float(event.get('y'))
    except Exception:
        return None
    if df.empty:
        return None
    dx = df['x_plot'].astype(float) - x
    dy = df['y_plot'].astype(float) - y
    dist2 = dx * dx + dy * dy
    if dist2.empty:
        return None
    i = int(dist2.idxmin())
    return int(df.loc[i, 'Seq'])


st.title('M31 multi-tracer cavity viewer')
st.caption('Interactive viewer for H I cavities in M31. The interface is ready for CO, UV, Hα, IR and X-ray cavity catalogues; only H I is populated in this version.')

if not CATALOG_PATH.exists():
    st.error(f'Missing catalogue: {CATALOG_PATH}')
    st.stop()

df = load_catalog(CATALOG_PATH)
needed = {'Seq', 'Seq_str', 'x_arcmin', 'y_arcmin', 'x_plot', 'y_plot', 'video_path', 'has_video', 'tracer'}
missing = needed - set(df.columns)
if missing:
    st.error(f'Missing columns in catalogue: {sorted(missing)}')
    st.stop()

valid_pos = df['x_plot'].notna() & df['y_plot'].notna()
df = df.loc[valid_pos].copy()
meta = load_background_meta(BACKGROUND_META)

if 'selected_seq' not in st.session_state or int(st.session_state['selected_seq']) not in set(df['Seq'].astype(int)):
    first_with_video = df.loc[df['has_video'], 'Seq']
    if not first_with_video.empty:
        st.session_state['selected_seq'] = int(first_with_video.iloc[0])
    else:
        st.session_state['selected_seq'] = int(df['Seq'].iloc[0])

left, right = st.columns([1.45, 1.0])

with right:
    st.subheader('Object selection')
    n_total = len(df)
    n_video = int(df['has_video'].sum())
    st.write(f'Videos found: `{n_video}/{n_total}`')
    show_only_with_video = st.checkbox('Show only H I holes with video in selector', value=True)
    df_select = df.copy()
    if show_only_with_video:
        df_select = df_select.loc[df_select['has_video']].copy()
    if df_select.empty:
        st.warning('No object with video found in the catalogue.')
        st.stop()
    seq_options = df_select['Seq_str'].tolist()
    selected_seq_str = f"{int(st.session_state['selected_seq']):03d}"
    index = seq_options.index(selected_seq_str) if selected_seq_str in seq_options else 0
    selected = st.selectbox('H I hole', seq_options, index=index)
    selected_seq = int(selected)
    st.session_state['selected_seq'] = selected_seq

with left:
    st.subheader('M31 map')

    ctrl1, ctrl2 = st.columns([1.15, 0.85])
    with ctrl1:
        selected_tracers = st.multiselect(
            'Traceurs affichés',
            TRACER_OPTIONS,
            default=['HI'],
            help='CO, UV, Hα, IR et X-ray sont préparés dans l’interface, mais les catalogues correspondants ne sont pas encore injectés dans cette V1.',
        )
    with ctrl2:
        visible_sources = sorted(df.loc[df['tracer'].isin(selected_tracers), 'source_catalog'].dropna().astype(str).unique())
        selected_sources = st.multiselect(
            'Catalogues affichés',
            visible_sources,
            default=visible_sources,
            help='Filtre par catalogue source. Pour cette V1, la couche remplie est le catalogue H I BB86/2DCG.',
        )

    if not selected_tracers:
        df_map = df.iloc[0:0].copy()
    else:
        df_map = df.loc[df['tracer'].isin(selected_tracers)].copy()
        if selected_sources:
            df_map = df_map.loc[df_map['source_catalog'].isin(selected_sources)].copy()

    missing_tracers = [t for t in selected_tracers if t not in set(df['tracer'].astype(str).unique())]
    if missing_tracers:
        st.caption('Couches non encore peuplées : ' + ', '.join(missing_tracers))

    st.caption(f'Objets affichés sur la carte : {len(df_map)} / {len(df)}')

    if not BACKGROUND_PNG.exists():
        st.warning('Missing background image: data/m31_background.png')
    fig = build_m31_figure(df_visible=df_map, df_extent=df, background_png=BACKGROUND_PNG, meta=meta, selected_seq=selected_seq)

    event = st.plotly_chart(
        fig,
        key='m31_map_click',
        on_select='rerun',
        selection_mode='points',
        config={'responsive': True},
        width='stretch',
    )

    points = []
    try:
        points = event.selection.points
    except Exception:
        try:
            points = event['selection']['points']
        except Exception:
            points = []

    if points:
        first_point = points[0]
        seq_clicked = None
        try:
            custom = first_point.get('customdata')
            if custom is not None and len(custom) > 0:
                seq_clicked = int(custom[0])
        except Exception:
            seq_clicked = None

        if seq_clicked is None:
            seq_clicked = nearest_seq_from_click(df_map, first_point)

        if seq_clicked is not None and int(seq_clicked) != int(st.session_state['selected_seq']):
            st.session_state['selected_seq'] = int(seq_clicked)
            st.rerun()

    row_left = df.loc[df['Seq'] == int(st.session_state['selected_seq'])].iloc[0]
    render_optional_png(
        row_left,
        'contrast_joint_refit',
        'contrast_joint_refit_png',
        'No contrast_joint_refit PNG found for this hole.',
    )
    render_optional_png(
        row_left,
        '2DCG summary',
        'summary_2dcg_png',
        'No 2DCG summary PNG found for this hole.',
    )

with right:
    selected_seq = int(st.session_state['selected_seq'])
    row = df.loc[df['Seq'] == selected_seq].iloc[0]
    tracer_label = str(row.get('tracer', 'HI'))
    st.markdown(f'### {tracer_label} object {selected_seq:03d}')
    meta_cols = [c for c in ['tracer', 'source_catalog', 'HRV', 'Maj', 'Min', 'PA', 'x_arcmin', 'y_arcmin', 'has_video', 'video_name'] if c in df.columns]
    meta_rows = [{'parameter': col, 'value': value_to_text(row[col])} for col in meta_cols]
    st.dataframe(pd.DataFrame(meta_rows), width='stretch', hide_index=True)

    st.markdown('### 2DCG + joint contrast refit')
    refit_table = build_refit_table(row)
    if refit_table.empty:
        st.info('No 2DCG/joint contrast numerical summary found for this hole.')
    else:
        st.dataframe(refit_table, width='stretch', hide_index=True)

    st.markdown('### PPV video')
    video_path = resolve_video_path(row)
    if video_path is None:
        st.warning('No video associated with this hole.')
    else:
        try:
            display_path = str(video_path.relative_to(APP_DIR))
        except Exception:
            display_path = str(video_path)
        st.code(display_path, language='text')
        if video_path.exists():
            size_mb = video_path.stat().st_size / 1024**2
            st.success(f'Video file found. Size: {size_mb:.1f} MB')
            try:
                video_player_from_file(video_path)
            except Exception as exc:
                st.error(f'Video playback error: {exc}')
        else:
            st.error(f'Catalogue points to a missing video: {video_path}')
    st.markdown('### Catalogue status')
    st.write(f'Displayed objects in selector: `{n_total}`')
    st.write(f'Objects visible on map: `{len(df_map)}`')
    st.write(f'Videos found: `{n_video}`')
    st.write(f'Videos missing: `{n_total - n_video}`')
