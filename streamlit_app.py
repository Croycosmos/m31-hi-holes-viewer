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

st.set_page_config(page_title='M31 H I holes PPV viewer', layout='wide')


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


def build_m31_figure(df: pd.DataFrame, background_png: Path, meta: dict, selected_seq: int | None) -> go.Figure:
    fig = go.Figure()
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
        fig.update_xaxes(range=[float(df['x_plot'].min()) - xpad, float(df['x_plot'].max()) + xpad])
        fig.update_yaxes(range=[float(df['y_plot'].min()) - ypad, float(df['y_plot'].max()) + ypad])

    df_video = df.loc[df['has_video']].copy()
    df_no_video = df.loc[~df['has_video']].copy()

    if not df_no_video.empty:
        fig.add_trace(go.Scatter(
            x=df_no_video['x_plot'], y=df_no_video['y_plot'], mode='markers', name='without video',
            marker=dict(size=7, color='rgba(180,180,180,0.55)', line=dict(width=0.5, color='white'), symbol='circle-open'),
            text=df_no_video['Seq_str'],
            customdata=df_no_video[['Seq', 'x_arcmin', 'y_arcmin', 'HRV', 'Maj', 'Min', 'PA']],
            hovertemplate=(
                'Hole %{text}<br>X catalog=%{customdata[1]:.1f} arcmin<br>Y catalog=%{customdata[2]:.1f} arcmin<br>'
                'HRV=%{customdata[3]:.1f} km/s<br>Maj=%{customdata[4]:.0f} pc<br>Min=%{customdata[5]:.0f} pc<br>'
                'PA=%{customdata[6]:.1f} deg<br>video: no<extra></extra>'
            ),
        ))

    if not df_video.empty:
        fig.add_trace(go.Scatter(
            x=df_video['x_plot'], y=df_video['y_plot'], mode='markers', name='with video',
            marker=dict(size=8, color='deepskyblue', opacity=0.88, line=dict(width=0.8, color='white')),
            text=df_video['Seq_str'],
            customdata=df_video[['Seq', 'x_arcmin', 'y_arcmin', 'HRV', 'Maj', 'Min', 'PA']],
            hovertemplate=(
                'Hole %{text}<br>X catalog=%{customdata[1]:.1f} arcmin<br>Y catalog=%{customdata[2]:.1f} arcmin<br>'
                'HRV=%{customdata[3]:.1f} km/s<br>Maj=%{customdata[4]:.0f} pc<br>Min=%{customdata[5]:.0f} pc<br>'
                'PA=%{customdata[6]:.1f} deg<br>video: yes<extra></extra>'
            ),
        ))

    if selected_seq is not None:
        sel = df.loc[df['Seq'] == int(selected_seq)]
        if not sel.empty:
            r = sel.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r['x_plot']], y=[r['y_plot']], mode='markers+text', name='selected',
                marker=dict(size=18, color='red', symbol='circle-open', line=dict(width=3.0, color='red')),
                text=[f"{int(r['Seq']):03d}"], textposition='top center', textfont=dict(color='red', size=13), hoverinfo='skip'
            ))

    x_title = 'Displayed X [arcmin] — East left / West right' if DISPLAY_FLIP_X else 'X [arcmin]'
    fig.update_layout(
        title='M31 — H I map with BB86 holes', template='plotly_dark', height=820,
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
    dx = df['x_plot'].astype(float) - x
    dy = df['y_plot'].astype(float) - y
    dist2 = dx * dx + dy * dy
    if dist2.empty:
        return None
    i = int(dist2.idxmin())
    return int(df.loc[i, 'Seq'])


st.title('M31 H I holes — PPV flythrough viewer')
st.caption('Interactive viewer for the 141 BB86 H I holes in M31. The background is an H I map; points are hole positions; videos show PPV flythroughs.')

if not CATALOG_PATH.exists():
    st.error(f'Missing catalogue: {CATALOG_PATH}')
    st.stop()

df = load_catalog(CATALOG_PATH)
needed = {'Seq', 'Seq_str', 'x_arcmin', 'y_arcmin', 'x_plot', 'y_plot', 'video_path', 'has_video'}
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
    st.subheader('Hole selection')
    n_total = len(df)
    n_video = int(df['has_video'].sum())
    st.write(f'Videos found: `{n_video}/{n_total}`')
    show_only_with_video = st.checkbox('Show only holes with video', value=True)
    df_select = df.copy()
    if show_only_with_video:
        df_select = df_select.loc[df_select['has_video']].copy()
    if df_select.empty:
        st.warning('No hole with video found in the catalogue.')
        st.stop()
    seq_options = df_select['Seq_str'].tolist()
    selected_seq_str = f"{int(st.session_state['selected_seq']):03d}"
    index = seq_options.index(selected_seq_str) if selected_seq_str in seq_options else 0
    selected = st.selectbox('H I hole', seq_options, index=index)
    selected_seq = int(selected)
    st.session_state['selected_seq'] = selected_seq

with left:
    st.subheader('M31 map')
    if not BACKGROUND_PNG.exists():
        st.warning('Missing background image: data/m31_background.png')
    fig = build_m31_figure(df=df, background_png=BACKGROUND_PNG, meta=meta, selected_seq=selected_seq)

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
            seq_clicked = nearest_seq_from_click(df, first_point)

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
    st.markdown(f'### BB86 hole {selected_seq:03d}')
    meta_cols = [c for c in ['HRV', 'Maj', 'Min', 'PA', 'x_arcmin', 'y_arcmin', 'has_video', 'video_name'] if c in df.columns]
    meta_rows = [{'parameter': col, 'value': value_to_text(row[col])} for col in meta_cols]
    st.dataframe(pd.DataFrame(meta_rows), width='stretch', hide_index=True)
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
    st.write(f'Displayed holes: `{n_total}`')
    st.write(f'Videos found: `{n_video}`')
    st.write(f'Videos missing: `{n_total - n_video}`')
