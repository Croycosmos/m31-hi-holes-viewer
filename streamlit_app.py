from __future__ import annotations

from pathlib import Path
import json
import base64

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
HI_CATALOG_PATH = DATA_DIR / "holes_catalog_streamlit.csv"
OBJECTS_CATALOG_PATH = DATA_DIR / "objects_catalog_streamlit.csv"
BACKGROUND_PNG = DATA_DIR / "m31_background.png"
BACKGROUND_META = DATA_DIR / "m31_background_meta.json"
DISPLAY_FLIP_X = True

TRACER_OPTIONS = ["HI", "UV", "Hα", "CO", "IR", "X-ray"]
TRACER_COLORS = {
    "HI": "deepskyblue",
    "UV": "violet",
    "Hα": "red",
    "CO": "orange",
    "IR": "gold",
    "X-ray": "lime",
}
TRACER_SYMBOLS = {
    "HI": "circle",
    "UV": "square",
    "Hα": "triangle-up",
    "CO": "diamond",
    "IR": "hexagon",
    "X-ray": "star",
}

st.set_page_config(page_title="M31 multi-tracer cavity viewer", layout="wide")


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def value_to_text(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


@st.cache_data(show_spinner=False)
def load_hi_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Seq"] = df["Seq"].astype(int)
    df["Seq_str"] = df["Seq"].map(lambda x: f"{x:03d}")
    if "has_video" in df.columns:
        df["has_video"] = df["has_video"].map(as_bool)
    else:
        df["has_video"] = df["video_path"].astype(str).str.len() > 0
    if "tracer" not in df.columns:
        df["tracer"] = "HI"
    if "source_catalog" not in df.columns:
        df["source_catalog"] = "BB86 + LGLBS 2DCG/joint contrast"
    if "object_label" not in df.columns:
        df["object_label"] = df["Seq"].map(lambda x: f"HI {int(x):03d}")
    df["x_plot"] = -df["x_arcmin"] if DISPLAY_FLIP_X else df["x_arcmin"]
    df["y_plot"] = df["y_arcmin"]
    return df


@st.cache_data(show_spinner=False)
def load_objects_catalog(path: Path, hi_df: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        obj = pd.read_csv(path)
    else:
        rows = []
        for _, r in hi_df.iterrows():
            rows.append({
                "object_uid": f"HI_BB86_{int(r['Seq']):03d}",
                "tracer": "HI",
                "source_catalog": "BB86 + LGLBS 2DCG/joint contrast",
                "display_label": f"HI {int(r['Seq']):03d}",
                "hi_seq": int(r["Seq"]),
                "ra_deg": r.get("ra_deg", pd.NA),
                "dec_deg": r.get("dec_deg", pd.NA),
                "x_arcmin": r["x_arcmin"],
                "y_arcmin": r["y_arcmin"],
                "major_arcsec": pd.NA,
                "minor_arcsec": pd.NA,
                "pa_deg": r.get("PA", pd.NA),
                "marker_size": 8.0,
                "has_video": bool(r.get("has_video", False)),
            })
        obj = pd.DataFrame(rows)

    for col in ["object_uid", "tracer", "source_catalog", "display_label"]:
        if col not in obj.columns:
            obj[col] = ""
        obj[col] = obj[col].fillna("").astype(str)

    if "hi_seq" not in obj.columns:
        obj["hi_seq"] = pd.NA
    if "has_video" in obj.columns:
        obj["has_video"] = obj["has_video"].map(as_bool)
    else:
        obj["has_video"] = False
    if "marker_size" not in obj.columns:
        obj["marker_size"] = 8.0
    obj["marker_size"] = pd.to_numeric(obj["marker_size"], errors="coerce").fillna(8.0)
    obj["x_arcmin"] = pd.to_numeric(obj["x_arcmin"], errors="coerce")
    obj["y_arcmin"] = pd.to_numeric(obj["y_arcmin"], errors="coerce")
    obj["x_plot"] = -obj["x_arcmin"] if DISPLAY_FLIP_X else obj["x_arcmin"]
    obj["y_plot"] = obj["y_arcmin"]
    return obj


@st.cache_data(show_spinner=False)
def load_background_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def image_file_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def transform_background_extent(meta: dict) -> dict:
    x_min = float(meta["x_min"])
    x_max = float(meta["x_max"])
    y_min = float(meta["y_min"])
    y_max = float(meta["y_max"])
    if DISPLAY_FLIP_X:
        x0 = -x_max
        x1 = -x_min
    else:
        x0 = x_min
        x1 = x_max
    out = {
        "x_min": float(min(x0, x1)),
        "x_max": float(max(x0, x1)),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }
    out["sizex"] = out["x_max"] - out["x_min"]
    out["sizey"] = out["y_max"] - out["y_min"]
    return out


def resolve_video_path(row: pd.Series) -> Path | None:
    raw = str(row.get("video_path", "")).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        path = Path("videos") / path.name
    return APP_DIR / path


def resolve_optional_path(row: pd.Series, column: str) -> Path | None:
    raw = str(row.get(column, "")).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return None
    path = Path(raw)
    if path.is_absolute():
        path = Path(path.name)
    return APP_DIR / path


def render_optional_png(row: pd.Series, title: str, column: str, missing_message: str) -> None:
    st.markdown(f"### {title}")
    path = resolve_optional_path(row, column)
    if path is not None and path.exists():
        st.image(str(path), width="stretch")
    else:
        st.info(missing_message)


def _row_value(row: pd.Series, column: str | None) -> str:
    if column is None or column not in row.index:
        return ""
    return value_to_text(row[column])


def build_refit_table(row: pd.Series) -> pd.DataFrame:
    specs = [
        ("Maj [pc]", "joint__Maj_best_pc", "cg2d__Maj_growth_pc"),
        ("Min [pc]", "joint__Min_best_pc", "cg2d__Min_growth_pc"),
        ("PA [deg]", "joint__PA_astro_best_deg", "cg2d__PA_geometry_deg"),
        ("HRV [km/s]", "joint__HRV_used_kms", "cg2d__hrv_kms"),
        ("N channels", "joint__nch_used", "cg2d__nch_best"),
        ("v min [km/s]", None, "cg2d__vel_lo_kms"),
        ("v max [km/s]", None, "cg2d__vel_hi_kms"),
        ("dx [arcsec]", "joint__dx_best_arcsec", None),
        ("dy [arcsec]", "joint__dy_best_arcsec", None),
        ("dr [arcsec]", "joint__dr_best_arcsec", None),
        ("contrast base", "joint__contrast_base", None),
        ("contrast best", "joint__contrast_best", None),
        ("ΔI/σ base", "joint__delta_I_sn_base", None),
        ("ΔI/σ best", "joint__delta_I_sn_best", "cg2d__delta_I_sn"),
        ("median Nbeam center", None, "cg2d__med_Nbeam_center"),
        ("median Nbeam ring", None, "cg2d__med_Nbeam_ring"),
        ("major r50 [arcsec]", None, "cg2d__major_r50_deficit_arcsec"),
        ("major r80 [arcsec]", None, "cg2d__major_r80_deficit_arcsec"),
        ("minor r50 [arcsec]", None, "cg2d__minor_r50_deficit_arcsec"),
        ("minor r80 [arcsec]", None, "cg2d__minor_r80_deficit_arcsec"),
        ("major growth score", None, "cg2d__major_growth_score"),
        ("minor growth score", None, "cg2d__minor_growth_score"),
        ("status / source", "joint__status", "cg2d__major_source"),
    ]
    rows = []
    for parameter, joint_col, cg2d_col in specs:
        joint_val = _row_value(row, joint_col)
        cg2d_val = _row_value(row, cg2d_col)
        if joint_val or cg2d_val:
            rows.append({"parameter": parameter, "joint_contrast_refit": joint_val, "2DCG": cg2d_val})
    return pd.DataFrame(rows)


def video_player_from_file(path: Path, height: int = 520) -> None:
    if not path.exists():
        st.error(f"Video file not found: {path}")
        return
    video_bytes = path.read_bytes()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    html = f"""
    <video width="100%" controls preload="metadata" style="background-color: black;">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser cannot play this MP4 video.
    </video>
    """
    components.html(html, height=height)
    st.download_button(label="Download MP4", data=video_bytes, file_name=path.name, mime="video/mp4", width="stretch")


def add_background(fig: go.Figure, background_png: Path, meta: dict, df_extent: pd.DataFrame) -> None:
    if background_png.exists() and bool(meta):
        bg = transform_background_extent(meta)
        background_source = image_file_to_data_uri(background_png)
        fig.add_layout_image(dict(
            source=background_source,
            xref="x", yref="y",
            x=bg["x_min"], y=bg["y_max"],
            sizex=bg["sizex"], sizey=bg["sizey"],
            sizing="stretch", opacity=1.0, layer="below",
        ))
        fig.update_xaxes(range=[bg["x_min"], bg["x_max"]])
        fig.update_yaxes(range=[bg["y_min"], bg["y_max"]])
    elif not df_extent.empty:
        xpad = 10.0
        ypad = 10.0
        fig.update_xaxes(range=[float(df_extent["x_plot"].min()) - xpad, float(df_extent["x_plot"].max()) + xpad])
        fig.update_yaxes(range=[float(df_extent["y_plot"].min()) - ypad, float(df_extent["y_plot"].max()) + ypad])
    else:
        fig.update_xaxes(range=[-120.0, 120.0])
        fig.update_yaxes(range=[-80.0, 80.0])


def build_m31_figure(df_visible: pd.DataFrame, df_extent: pd.DataFrame, background_png: Path, meta: dict, selected_hi_seq: int | None) -> go.Figure:
    fig = go.Figure()
    add_background(fig, background_png, meta, df_extent)

    if not df_visible.empty:
        order = {name: i for i, name in enumerate(TRACER_OPTIONS)}
        tracers = sorted(df_visible["tracer"].dropna().astype(str).unique(), key=lambda x: order.get(x, 99))
        for tracer in tracers:
            df_tracer = df_visible.loc[df_visible["tracer"].astype(str) == tracer].copy()
            if df_tracer.empty:
                continue
            color = TRACER_COLORS.get(tracer, "white")
            symbol = TRACER_SYMBOLS.get(tracer, "circle")
            opacity = 0.9 if tracer == "HI" else 0.58
            sizes = pd.to_numeric(df_tracer.get("marker_size", 8), errors="coerce").fillna(8).clip(4, 18)
            fig.add_trace(go.Scatter(
                x=df_tracer["x_plot"],
                y=df_tracer["y_plot"],
                mode="markers",
                name=f"{tracer} — {len(df_tracer)} objects",
                marker=dict(size=sizes, color=color, opacity=opacity, symbol=symbol, line=dict(width=0.5, color="white")),
                text=df_tracer["display_label"],
                customdata=df_tracer[["object_uid", "tracer", "source_catalog", "hi_seq", "x_arcmin", "y_arcmin", "major_arcsec", "minor_arcsec", "pa_deg"]].fillna("").values,
                hovertemplate=(
                    "%{text}<br>"
                    "Tracer=%{customdata[1]}<br>"
                    "Catalog=%{customdata[2]}<br>"
                    "X=%{customdata[4]:.1f} arcmin<br>"
                    "Y=%{customdata[5]:.1f} arcmin<br>"
                    "Major=%{customdata[6]} arcsec<br>"
                    "Minor=%{customdata[7]} arcsec<br>"
                    "PA=%{customdata[8]} deg<br>"
                    "<extra></extra>"
                ),
            ))

    if selected_hi_seq is not None:
        sel = df_extent.loc[(df_extent["tracer"] == "HI") & (pd.to_numeric(df_extent["hi_seq"], errors="coerce") == int(selected_hi_seq))]
        if not sel.empty:
            r = sel.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r["x_plot"]], y=[r["y_plot"]], mode="markers+text", name="selected H I hole",
                marker=dict(size=20, color="red", symbol="circle-open", line=dict(width=3.0, color="red")),
                text=[f"HI {int(selected_hi_seq):03d}"], textposition="top center", textfont=dict(color="red", size=13), hoverinfo="skip"
            ))

    x_title = "Displayed X [arcmin] — East left / West right" if DISPLAY_FLIP_X else "X [arcmin]"
    fig.update_layout(
        title="M31 — multi-tracer structures map",
        template="plotly_dark", height=820,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.25)", borderwidth=1),
        xaxis_title=x_title, yaxis_title="Y [arcmin] — North/South", clickmode="event+select",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig


def get_clicked_object(event) -> dict | None:
    points = []
    try:
        points = event.selection.points
    except Exception:
        try:
            points = event["selection"]["points"]
        except Exception:
            points = []
    if not points:
        return None
    p = points[0]
    try:
        custom = p.get("customdata")
    except Exception:
        custom = None
    if custom is None or len(custom) < 4:
        return None
    return {"object_uid": str(custom[0]), "tracer": str(custom[1]), "source_catalog": str(custom[2]), "hi_seq": custom[3]}


def build_external_object_table(obj_row: pd.Series) -> pd.DataFrame:
    cols = [
        ("tracer", "tracer"),
        ("catalog", "source_catalog"),
        ("label", "display_label"),
        ("RA [deg]", "ra_deg"),
        ("Dec [deg]", "dec_deg"),
        ("X [arcmin]", "x_arcmin"),
        ("Y [arcmin]", "y_arcmin"),
        ("major [arcsec]", "major_arcsec"),
        ("minor [arcsec]", "minor_arcsec"),
        ("PA [deg]", "pa_deg"),
        ("flux", "flux"),
        ("luminosity", "luminosity"),
        ("area [pc²]", "area_pc2"),
        ("age [Myr]", "age_myr"),
        ("mass [M☉]", "mass_msun"),
        ("notes", "notes"),
    ]
    rows = []
    for label, col in cols:
        if col in obj_row.index:
            val = value_to_text(obj_row[col])
            if val:
                rows.append({"parameter": label, "value": val})
    return pd.DataFrame(rows)


st.title("M31 multi-tracer structures viewer")
st.caption(
    "H I objects are cavities/holes. UV and Hα layers are star-forming or ionized regions, not cavities. "
    "CO, IR and X-ray layers are prepared for later catalogues."
)

if not HI_CATALOG_PATH.exists():
    st.error(f"Missing catalogue: {HI_CATALOG_PATH}")
    st.stop()

hi_df = load_hi_catalog(HI_CATALOG_PATH)
objects_df = load_objects_catalog(OBJECTS_CATALOG_PATH, hi_df)
objects_df = objects_df.loc[objects_df["x_plot"].notna() & objects_df["y_plot"].notna()].copy()
meta = load_background_meta(BACKGROUND_META)

if "selected_hi_seq" not in st.session_state or int(st.session_state["selected_hi_seq"]) not in set(hi_df["Seq"].astype(int)):
    first_with_video = hi_df.loc[hi_df["has_video"], "Seq"]
    st.session_state["selected_hi_seq"] = int(first_with_video.iloc[0]) if not first_with_video.empty else int(hi_df["Seq"].iloc[0])
if "clicked_object_uid" not in st.session_state:
    st.session_state["clicked_object_uid"] = ""

left, right = st.columns([1.45, 1.0])

with right:
    st.subheader("H I hole selection")
    n_hi = len(hi_df)
    n_video = int(hi_df["has_video"].sum())
    st.write(f"H I videos found: `{n_video}/{n_hi}`")
    show_only_with_video = st.checkbox("Show only H I holes with video in selector", value=True)
    hi_select = hi_df.loc[hi_df["has_video"]].copy() if show_only_with_video else hi_df.copy()
    seq_options = hi_select["Seq_str"].tolist()
    selected_seq_str = f"{int(st.session_state['selected_hi_seq']):03d}"
    index = seq_options.index(selected_seq_str) if selected_seq_str in seq_options else 0
    selected = st.selectbox("H I hole", seq_options, index=index)
    st.session_state["selected_hi_seq"] = int(selected)

with left:
    st.subheader("M31 map")
    ctrl1, ctrl2 = st.columns([1.0, 1.0])
    with ctrl1:
        available_tracers = [t for t in TRACER_OPTIONS if t in set(objects_df["tracer"].astype(str))]
        selected_tracers = st.multiselect(
            "Traceurs affichés",
            TRACER_OPTIONS,
            default=[t for t in ["HI", "UV", "Hα"] if t in available_tracers],
            help="H I = cavités. UV/Hα = régions de formation stellaire ou ionisées. CO/IR/X-ray attendent leurs catalogues.",
        )
    with ctrl2:
        possible_sources = sorted(objects_df.loc[objects_df["tracer"].isin(selected_tracers), "source_catalog"].dropna().astype(str).unique())
        selected_sources = st.multiselect("Catalogues affichés", possible_sources, default=possible_sources)

    df_map = objects_df.loc[objects_df["tracer"].isin(selected_tracers)].copy() if selected_tracers else objects_df.iloc[0:0].copy()
    if selected_sources:
        df_map = df_map.loc[df_map["source_catalog"].isin(selected_sources)].copy()

    missing_tracers = [t for t in selected_tracers if t not in set(objects_df["tracer"].astype(str).unique())]
    if missing_tracers:
        st.caption("Couches non encore peuplées : " + ", ".join(missing_tracers))

    counts = df_map.groupby("tracer").size().to_dict() if not df_map.empty else {}
    counts_text = " | ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "aucun objet"
    st.caption(f"Objets affichés : {len(df_map)} / {len(objects_df)} — {counts_text}")

    if not BACKGROUND_PNG.exists():
        st.warning("Missing background image: data/m31_background.png")
    fig = build_m31_figure(df_visible=df_map, df_extent=objects_df, background_png=BACKGROUND_PNG, meta=meta, selected_hi_seq=int(st.session_state["selected_hi_seq"]))
    event = st.plotly_chart(fig, key="m31_map_click", on_select="rerun", selection_mode="points", config={"responsive": True}, width="stretch")

    clicked = get_clicked_object(event)
    if clicked:
        st.session_state["clicked_object_uid"] = clicked["object_uid"]
        if clicked["tracer"] == "HI" and str(clicked["hi_seq"]).strip() not in {"", "nan", "None"}:
            try:
                new_seq = int(float(clicked["hi_seq"]))
                if new_seq != int(st.session_state["selected_hi_seq"]):
                    st.session_state["selected_hi_seq"] = new_seq
                    st.rerun()
            except Exception:
                pass

    row_left = hi_df.loc[hi_df["Seq"] == int(st.session_state["selected_hi_seq"])].iloc[0]
    render_optional_png(row_left, "contrast_joint_refit", "contrast_joint_refit_png", "No contrast_joint_refit PNG found for this H I hole.")
    render_optional_png(row_left, "2DCG summary", "summary_2dcg_png", "No 2DCG summary PNG found for this H I hole.")

with right:
    selected_hi_seq = int(st.session_state["selected_hi_seq"])
    row = hi_df.loc[hi_df["Seq"] == selected_hi_seq].iloc[0]
    st.markdown(f"### Selected H I hole {selected_hi_seq:03d}")
    meta_cols = [c for c in ["tracer", "source_catalog", "HRV", "Maj", "Min", "PA", "x_arcmin", "y_arcmin", "has_video", "video_name"] if c in hi_df.columns]
    meta_rows = [{"parameter": col, "value": value_to_text(row[col])} for col in meta_cols]
    st.dataframe(pd.DataFrame(meta_rows), width="stretch", hide_index=True)

    st.markdown("### 2DCG + joint contrast refit")
    refit_table = build_refit_table(row)
    if refit_table.empty:
        st.info("No 2DCG/joint contrast numerical summary found for this H I hole.")
    else:
        st.dataframe(refit_table, width="stretch", hide_index=True)

    clicked_uid = str(st.session_state.get("clicked_object_uid", ""))
    if clicked_uid:
        clicked_rows = objects_df.loc[objects_df["object_uid"] == clicked_uid]
        if not clicked_rows.empty:
            obj_row = clicked_rows.iloc[0]
            st.markdown("### Last clicked map object")
            st.dataframe(build_external_object_table(obj_row), width="stretch", hide_index=True)

    st.markdown("### PPV video")
    video_path = resolve_video_path(row)
    if video_path is None:
        st.warning("No video associated with this H I hole.")
    else:
        try:
            display_path = str(video_path.relative_to(APP_DIR))
        except Exception:
            display_path = str(video_path)
        st.code(display_path, language="text")
        if video_path.exists():
            size_mb = video_path.stat().st_size / 1024**2
            st.success(f"Video file found. Size: {size_mb:.1f} MB")
            try:
                video_player_from_file(video_path)
            except Exception as exc:
                st.error(f"Video playback error: {exc}")
        else:
            st.error(f"Catalogue points to a missing video: {video_path}")

    st.markdown("### Catalogue status")
    st.write(f"H I holes: `{len(hi_df)}`")
    st.write(f"All map objects: `{len(objects_df)}`")
    st.write(f"Objects visible on map: `{len(df_map)}`")
    st.write(f"H I videos found: `{n_video}`")
    st.write(f"H I videos missing: `{n_hi - n_video}`")
