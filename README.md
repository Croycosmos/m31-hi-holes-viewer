# M31 H I holes — PPV flythrough viewer

Streamlit viewer for the 141 Brinks & Bajaja (1986) H I holes in M31, prepared for multi-tracer cavity layers.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Contents

- `streamlit_app.py`: Streamlit app.
- `data/holes_catalog_streamlit.csv`: catalogue used by the app.
- `data/m31_background.png`: M31 H I background image.
- `data/m31_background_meta.json`: background image extent metadata.
- `videos/`: PPV flythrough MP4 videos.
- `figures/contrast_joint_refit/`: contrast-joint-refit PNGs for selected holes.
- `figures/2dcg_summary/`: 2D cumulative-growth summary PNGs for selected holes.
- tracer filters in the app are ready for future CO, UV, Hα, IR and X-ray catalogues.

This exported version uses relative paths only. It does not depend on `/johannes/...`, SSH tunnels, or a local video server.
