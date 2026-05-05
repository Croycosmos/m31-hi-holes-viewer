# M31 multi-tracer structures viewer

Streamlit viewer for M31 H I holes and contextual multi-tracer structures.

Current populated layers:

- H I: Brinks & Bajaja (1986) holes, refit with LGLBS/2DCG products.
- UV: Kang et al. (2009) GALEX star-forming regions.
- H-alpha: Azimlu et al. (2011) H II regions.

Important terminology: only H I objects are called cavities/holes in this app. UV and H-alpha layers are contextual tracers of young stellar populations and ionized gas.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
