"""Argentina · Liquidez & Política Económica — entry point para Streamlit Cloud.

Run local:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from argentina_macro.dashboard import panels

st.set_page_config(
    page_title="Argentina · Liquidez & Política Económica",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Estilos: tipografía compacta + look moderno ───────────────────────
st.markdown(
    """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }
  html, body, [class*="st-"] { font-size: 13.5px; }
  h1 { font-size: 1.45rem !important; font-weight: 600 !important;
       margin-bottom: .15rem !important; letter-spacing: -0.01em !important; }
  h2 { font-size: 1.1rem !important; font-weight: 600 !important; margin-top: .6rem !important; }
  h3 { font-size: .98rem !important; font-weight: 600 !important; color: #334155; }
  .stCaption, [data-testid="stCaptionContainer"] {
    font-size: 12px !important; color: #64748b !important; }
  .stAlert { padding: .55rem .8rem !important; font-size: 12.5px !important; border-radius: 6px; }
  .stAlert p { margin: 0 !important; }
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 1px solid #e2e8f0; margin-bottom: .5rem; }
  .stTabs [data-baseweb="tab"] {
    height: 36px; padding: 0 14px; font-size: 13px; font-weight: 500;
    color: #64748b; background: transparent; border-radius: 6px 6px 0 0; }
  .stTabs [aria-selected="true"] {
    color: #0f172a !important; background: #f1f5f9 !important;
    border-bottom: 2px solid #6366f1 !important; }
  [data-testid="stDateInput"] label { font-size: 11px !important; color: #64748b !important; }
  [data-testid="stDateInput"] input { font-size: 12.5px !important; padding: .25rem .5rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
  [data-testid="stMetricLabel"] { font-size: 11.5px !important; color: #64748b !important; }
  [data-testid="stMetricDelta"] { font-size: 11px !important; }
  .ar-meta { font-size: 11px; color: #94a3b8; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Cabecera ──────────────────────────────────────────────────────────
hdr_title, hdr_dates = st.columns([3, 2])
with hdr_title:
    st.markdown("# Argentina · liquidez como lente de política económica")
    st.markdown(
        '<div class="ar-meta">Foco: cómo agregados monetarios y pasivos remunerados '
        "se traducen en inflación, con contexto fiscal, externo y real.</div>",
        unsafe_allow_html=True,
    )
with hdr_dates:
    d1, d2 = st.columns(2)
    with d1:
        start = st.date_input("Desde", value=date(2023, 1, 1), label_visibility="visible")
    with d2:
        end = st.date_input("Hasta", value=date.today(), label_visibility="visible")

# ─── Tabs ──────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Liquidez monetaria",
    "📈 Inflación y tasas",
    "💱 Sector externo / cambiario",
    "🏛️ Deuda y fiscal",
    "🏗️ Sector real",
    "🔮 Predictor inflación",
])

with tabs[0]:
    panels.render_monetary(start, end)
with tabs[1]:
    panels.render_inflation(start, end)
with tabs[2]:
    panels.render_fx(start, end)
with tabs[3]:
    panels.render_debt_fiscal(start, end)
with tabs[4]:
    panels.render_real_sector(start, end)
with tabs[5]:
    panels.render_inflation_predictor(start, end)
