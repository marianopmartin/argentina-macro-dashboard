"""Panels del dashboard de Argentina. Cada función recibe rango de fechas y renderiza su sección.

Todos los fetches están envueltos en st.cache_data (ttl 1h).
"""

from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from argentina_macro.analysis import distributed_lag_predictor as dl_mod
from argentina_macro.data import bcra, indec, mecon, reservas, secfin

# Fecha del cambio de régimen monetario: devaluación + ajuste fiscal Milei.
# Nota: usar string para add_vline (pasar pd.Timestamp rompe con pandas reciente).
REGIME_BREAK = "2023-12-01"
REGIME_BREAK_TS = pd.Timestamp(REGIME_BREAK)

# ─────────────────────────────────────────────────────────────────────
# Paleta moderna unificada — sin verde/rojo de semáforo
# ─────────────────────────────────────────────────────────────────────
# Positivo / negativo (en lugar de verde/rojo): indigo/amber
COLOR_POS = "#6366f1"        # indigo-500
COLOR_NEG = "#f59e0b"        # amber-500
COLOR_POS_DEEP = "#4f46e5"   # indigo-600 (para texto sobre claro)
COLOR_NEG_DEEP = "#d97706"   # amber-600
# Acentos / categorias
COLOR_ACCENT = "#0ea5e9"     # sky-500
COLOR_BLUE   = "#2563eb"     # blue-600 (datos primarios)
COLOR_INK    = "#0f172a"     # slate-900 (líneas principales)
COLOR_MUTED  = "#94a3b8"     # slate-400 (referencias)
COLOR_GRID   = "#e5e7eb"     # gray-200

# Paleta categórica (para sectores, instrumentos, etc.)
PALETTE = [
    "#6366f1",  # indigo
    "#0ea5e9",  # sky
    "#0d9488",  # teal
    "#84cc16",  # lime
    "#f59e0b",  # amber
    "#ec4899",  # pink
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#a16207",  # yellow-700
    "#64748b",  # slate
    "#fb923c",  # orange-light
    "#a855f7",  # purple
    "#22c55e",  # green-suave
    "#0284c7",  # sky-darker
    "#f43f5e",  # rose
]


def _modern_layout(fig, height: int = 360) -> None:
    """Aplica layout moderno consistente: tipografía sans, grid sutil.
    Conservador — no toca background para evitar romper charts existentes."""
    fig.update_layout(
        font=dict(family="system-ui, -apple-system, sans-serif", size=12,
                  color=COLOR_INK),
        height=height,
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=COLOR_GRID, zeroline=False)


# ─────────────────────────────────────────────────────────────────────
# Fetchers cacheados
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _bcra(name: str, start: date, end: date) -> pd.DataFrame:
    return bcra.fetch(name, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _bcra_m3(start: date, end: date) -> pd.DataFrame:
    """M3 diario construido desde BCRA (M2 + plazo fijo privado), no el mensual de
    datos.gob.ar que tiene ~2-3 meses de lag."""
    return bcra.fetch_m3(start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _indec(name: str, start: date, end: date, representation: str = "value") -> pd.DataFrame:
    return indec.fetch(name, start, end, representation)


@st.cache_data(ttl=3600, show_spinner=False)
def _emae_general(serie: str = "desest") -> pd.DataFrame:
    """EMAE nivel general directo del XLS de INDEC (más actualizado que datos.gob.ar)."""
    return indec.fetch_emae_general(serie)


@st.cache_data(ttl=3600, show_spinner=False)
def _emae_sector(sector: str) -> pd.DataFrame:
    """EMAE por sector económico directo del XLS de INDEC."""
    return indec.fetch_emae_sector(sector)


@st.cache_data(ttl=3600, show_spinner=False)
def _mecon(name: str, start: date, end: date, representation: str = "value") -> pd.DataFrame:
    return mecon.fetch(name, start, end, representation)


@st.cache_data(ttl=86400, show_spinner=False)
def _lecap_stock() -> pd.DataFrame:
    return secfin.lecap_stock_history()


@st.cache_data(ttl=86400, show_spinner=False)
def _lecap_tasa() -> pd.DataFrame:
    return secfin.lecap_avg_tasa_history()


@st.cache_data(ttl=86400, show_spinner=False)
def _intereses_capitalizados() -> pd.DataFrame:
    return secfin.intereses_capitalizados_history()


@st.cache_data(ttl=86400, show_spinner=False)
def _tesoro_breakdown() -> pd.DataFrame:
    return secfin.total_pesos_stock_breakdown()


@st.cache_data(ttl=86400, show_spinner=False)
def _deuda_total_bruta() -> pd.DataFrame:
    return secfin.deuda_total_bruta_history()


@st.cache_data(ttl=86400, show_spinner=False)
def _deuda_breakdown() -> pd.DataFrame:
    return secfin.deuda_breakdown_categorias()


@st.cache_data(ttl=86400, show_spinner=False)
def _deuda_total_extendida() -> pd.DataFrame:
    return secfin.deuda_total_bruta_extendida()


@st.cache_data(ttl=86400, show_spinner=False)
def _deuda_breakdown_extendido() -> pd.DataFrame:
    return secfin.deuda_breakdown_extendido()


@st.cache_data(ttl=86400, show_spinner=False)
def _deuda_pbi_oficial() -> pd.DataFrame:
    return secfin.deuda_pbi_oficial()


@st.cache_data(ttl=3600, show_spinner=False)
def _reservas_netas(start: date, end: date) -> pd.DataFrame:
    return reservas.reservas_netas_history(start, end)


@st.cache_data(ttl=3600, show_spinner=True)
def _distributed_lag_run(start_iso: str) -> dict:
    """Lasso distributed-lag (sin inercia, todos los monetarios)."""
    return dl_mod.evaluate(start=date.fromisoformat(start_iso))


@st.cache_data(ttl=900, show_spinner=False)
def _dolar_snapshot() -> pd.DataFrame:
    """Snapshot actual de cotizaciones desde dolarapi.com (sin histórico)."""
    r = httpx.get("https://dolarapi.com/v1/dolares", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _add_regime_line(fig: go.Figure) -> None:
    """Marca vertical en la fecha del cambio de régimen (dic-2023). Sin annotation
    (annotation_text rompe add_vline con timestamps en pandas reciente)."""
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=REGIME_BREAK, x1=REGIME_BREAK, y0=0, y1=1,
        line=dict(color="#888", width=1.2, dash="dot"),
    )


def _line(df: pd.DataFrame, title: str, y_label: str, height: int = 320,
          regime_line: bool = True) -> None:
    if df.empty:
        st.warning(f"Sin datos para: {title}")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["valor"],
                             line=dict(color=COLOR_BLUE, width=2),
                             mode="lines", showlegend=False))
    fig.update_layout(title=title, yaxis_title=y_label, xaxis_title=None,
                      margin=dict(t=40, b=20, l=10, r=10))
    _modern_layout(fig, height=height)
    if regime_line:
        _add_regime_line(fig)
    st.plotly_chart(fig, width="stretch")


def _bar(df: pd.DataFrame, title: str, y_label: str, color_by_sign: bool = False,
         regime_line: bool = True) -> None:
    if df.empty:
        st.warning(f"Sin datos para: {title}")
        return
    fig = go.Figure()
    if color_by_sign:
        colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df["valor"]]
        fig.add_trace(go.Bar(x=df["fecha"], y=df["valor"], marker_color=colors,
                             marker_line_width=0))
    else:
        fig.add_trace(go.Bar(x=df["fecha"], y=df["valor"], marker_color=COLOR_BLUE,
                             marker_line_width=0))
    fig.update_layout(title=title, yaxis_title=y_label, xaxis_title=None,
                      margin=dict(t=40, b=20, l=10, r=10))
    _modern_layout(fig, height=320)
    if regime_line:
        _add_regime_line(fig)
    st.plotly_chart(fig, width="stretch")


def _line_nominal_real_usd(
    serie_pesos: pd.DataFrame,
    ipc: pd.DataFrame,
    tc: pd.DataFrame,
    title: str,
    height: int = 320,
) -> None:
    """Plot 3 líneas indexadas a primer mes visible = 100.
    Nominal $, real (IPC-deflactado a último IPC), y en USD (dividido por TC mayorista).

    Permite ver crecimiento nominal vs real vs en dólares — el contraste
    cuenta la historia de licuación / apreciación cambiaria.
    """
    if serie_pesos.empty or ipc.empty or tc.empty:
        st.warning(f"Sin datos suficientes para: {title}")
        return
    s = serie_pesos.set_index("fecha")["valor"]
    if len(s) > 50:  # serie diaria (BCRA daily)
        s = s.resample("MS").last()
    ipc_s = ipc.set_index("fecha")["valor"]
    tc_s = tc.set_index("fecha")["valor"]
    if len(tc_s) > 100:
        tc_s = tc_s.resample("MS").last()
    # Alinear todos los índices a inicio de mes
    s.index = s.index.to_period("M").to_timestamp()
    ipc_s.index = ipc_s.index.to_period("M").to_timestamp()
    tc_s.index = tc_s.index.to_period("M").to_timestamp()
    # Outer join — cada serie tiene su propio rango de validez
    df = pd.concat([s.rename("nom"), ipc_s.rename("ipc"), tc_s.rename("tc")], axis=1).sort_index()
    # Recortar al rango donde al menos el nominal exista
    df = df[df["nom"].notna()]
    if df.empty or len(df) < 2:
        st.warning(f"Sin overlap para: {title}")
        return
    # Período base de indexación: primer mes donde existen las 3 series (para que los
    # tres índices estén normalizados a la misma fecha = 100)
    base_idx = df.dropna(subset=["nom", "ipc", "tc"]).index
    if len(base_idx) == 0:
        st.warning(f"Sin overlap inicial 3-series para: {title}")
        return
    base_date = base_idx[0]
    base_nom = float(df.loc[base_date, "nom"])
    base_ipc = float(df.loc[base_date, "ipc"])
    base_tc = float(df.loc[base_date, "tc"])

    # Real: requiere ipc disponible; usa último IPC observado como deflactor
    ipc_last_idx = df["ipc"].dropna().index[-1]
    ipc_last_val = float(df.loc[ipc_last_idx, "ipc"])
    df["real"] = df["nom"] * ipc_last_val / df["ipc"]
    df["usd"] = df["nom"] / df["tc"]

    # Índices (base mes inicial = 100). Real se trunca al último IPC.
    df["nom_idx"] = df["nom"] / base_nom * 100
    df["real_idx"] = (df["nom"] * ipc_last_val / df["ipc"]) / (base_nom * ipc_last_val / base_ipc) * 100
    df["usd_idx"] = df["usd"] / (base_nom / base_tc) * 100

    base_label = base_date.strftime("%b-%Y")
    ipc_last_label = ipc_last_idx.strftime("%b-%Y")
    real_series = df["real_idx"].where(df["ipc"].notna())
    usd_series = df["usd_idx"].where(df["tc"].notna())

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["nom_idx"], name="Nominal $",
                             line=dict(color="#1f77b4", width=2.2),
                             connectgaps=False),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=real_series,
                             name=f"Real (en $ {ipc_last_label})",
                             line=dict(color="#6366f1", width=2, dash="dash"),
                             connectgaps=False),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=usd_series, name="En USD",
                             line=dict(color="#f59e0b", width=2, dash="dot"),
                             connectgaps=False),
                  secondary_y=True)
    fig.update_layout(title=title,
                      xaxis_title=None, height=height + 50, hovermode="x unified",
                      margin=dict(t=40, b=80, l=10, r=10),
                      legend=dict(orientation="h", y=-0.20, x=0.5,
                                  xanchor="center", yanchor="top", font=dict(size=11)))
    fig.update_yaxes(title_text=f"Nominal · {base_label}=100",
                     secondary_y=False, color="#1f77b4")
    fig.update_yaxes(title_text=f"Real / USD · {base_label}=100",
                     secondary_y=True, color="#374151")
    _add_regime_line(fig)
    st.plotly_chart(fig, width="stretch")


def _to_monthly_eom(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    """Reduce serie diaria a mensual fin-de-período (default) o promedio."""
    if df.empty:
        return df
    g = df.set_index("fecha").resample("MS")
    s = g["valor"].last() if how == "last" else g["valor"].mean()
    return s.dropna().reset_index()


def _deflate(nominal: pd.DataFrame, ipc_idx: pd.DataFrame) -> pd.DataFrame:
    """Deflactar serie nominal por IPC (índice). Resultado en pesos del último IPC disponible."""
    if nominal.empty or ipc_idx.empty:
        return nominal
    ipc = ipc_idx.set_index("fecha")["valor"]
    base = ipc.iloc[-1]
    factor = ipc / base  # pasado < 1
    out = nominal.copy().set_index("fecha")
    out["valor"] = out["valor"] / factor.reindex(out.index, method="nearest")
    return out.dropna().reset_index()


def _freshness(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"{label}: sin datos"
    last = pd.Timestamp(df["fecha"].iloc[-1]).strftime("%Y-%m-%d")
    return f"{label}: última obs {last}"


# ─────────────────────────────────────────────────────────────────────
# 1. Liquidez → Inflación  (panel core, primero por importancia)
# ─────────────────────────────────────────────────────────────────────

def _lag_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    """Pearson(x.shift(k), y) para k en [0, max_lag]."""
    rows = []
    for lag in range(max_lag + 1):
        x_lag = x.shift(lag)
        valid = x_lag.notna() & y.notna()
        if valid.sum() < 6:
            continue
        rho = float(np.corrcoef(x_lag[valid], y[valid])[0, 1])
        rows.append({"lag_meses": lag, "correlacion": rho, "n": int(valid.sum())})
    return pd.DataFrame(rows)


def _liquidity_inflation_block(
    serie_mensual: pd.Series, ipc_mm: pd.DataFrame, agregado_label: str,
) -> None:
    """Genera la dupla de plots (overlay m/m + lag correlation pre/post) para un
    agregado monetario dado.
    """
    if serie_mensual.empty:
        st.warning(f"Sin datos para {agregado_label}.")
        return
    serie_df = serie_mensual.reset_index()
    serie_df.columns = ["fecha", "valor"]
    serie_df["mm"] = serie_df["valor"].pct_change(1)
    df = serie_df.merge(ipc_mm, on="fecha", how="inner").dropna()
    if df.empty:
        st.warning(f"Sin overlap {agregado_label} con IPC.")
        return

    # Plot 1: overlay
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df["fecha"], y=df["mm"] * 100, name=f"{agregado_label} m/m",
               marker_color="#1f77b4", opacity=0.55),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["fecha"], y=df["ipc_mm"] * 100, name="IPC m/m",
                   line=dict(color="#f59e0b", width=2.5)),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{agregado_label} m/m vs IPC m/m",
        height=380, hovermode="x unified",
        margin=dict(t=40, b=70, l=10, r=10),
        legend=dict(orientation="h", y=-0.18, x=0.5,
                    xanchor="center", yanchor="top"),
    )
    fig.update_yaxes(title_text=f"{agregado_label} m/m (%)", secondary_y=False)
    fig.update_yaxes(title_text="IPC m/m (%)", secondary_y=True)
    _add_regime_line(fig)
    st.plotly_chart(fig, width="stretch")

    # Plot 2: lag correlation pre/post régimen
    df["pre"] = df["fecha"] < REGIME_BREAK_TS
    pre = df[df["pre"]]
    post = df[~df["pre"]]
    c1, c2 = st.columns(2)
    for col, sub, label in [(c1, pre, "PRE (≤ nov-2023)"), (c2, post, "POST (≥ dic-2023)")]:
        with col:
            if len(sub) < 8:
                st.info(f"{label}: <8 obs.")
                continue
            max_lag = min(12, len(sub) // 3)
            corr = _lag_correlation(sub["mm"], sub["ipc_mm"], max_lag)
            if corr.empty:
                st.info(f"{label}: sin correlación calculable.")
                continue
            fig = go.Figure()
            fig.add_trace(go.Bar(x=corr["lag_meses"], y=corr["correlacion"],
                                 text=[f"n={n}" for n in corr["n"]], textposition="outside"))
            fig.add_hline(y=0, line_color="#888", line_width=1)
            fig.update_layout(title=label, xaxis_title="Lag (meses)",
                              yaxis_title="ρ", height=280,
                              margin=dict(t=40, b=30, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
            best = corr.loc[corr["correlacion"].abs().idxmax()]
            st.caption(f"Lag máx |ρ|: **{int(best['lag_meses'])} meses**, "
                       f"ρ = {best['correlacion']:+.2f}, n={int(best['n'])}.")


def render_liquidity_inflation(start: date, end: date) -> None:
    st.subheader("Liquidez → Inflación (lente principal)")
    st.caption(
        "Tres agregados monetarios contrastados con IPC m/m + correlación cruzada "
        "partida en dic-2023 (cambio de régimen rompe la estacionariedad). "
        "Lectura: bajo régimen actual la base monetaria es endógena al rollover de LECAPs "
        "del Tesoro, M2/M3 son mejores proxies de \"dinero en manos del público\"."
    )

    ipc_mm = _indec("ipc_nacional", start, end, representation="percent_change")
    if ipc_mm.empty:
        st.warning("Sin datos de IPC m/m.")
        return
    ipc_mm = ipc_mm.rename(columns={"valor": "ipc_mm"})

    base = _to_monthly_eom(_bcra("base_monetaria", start, end), how="last")
    m2 = _to_monthly_eom(_bcra("m2_nivel", start, end), how="last")
    m3_eom = _to_monthly_eom(_bcra_m3(start, end), how="last")
    m3 = (m3_eom.set_index("fecha")["valor"]
          if not m3_eom.empty else pd.Series(dtype=float))

    base_s = (base.set_index("fecha")["valor"]
              if not base.empty else pd.Series(dtype=float))
    m2_s = (m2.set_index("fecha")["valor"]
            if not m2.empty else pd.Series(dtype=float))

    st.markdown("### Base monetaria")
    _liquidity_inflation_block(base_s, ipc_mm, "Base monetaria")

    st.markdown("### M2 (nivel)")
    _liquidity_inflation_block(m2_s, ipc_mm, "M2")

    st.markdown("### M3 (pesos)")
    _liquidity_inflation_block(m3, ipc_mm, "M3")


# ─────────────────────────────────────────────────────────────────────
# 2. Liquidez monetaria
# ─────────────────────────────────────────────────────────────────────

def render_monetary(start: date, end: date) -> None:
    st.subheader("Liquidez monetaria")
    st.caption(
        "Cada agregado en pesos lleva 3 líneas indexadas (mes inicial = 100): nominal $, "
        "real (IPC-deflactado a $ del último mes) y en USD (dividido por TC mayorista). "
        "El contraste muestra licuación nominal vs erosión real vs apreciación/depreciación cambiaria."
    )

    ipc = _indec("ipc_nacional", start, end)
    tc = _bcra("fx_mayorista", start, end)

    c1, c2 = st.columns(2)
    with c1:
        _line_nominal_real_usd(_bcra("base_monetaria", start, end), ipc, tc,
                               "Base monetaria")
        _line_nominal_real_usd(_bcra_m3(start, end), ipc, tc,
                               "M3 en pesos (diario, M2 + plazo fijo)")
    with c2:
        _line_nominal_real_usd(_bcra("m2_nivel", start, end), ipc, tc,
                               "M2 (nivel)")
        _line_nominal_real_usd(_indec("m3_pesos_y_usd", start, end), ipc, tc,
                               "M3* (pesos + depósitos USD — mensual, INDEC)")



# ─────────────────────────────────────────────────────────────────────
# 3. Inflación y tasas
# ─────────────────────────────────────────────────────────────────────

def render_inflation(start: date, end: date) -> None:
    st.subheader("Inflación y tasas")
    st.caption(
        "Headline vs núcleo (núcleo excluye estacionales y regulados). "
        "Tasa real ex-post correctamente computada como TPM/12 menos IPC m/m."
    )

    ipc_mm = _indec("ipc_nacional", start, end, representation="percent_change")
    nucleo_mm = _indec("ipc_nucleo", start, end, representation="percent_change")

    if not ipc_mm.empty and not nucleo_mm.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ipc_mm["fecha"], y=ipc_mm["valor"] * 100,
                                 name="IPC headline", line=dict(width=2.5)))
        fig.add_trace(go.Scatter(x=nucleo_mm["fecha"], y=nucleo_mm["valor"] * 100,
                                 name="IPC núcleo", line=dict(width=2.5, dash="dash")))
        fig.update_layout(title="IPC nacional — variación mensual",
                          yaxis_title="% m/m", xaxis_title=None,
                          height=380, hovermode="x unified",
                          margin=dict(t=40, b=20, l=10, r=10))
        _add_regime_line(fig)
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        _line(_bcra("tasa_politica_monetaria", start, end),
              "Tasa de política monetaria", "% TNA")
    with c2:
        _line(_bcra("badlar_privados", start, end), "BADLAR bancos privados", "% TNA")

    # Tasa real ex-post: TPM mensual (TNA/12) - IPC m/m. End-of-month TPM.
    tpm = _bcra("tasa_politica_monetaria", start, end)
    if not tpm.empty and not ipc_mm.empty:
        tpm_eom = _to_monthly_eom(tpm, how="last").rename(columns={"valor": "tpm_tna"})
        tpm_eom["tpm_mm"] = tpm_eom["tpm_tna"] / 100 / 12
        merged = tpm_eom.merge(ipc_mm.rename(columns={"valor": "ipc_mm"}),
                               on="fecha", how="inner")
        merged["tasa_real_mm"] = (merged["tpm_mm"] - merged["ipc_mm"]) * 100
        if not merged.empty:
            df_tr = merged[["fecha", "tasa_real_mm"]].rename(columns={"tasa_real_mm": "valor"})
            _bar(df_tr, "Tasa real ex-post (TPM/12 − IPC m/m)",
                 "% mensual", color_by_sign=True)


# ─────────────────────────────────────────────────────────────────────
# 4. Sector externo / cambiario
# ─────────────────────────────────────────────────────────────────────

def _tcr_base_ultimo(tc_daily: pd.DataFrame, ipc_monthly: pd.DataFrame) -> pd.DataFrame:
    """TC ajustado por inflación, normalizado al último IPC disponible.

    TCR(t) = TC(t) × IPC(último) / IPC(t).
    Lectura: si TCR(t) > TC actual, el peso estaba más devaluado en t (en términos reales);
    si TCR(t) < TC actual, el peso estaba apreciado.
    """
    if tc_daily.empty or ipc_monthly.empty:
        return pd.DataFrame(columns=["fecha", "valor"])
    tc_m = tc_daily.set_index("fecha")["valor"].resample("MS").last().rename("tc")
    ipc = ipc_monthly.set_index("fecha")["valor"].rename("ipc")
    df = pd.concat([tc_m, ipc], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=["fecha", "valor"])
    ipc_last = df["ipc"].iloc[-1]
    df["valor"] = df["tc"] * ipc_last / df["ipc"]
    return df[["valor"]].reset_index()


def _pbi_anual_extrapolado(start: date, end: date) -> pd.DataFrame:
    """PBI anualizado nominal mensual, extrapolado por IPC desde el último trimestre.

    Q4-2024 es el último PBI publicado (lag típico). Para t > Q4-2024 asumimos
    PBI real ≈ constante y escalamos por IPC. Es una aproximación honesta para
    ratios deuda/PBI a 1-2 trimestres adelante.
    """
    pbi_q = _indec("pbi_nominal", start, end)
    ipc = _indec("ipc_nacional", start, end)
    if pbi_q.empty or ipc.empty:
        return pd.DataFrame(columns=["fecha", "valor"])
    last_pbi = pbi_q.iloc[-1]
    last_q_date = last_pbi["fecha"]
    pbi_anual_at_last_q = float(last_pbi["valor"]) * 4  # anualizado
    ipc_s = ipc.set_index("fecha")["valor"]
    ipc_at_last_q = ipc_s.asof(last_q_date)
    if pd.isna(ipc_at_last_q):
        ipc_at_last_q = ipc_s.iloc[0]
    df = ipc.copy()
    df["valor"] = pbi_anual_at_last_q * df["valor"] / ipc_at_last_q
    return df[["fecha", "valor"]]


def render_fx(start: date, end: date) -> None:
    st.subheader("Sector externo y cambiario")
    st.caption(
        "TC oficial vs TCR (base último día = 100), brecha cambiaria, balanza "
        "comercial y cuenta capital y financiera."
    )

    ipc_idx = _indec("ipc_nacional", start, end)

    # ── 1. TC nominal vs TC ajustado por inflación ────────────────────
    tc = _bcra("fx_mayorista", start, end)
    tcr = _tcr_base_ultimo(tc, ipc_idx)
    if not tc.empty and not tcr.empty:
        tc_m = tc.set_index("fecha")["valor"].resample("MS").last().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tc_m["fecha"], y=tc_m["valor"],
                                 name="TC mayorista nominal", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=tcr["fecha"], y=tcr["valor"],
                                 name=f"TCR (en $ de {ipc_idx['fecha'].iloc[-1].strftime('%b-%Y')})",
                                 line=dict(color="#f59e0b", width=2, dash="dash")))
        fig.update_layout(title="Tipo de cambio nominal vs ajustado por inflación",
                          yaxis_title="$/USD", xaxis_title=None,
                          height=400, hovermode="x unified",
                          margin=dict(t=40, b=70, l=10, r=10),
                          legend=dict(orientation="h", y=-0.18, x=0.5,
                                      xanchor="center", yanchor="top"))
        _add_regime_line(fig)
        st.plotly_chart(fig, width="stretch")
        # Métrica: cuánto se apreció/depreció desde el régime break
        tcr_idx = tcr.set_index("fecha")["valor"]
        tc_now = tc_m.set_index("fecha")["valor"].iloc[-1]
        if REGIME_BREAK_TS in tcr_idx.index or any(tcr_idx.index >= REGIME_BREAK_TS):
            tcr_at_break = tcr_idx.asof(REGIME_BREAK_TS)
            apreciacion = (tc_now / tcr_at_break - 1) * 100 if tcr_at_break else None
            if apreciacion is not None:
                st.caption(
                    f"Desde el cambio de régimen (dic-2023): el peso "
                    f"{'se apreció' if apreciacion < 0 else 'se depreció'} "
                    f"{abs(apreciacion):.1f}% en términos reales vs el TC oficial."
                )

    # ── 2. Brecha cambiaria snapshot ──────────────────────────────────
    st.markdown("**Brecha cambiaria — cotizaciones actuales**")
    try:
        snap = _dolar_snapshot()
        oficial = float(snap[snap["casa"] == "oficial"]["venta"].iloc[0])
        cards = []
        for casa, label in [("oficial", "Oficial"), ("blue", "Blue"),
                            ("bolsa", "MEP (Bolsa)"), ("contadoconliqui", "CCL")]:
            row = snap[snap["casa"] == casa]
            if row.empty:
                continue
            venta = float(row["venta"].iloc[0])
            brecha = (venta / oficial - 1) * 100 if casa != "oficial" else 0.0
            cards.append((label, venta, brecha))
        cols = st.columns(len(cards))
        for col, (label, venta, brecha) in zip(cols, cards):
            delta = None if label == "Oficial" else f"{brecha:+.1f}% vs oficial"
            col.metric(label, f"${venta:,.0f}", delta=delta, delta_color="inverse")
    except Exception as exc:
        st.caption(f"dolarapi no disponible: {exc}")

    # ── 2.5 Reservas: brutas vs netas (XLS oficial BCRA semanal) ────
    st.markdown("**Reservas internacionales — brutas vs netas (XLS oficial BCRA)**")
    rn = _reservas_netas(start, end)
    if not rn.empty:
        # Plot 1: brutas + 3 versiones de netas
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rn["fecha"], y=rn["brutas"],
                                 name="Brutas", line=dict(color="#1f77b4", width=2.5),
                                 fill="tozeroy", fillcolor="rgba(31,119,180,0.10)"))
        fig.add_trace(go.Scatter(x=rn["fecha"], y=rn["netas_balance"],
                                 name="Netas balance (XLS, sin ajustes)",
                                 line=dict(color="#9333ea", width=2)))
        fig.add_trace(go.Scatter(x=rn["fecha"], y=rn["netas_tradicional"],
                                 name="Netas tradicional (− swap China)",
                                 line=dict(color="#6366f1", width=2)))
        fig.add_trace(go.Scatter(x=rn["fecha"], y=rn["netas_colega"],
                                 name="Netas pasivos BCRA <1y (incluye BOPREAL)",
                                 line=dict(color="#f59e0b", width=2, dash="dash")))
        fig.add_hline(y=0, line_color="#888", line_width=1)
        fig.update_layout(title="Reservas brutas y 3 versiones de netas (millones USD, semanal)",
                          yaxis_title="Millones USD", xaxis_title=None,
                          height=420, hovermode="x unified",
                          margin=dict(t=40, b=80, l=10, r=10),
                          legend=dict(orientation="h", y=-0.20, x=0.5,
                                      xanchor="center", yanchor="top", font=dict(size=10)))
        _add_regime_line(fig)
        st.plotly_chart(fig, width="stretch")

        last = rn.iloc[-1]
        # Componentes del balance (líneas del XLS, no manuales)
        st.markdown("**Deducciones del balance BCRA** (líneas del XLS oficial — actualización semanal):")
        c1, c2, c3, c4, c5 = st.columns(5)
        bal_items = [
            (c1, "Ctas. Ctes. otras monedas", "ctas_ctes_otras_monedas",
             "Encajes USD del sistema financiero"),
            (c2, "Obligaciones org. internac.", "obligaciones_organismos",
             "BIS + multilaterales"),
            (c3, "Repos pasivos", "repos_pasivos",
             "SEDESA + bancos comerciales"),
            (c4, "Depósitos Tesoro Nacional", "depositos_tesoro",
             "Fondos Tesoro en BCRA (no resta colega)"),
            (c5, "DEG neto (FMI SDR)", "deg_neto",
             "IMF Special Drawing Rights"),
        ]
        for col, lbl, key, _ in bal_items:
            col.metric(lbl, f"−${last[key]:,.0f} M", delta_color="off")

        # Ajustes manuales / dinámicos (no separables del balance)
        st.markdown(
            "**Ajustes adicionales** (no extraíbles del balance):"
        )
        c6, c7 = st.columns(2)
        c6.metric("Swap China (PBoC)", f"−${last['swap_china']:,.0f} M",
                  delta="130 bn CNY / CNY-USD spot", delta_color="off")
        c7.metric("BOPREAL <1 año", f"−${last['bopreal_corto_plazo']:,.0f} M",
                  delta="snapshot manual por serie", delta_color="off")

        st.caption(
            f"**{last['fecha'].strftime('%d-%b-%Y')}**: "
            f"Brutas **${last['brutas']:,.0f}M** → "
            f"Netas balance **${last['netas_balance']:,.0f}M** → "
            f"Netas tradicional **${last['netas_tradicional']:,.0f}M** → "
            f"Netas pasivos BCRA <1y **${last['netas_colega']:,.0f}M**. "
            f"Fuente: serieanuali.xls (BCRA, semanal). Conversión a USD vía TC mayorista del balance."
        )
        st.info(
            "**Notas metodológicas**:  \n"
            "• **Netas balance**: solo líneas extraíbles del XLS oficial. La versión más rigurosa "
            "contablemente, pero diverge de la prensa porque el balance NO separa swap China de "
            "\"Otros pasivos\".  \n"
            "• **Netas tradicional**: agrega el swap China **calculado dinámicamente** como "
            "`130 bn CNY × (1/CNY-USD spot)`. Por eso distintas consultoras reportan $18-19 bn "
            "según el yuan del momento.  \n"
            "• **Netas pasivos BCRA <1y**: metodología \"todos los pasivos USD de corto plazo del "
            "BCRA\". Resta encajes, swap China, repos, obligaciones organismos y **BOPREAL con "
            "vencimiento <1 año**. **No** descuenta depósitos del Tesoro (no es pasivo del BCRA), "
            "swap EEUU ni desembolsos FMI (es deuda del Tesoro, no del BCRA).  \n"
            "• **BOPREAL stock total** (~$12.4 bn USD, Row 83) está agregado en el balance sin "
            "desglose por serie. El snapshot manual estima la porción <1 año."
        )

    # ── 3. Balanza comercial ──────────────────────────────────────────
    st.markdown("**Balanza comercial mensual** (millones USD)")
    exp = _indec("exportaciones_total", start, end)
    imp = _indec("importaciones_total", start, end)
    if not exp.empty and not imp.empty:
        bc = exp.merge(imp, on="fecha", suffixes=("_exp", "_imp"))
        bc["saldo"] = bc["valor_exp"] - bc["valor_imp"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=bc["fecha"], y=bc["saldo"],
                             marker_color=["#6366f1" if v >= 0 else "#f59e0b" for v in bc["saldo"]],
                             name="Saldo (exp − imp)", opacity=0.85), secondary_y=False)
        fig.add_trace(go.Scatter(x=bc["fecha"], y=bc["valor_exp"], name="Exportaciones",
                                 line=dict(color="#1f77b4", width=1.5)), secondary_y=True)
        fig.add_trace(go.Scatter(x=bc["fecha"], y=bc["valor_imp"], name="Importaciones",
                                 line=dict(color="#ff7f0e", width=1.5)), secondary_y=True)
        fig.update_layout(title="Balanza comercial — saldo mensual y flujos",
                          height=380, hovermode="x unified",
                          margin=dict(t=40, b=70, l=10, r=10),
                          legend=dict(orientation="h", y=-0.18, x=0.5,
                                      xanchor="center", yanchor="top"))
        fig.update_yaxes(title_text="Saldo (MUSD)", secondary_y=False)
        fig.update_yaxes(title_text="Exp / Imp (MUSD)", secondary_y=True)
        _add_regime_line(fig)
        st.plotly_chart(fig, width="stretch")

    # ── 4. Cuenta capital y financiera (Balance Cambiario BCRA) ──────
    st.markdown("**Cuenta Capital y Financiera Cambiaria** (millones USD, mensual)")
    cc = _indec("cuenta_capital_financiera", start, end)
    if not cc.empty:
        _bar(cc, "Cuenta Capital y Financiera (positivo = ingreso neto de divisas)",
             "Millones USD", color_by_sign=True)



# ─────────────────────────────────────────────────────────────────────
# 5. Deuda y fiscal (deflactado)
# ─────────────────────────────────────────────────────────────────────

def _bcra_pasivos_remunerados(start: date, end: date) -> pd.DataFrame:
    """Suma diaria pases pasivos + LELIQ/NOTALQ + LEFI. Devuelve serie mensual fin-de-período."""
    components = []
    for name in ("pases_pasivos_bcra", "leliq_notalq", "lefi_cartera_efs"):
        df = _bcra(name, start, end)
        if df.empty:
            continue
        components.append(df.set_index("fecha")["valor"].rename(name))
    if not components:
        return pd.DataFrame(columns=["fecha", "valor"])
    total = pd.concat(components, axis=1).fillna(0).sum(axis=1).rename("valor")
    monthly = total.resample("MS").last().dropna()
    return monthly.reset_index()


def render_debt_fiscal(start: date, end: date) -> None:
    st.subheader("Deuda y fiscal — vista consolidada")
    st.caption(
        "Stock de LECAPs reconstruido desde XLSX de Sec. Finanzas (años completos 2024-2025). "
        "Resultado fiscal ajustado descuenta del primario IMIG los intereses LECAP "
        "capitalizados que el oficial no contabiliza como gasto."
    )

    ipc_idx = _indec("ipc_nacional", start, end)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # ── 1. Deuda total consolidada en pesos: BCRA + Tesoro (todos los instrumentos) ──
    pas_bcra = _bcra_pasivos_remunerados(start, end)
    breakdown = _tesoro_breakdown()
    breakdown = breakdown[(breakdown["fecha"] >= start_ts) & (breakdown["fecha"] <= end_ts)]
    lecap_stock = _lecap_stock()
    lecap_stock = lecap_stock[(lecap_stock["fecha"] >= start_ts) & (lecap_stock["fecha"] <= end_ts)]

    if not breakdown.empty:
        cats_orden = ["BCRA pas. remunerados (pases+LELIQ+LEFI)",
                      "LECAP", "BONCAP", "CER-linked", "TAMAR-linked",
                      "Dollar-linked", "BONTE", "Letra otra", "Otro"]
        colores = {
            "BCRA pas. remunerados (pases+LELIQ+LEFI)": "rgba(31,119,180,0.65)",
            "LECAP": "rgba(214,39,40,0.65)",
            "BONCAP": "rgba(255,127,14,0.65)",
            "CER-linked": "rgba(34,197,94,0.55)",
            "TAMAR-linked": "rgba(148,103,189,0.65)",
            "Dollar-linked": "rgba(140,86,75,0.65)",
            "BONTE": "rgba(127,127,127,0.65)",
            "Letra otra": "rgba(188,189,34,0.5)",
            "Otro": "rgba(23,190,207,0.5)",
        }
        # Construir df ancho con todas las categorías + BCRA
        wide = breakdown.set_index("fecha").drop(columns=["total"], errors="ignore")
        if not pas_bcra.empty:
            bcra_idx = pas_bcra.set_index("fecha")["valor"]
            wide["BCRA pas. remunerados (pases+LELIQ+LEFI)"] = bcra_idx.reindex(wide.index, method="nearest")
        wide = wide.fillna(0) / 1e6  # millones → billones
        # Total
        total = wide.sum(axis=1)

        fig = go.Figure()
        for cat in cats_orden:
            if cat not in wide.columns or wide[cat].sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=wide.index, y=wide[cat], name=cat, stackgroup="one",
                line=dict(width=0), fillcolor=colores.get(cat, "rgba(0,0,0,0.3)"),
                hovertemplate=f"<b>{cat}</b>: %{{y:.1f}} bn $<extra></extra>",
            ))
        fig.add_trace(go.Scatter(x=total.index, y=total.values,
                                 name="Total consolidado", line=dict(color="#111827", width=2),
                                 hovertemplate="<b>Total</b>: %{y:.1f} bn $<extra></extra>"))
        fig.update_layout(title="Deuda total consolidada en pesos (BCRA + Tesoro, por instrumento)",
                          yaxis_title="Billones $ nominales", xaxis_title=None,
                          height=460, hovermode="x unified",
                          margin=dict(t=40, b=110, l=10, r=10),
                          legend=dict(orientation="h", y=-0.22, x=0.5,
                                      xanchor="center", yanchor="top", font=dict(size=10)))
        _add_regime_line(fig)
        st.plotly_chart(fig, width="stretch")

        # Snapshot del último mes
        last_row = wide.iloc[-1]
        last_total = float(last_row.sum())
        breakdown_str = ", ".join(
            f"{cat}: ${val:.1f} bn ({val/last_total*100:.0f}%)"
            for cat, val in last_row.sort_values(ascending=False).items() if val > 0
        )
        st.caption(
            f"**Composición {wide.index[-1].strftime('%b-%Y')}** "
            f"(total ${last_total:.1f} bn nominales): {breakdown_str}"
        )

        st.caption(
            "Esta vista (en pesos) es la lente para inflación. Para la solvencia / "
            "comparación internacional, ver el ratio **Deuda total / PBI** abajo."
        )

    # ── 1.5 Deuda bruta TOTAL / PBI — usando ratios oficiales A.4.7 ──
    st.markdown("**Deuda pública bruta total / PBI nominal** (Sec. Finanzas A.4.7)")
    pbi_oficial = _deuda_pbi_oficial()
    if not pbi_oficial.empty:
        pbi_oficial = pbi_oficial.dropna(subset=["deuda_bruta_pct_pbi"])
        pbi_oficial["pct_total"] = pbi_oficial["deuda_bruta_pct_pbi"] * 100
        pbi_oficial["pct_externa"] = pbi_oficial["deuda_externa_pct_pbi"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pbi_oficial["anio"], y=pbi_oficial["pct_total"],
                             name="Deuda bruta total", marker_color="#7c3aed",
                             text=[f"{v:.0f}%" for v in pbi_oficial["pct_total"]],
                             textposition="outside"))
        fig.add_trace(go.Scatter(x=pbi_oficial["anio"], y=pbi_oficial["pct_externa"],
                                 name="Deuda externa (subset)",
                                 line=dict(color="#f59e0b", width=2.5),
                                 mode="lines+markers", marker=dict(size=6)))
        fig.update_layout(
            title="Deuda bruta Adm. Central como % del PBI nominal (oficial)",
            yaxis_title="% del PBI", xaxis_title=None,
            height=380, hovermode="x unified",
            margin=dict(t=40, b=70, l=10, r=10),
            legend=dict(orientation="h", y=-0.18, x=0.5,
                        xanchor="center", yanchor="top"),
        )
        st.plotly_chart(fig, width="stretch")
        last = pbi_oficial.iloc[-1]
        st.caption(
            f"**{int(last['anio'])}**: deuda bruta = **{last['pct_total']:.1f}%** del PBI; "
            f"deuda externa = **{last['pct_externa']:.1f}%** del PBI. "
            f"Pico 2023 ({pbi_oficial.loc[pbi_oficial['pct_total'].idxmax(), 'pct_total']:.0f}%) "
            f"refleja el shock cambiario de fin de año (deuda USD se infla en pesos). "
            f"Fuente: Sec. Finanzas A.4.7 — ratios oficiales con metodología INDEC."
        )

        # Breakdown por categoría como % PBI (deriva PBI_USD del ratio oficial)
        st.markdown("**Composición de la deuda bruta como % del PBI** (mismo trimestral del breakdown)")
        deuda_total_ext = _deuda_total_extendida()
        breakdown_ext = _deuda_breakdown_extendido()
        if not deuda_total_ext.empty and not breakdown_ext.empty:
            # Para cada año, pbi_usd = deuda_total_anual / ratio_anual
            # Tomamos Q4 de cada año como proxy del fin de año
            ratios_by_year = pbi_oficial.set_index("anio")["deuda_bruta_pct_pbi"]
            deuda_total_ext = deuda_total_ext.copy()
            deuda_total_ext["anio"] = pd.to_datetime(deuda_total_ext["fecha"]).dt.year
            # PBI_USD anual deducido del ratio oficial Q4 de cada año
            q4_deuda = deuda_total_ext[
                pd.to_datetime(deuda_total_ext["fecha"]).dt.month == 12
            ].set_index("anio")["valor_musd"]
            pbi_usd_implicito = (q4_deuda / ratios_by_year).dropna()
            # Para cada fecha trimestral del breakdown, usar el PBI USD del año
            br = breakdown_ext.copy()
            br["anio"] = pd.to_datetime(br["fecha"]).dt.year
            br["pbi_usd"] = br["anio"].map(pbi_usd_implicito.to_dict())
            cat_cols = [c for c in br.columns if c not in ("fecha", "anio", "pbi_usd")]
            for c in cat_cols:
                br[c] = br[c] / br["pbi_usd"] * 100
            br = br.dropna(subset=["pbi_usd"])
            cats_orden = ["Títulos USD (Long Term)", "Títulos $ (Long Term)",
                          "Letras del Tesoro (LP+CP)", "FMI", "BID", "BIRF", "CAF",
                          "Adelantos BCRA", "Avales", "Pago diferido + reestructuración"]
            colores = {
                "Títulos USD (Long Term)": "rgba(214,39,40,0.7)",
                "Títulos $ (Long Term)": "rgba(31,119,180,0.7)",
                "Letras del Tesoro (LP+CP)": "rgba(255,127,14,0.7)",
                "FMI": "rgba(148,103,189,0.8)",
                "BID": "rgba(34,197,94,0.5)",
                "BIRF": "rgba(140,86,75,0.6)",
                "CAF": "rgba(227,119,194,0.6)",
                "Adelantos BCRA": "rgba(127,127,127,0.6)",
                "Avales": "rgba(188,189,34,0.5)",
                "Pago diferido + reestructuración": "rgba(23,190,207,0.5)",
            }
            fig = go.Figure()
            for cat in cats_orden:
                if cat not in br.columns or br[cat].sum() == 0:
                    continue
                fig.add_trace(go.Bar(x=br["fecha"], y=br[cat], name=cat,
                                     marker_color=colores.get(cat, "rgba(0,0,0,0.3)"),
                                     hovertemplate=f"<b>{cat}</b>: %{{y:.1f}}% del PBI<extra></extra>"))
            fig.update_layout(title="Composición de la deuda bruta como % del PBI (trimestral)",
                              yaxis_title="% del PBI", xaxis_title=None,
                              barmode="stack", height=440, hovermode="x unified",
                              margin=dict(t=40, b=110, l=10, r=10),
                              bargap=0.15,
                              legend=dict(orientation="h", y=-0.25, x=0.5,
                                          xanchor="center", yanchor="top",
                                          font=dict(size=10)))
            st.plotly_chart(fig, width="stretch")

    # ── 2. Resultado fiscal: oficial vs ajustado ─────────────────────
    primario_nom = _mecon("resultado_primario_imig", start, end)
    intereses = _intereses_capitalizados()
    intereses = intereses[(intereses["fecha"] >= start_ts) & (intereses["fecha"] <= end_ts)]

    if not primario_nom.empty and not intereses.empty:
        # Alinear: IMIG es a primer día del mes; intereses a fin de mes. Normalizo a primero.
        prim = primario_nom.set_index("fecha")["valor"].rename("primario_nom")
        intc = intereses.set_index("fecha")["intereses_capitalizados_mm"]
        intc.index = intc.index.to_period("M").to_timestamp()  # default: start of period
        intc = intc.rename("intereses_capitalizados")
        df = pd.concat([prim, intc], axis=1)
        df["primario_ajustado"] = df["primario_nom"] - df["intereses_capitalizados"]
        # Deflactar ambas
        ipc = ipc_idx.set_index("fecha")["valor"]
        base = ipc.iloc[-1]
        factor = (ipc / base).reindex(df.index, method="nearest")
        df_real = df.div(factor, axis=0).dropna(subset=["primario_nom"])

        # Acumulado del año (resetea cada enero) — útil para leer si el año cierra con
        # superávit o déficit y cuánto pesa cada mes nuevo.
        df_for_ytd = df_real.copy()
        # Para el ajustado: cuando no hay LECAPs (pre-2024), ajustado = oficial.
        df_for_ytd["primario_ajustado"] = df_for_ytd["primario_ajustado"].fillna(
            df_for_ytd["primario_nom"]
        )
        years = df_for_ytd.index.year
        df_for_ytd["ytd_nom"] = df_for_ytd.groupby(years)["primario_nom"].cumsum()
        df_for_ytd["ytd_aj"] = df_for_ytd.groupby(years)["primario_ajustado"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_real.index, y=df_real["primario_nom"] / 1e6,
                             name="Primario oficial (mensual)", marker_color="#2563eb",
                             opacity=0.55))
        fig.add_trace(go.Bar(x=df_real.index, y=df_real["primario_ajustado"] / 1e6,
                             name="Primario ajustado (mensual)",
                             marker_color="#f59e0b", opacity=0.55))
        fig.add_trace(go.Scatter(x=df_for_ytd.index, y=df_for_ytd["ytd_nom"] / 1e6,
                                 name="Acumulado año oficial",
                                 line=dict(color="#2563eb", width=2.8),
                                 mode="lines+markers", marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=df_for_ytd.index, y=df_for_ytd["ytd_aj"] / 1e6,
                                 name="Acumulado año ajustado",
                                 line=dict(color="#f59e0b", width=2.8, dash="dash"),
                                 mode="lines+markers", marker=dict(size=5)))
        fig.update_layout(title="Resultado primario: oficial vs ajustado (mensual + acumulado del año)",
                          yaxis_title=f"Billones $ de {ipc_idx['fecha'].iloc[-1].strftime('%b-%Y')}",
                          barmode="group", height=460, hovermode="x unified",
                          margin=dict(t=40, b=90, l=10, r=10),
                          legend=dict(orientation="h", y=-0.22, x=0.5,
                                      xanchor="center", yanchor="top", font=dict(size=10)))
        fig.add_hline(y=0, line_color="#888", line_width=1)
        _add_regime_line(fig)
        st.plotly_chart(fig, width="stretch")

        # Métricas resumen del último mes disponible
        last = df_real.dropna(subset=["primario_ajustado"]).iloc[-1] if not df_real.dropna(subset=["primario_ajustado"]).empty else None
        if last is not None:
            mes = df_real.dropna(subset=["primario_ajustado"]).index[-1].strftime("%b-%Y")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Primario oficial ({mes})",
                      f"{last['primario_nom']/1e6:+.2f} bn",
                      delta="real")
            c2.metric("Intereses LECAP capitalizados",
                      f"−{last['intereses_capitalizados']/1e6:.2f} bn",
                      delta="no contabilizados como gasto", delta_color="off")
            c3.metric("Primario AJUSTADO",
                      f"{last['primario_ajustado']/1e6:+.2f} bn",
                      delta=f"vs oficial: {(last['primario_ajustado']-last['primario_nom'])/1e6:+.2f}",
                      delta_color="inverse")

        st.caption(
            "Fórmula: `intereses_capitalizados_t ≈ stock_LECAPs_(t-1) × tasa_mensual_pond_t`. "
            "Tasa ponderada por nominal vivo, solo LECAPs de tasa fija (excluye TAMAR-linked). "
            "Limitación: el XLSX oficial sólo cubre años completos, así que LECAPs emitidas en "
            "2026 no están — el stock decae por vencimientos sin reposición visible. "
            "Reconstruir 2026 requiere scrapear los anuncios individuales de licitación."
        )

    # ── 3. Stock LECAPs detalle (contexto) ──────────────────────────
    if not lecap_stock.empty:
        df = lecap_stock.copy()
        df["stock_billones"] = df["stock_lecap_nominal_mm"] / 1e6
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["fecha"], y=df["stock_billones"],
                             marker_color="#f59e0b", opacity=0.6, name="Stock"))
        fig.add_trace(go.Scatter(x=df["fecha"], y=df["n_lecaps_vivas"],
                                 yaxis="y2", name="N° LECAPs vivas",
                                 line=dict(color="#374151", width=2)))
        fig.update_layout(
            title="Stock LECAPs en pesos (nominal) y cantidad de instrumentos vivos",
            yaxis=dict(title="Billones $ nominales"),
            yaxis2=dict(title="N° LECAPs", overlaying="y", side="right"),
            height=360, hovermode="x unified",
            margin=dict(t=40, b=70, l=10, r=10),
            legend=dict(orientation="h", y=-0.22, x=0.5,
                        xanchor="center", yanchor="top"),
        )
        st.plotly_chart(fig, width="stretch")

    # ── 3.5 Deuda bruta total — breakdown por categoría (serie extendida) ──
    st.markdown("**Composición trimestral de la deuda bruta — Administración Central** (2022→presente)")
    deuda_br = _deuda_breakdown_extendido()
    if not deuda_br.empty:
        long = deuda_br.melt(id_vars=["fecha"], var_name="categoria", value_name="musd")
        fig = go.Figure()
        cats_orden = ["Títulos USD (Long Term)", "Títulos $ (Long Term)",
                      "Letras del Tesoro (LP+CP)", "FMI", "BID", "BIRF", "CAF",
                      "Adelantos BCRA", "Avales", "Pago diferido + reestructuración"]
        colores = {
            "Títulos USD (Long Term)": "rgba(214,39,40,0.7)",
            "Títulos $ (Long Term)": "rgba(31,119,180,0.7)",
            "Letras del Tesoro (LP+CP)": "rgba(255,127,14,0.7)",
            "FMI": "rgba(148,103,189,0.8)",
            "BID": "rgba(34,197,94,0.5)",
            "BIRF": "rgba(140,86,75,0.6)",
            "CAF": "rgba(227,119,194,0.6)",
            "Adelantos BCRA": "rgba(127,127,127,0.6)",
            "Avales": "rgba(188,189,34,0.5)",
            "Pago diferido + reestructuración": "rgba(23,190,207,0.5)",
        }
        for cat in cats_orden:
            sub = long[long["categoria"] == cat]
            if sub.empty or sub["musd"].sum() == 0:
                continue
            fig.add_trace(go.Bar(x=sub["fecha"], y=sub["musd"] / 1000, name=cat,
                                 marker_color=colores.get(cat, "rgba(0,0,0,0.3)"),
                                 width=1000*60*60*24*60,  # 60 días = barras finitas
                                 hovertemplate=f"<b>{cat}</b>: $%{{y:.1f}} bn USD<extra></extra>"))
        fig.update_layout(title="Deuda bruta total por categoría (bn USD, trimestral)",
                          yaxis_title="$ bn USD", xaxis_title=None,
                          bargap=0.1,
                          barmode="stack", height=440, hovermode="x unified",
                          margin=dict(t=40, b=110, l=10, r=10),
                          legend=dict(orientation="h", y=-0.25, x=0.5,
                                      xanchor="center", yanchor="top", font=dict(size=10)))
        st.plotly_chart(fig, width="stretch")
        # Snapshot latest
        last_q = deuda_br.iloc[-1]
        items = [(c, float(last_q[c])) for c in deuda_br.columns if c != "fecha" and last_q[c] > 0]
        items.sort(key=lambda x: -x[1])
        total = sum(v for _, v in items)
        breakdown_str = ", ".join(f"{c}: ${v/1000:.1f} bn ({v/total*100:.0f}%)" for c, v in items)
        st.caption(
            f"**{last_q['fecha'].strftime('%b-%Y')}** "
            f"(total bruto sin avales/diferido ≈ ${total/1000:.0f} bn USD): {breakdown_str}"
        )

    # ── 4. Recaudación (contexto) ─────────────────────────────────────
    recaud_real = _deflate(_mecon("recaudacion_total", start, end), ipc_idx)
    if not recaud_real.empty:
        df = recaud_real.copy()
        df["valor"] = df["valor"] / 1e6
        _line(df, "Recaudación tributaria (real, contexto)",
              f"Billones $ de {ipc_idx['fecha'].iloc[-1].strftime('%b-%Y')}", height=260)


# ─────────────────────────────────────────────────────────────────────
# 6. Sector real
# ─────────────────────────────────────────────────────────────────────

def render_real_sector(start: date, end: date) -> None:
    st.subheader("Sector real")
    st.caption(
        "EMAE = Estimador Mensual de Actividad Económica (proxy mensual del PBI). "
        "Datos directos del XLS oficial de INDEC (más actualizado que datos.gob.ar). "
        "Sectores en índice base 2004=100."
    )

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    def _filtrar(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        return df[(df["fecha"] >= start_ts) & (df["fecha"] <= end_ts)]

    # ── Nivel general ───────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        desest = _filtrar(_emae_general("desest"))
        _line(desest, "EMAE nivel general desestacionalizado (base 2004=100)", "Índice")
    with c2:
        yoy = _filtrar(_emae_general("yoy_orig"))
        if not yoy.empty:
            _bar(yoy, "EMAE general — variación interanual", "% i.a.", color_by_sign=True)

    # ── Desglose por sector económico ───────────────────────────────
    st.markdown("**Desglose por sector económico** (índice base 2004=100)")

    sector_keys_labels = [
        ("industria", "Industria Manufacturera"),
        ("construccion", "Construcción"),
        ("comercio", "Comercio"),
        ("agricultura", "Agro y ganadería"),
        ("mineria", "Minas y canteras"),
        ("electricidad_gas_agua", "Electricidad/gas/agua"),
        ("transporte_comunicaciones", "Transporte y comunic."),
        ("intermediacion_financiera", "Intermediación financiera"),
        ("inmobiliarias", "Inmobiliarias"),
        ("admin_publica", "Adm. pública"),
        ("ensenanza", "Enseñanza"),
        ("salud", "Salud"),
        ("hoteles_restaurantes", "Hoteles y restaurantes"),
        ("pesca", "Pesca"),
        ("servicios_comunitarios", "Servicios comunitarios"),
    ]
    sectores_def = [(k, l, PALETTE[i % len(PALETTE)])
                    for i, (k, l) in enumerate(sector_keys_labels)]

    # Plot 1: niveles overlay (todas las líneas)
    fig = go.Figure()
    sector_data = {}
    for sid, label, color in sectores_def:
        try:
            df = _filtrar(_emae_sector(sid))
        except Exception:
            continue
        if df.empty: continue
        sector_data[sid] = (df, label, color)
        fig.add_trace(go.Scatter(x=df["fecha"], y=df["valor"], name=label,
                                 line=dict(width=1.5, color=color),
                                 hovertemplate=f"<b>{label}</b>: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(
        title="EMAE por sector — niveles (base 2004=100)",
        yaxis_title="Índice", xaxis_title=None,
        height=460, hovermode="x unified",
        margin=dict(t=40, b=110, l=10, r=10),
        legend=dict(orientation="h", y=-0.25, x=0.5,
                    xanchor="center", yanchor="top", font=dict(size=10)),
    )
    _add_regime_line(fig)
    st.plotly_chart(fig, width="stretch")

    # Plot 2: variación interanual por sector — último mes disponible
    st.markdown("**Variación interanual por sector — último mes disponible**")
    rows = []
    for sid, (df, label, _) in sector_data.items():
        if len(df) < 13: continue
        df = df.sort_values("fecha")
        latest = df.iloc[-1]
        year_ago = df.iloc[-13]
        if year_ago["valor"] == 0: continue
        yoy_v = (latest["valor"] / year_ago["valor"] - 1) * 100
        rows.append({"sector": label, "yoy": yoy_v, "fecha": latest["fecha"]})
    if rows:
        df_yoy = pd.DataFrame(rows).sort_values("yoy")
        fecha_label = df_yoy["fecha"].iloc[0].strftime("%b-%Y")
        fig = go.Figure()
        colors = ["#f59e0b" if v < 0 else "#6366f1" for v in df_yoy["yoy"]]
        fig.add_trace(go.Bar(y=df_yoy["sector"], x=df_yoy["yoy"], orientation="h",
                             marker_color=colors,
                             text=[f"{v:+.1f}%" for v in df_yoy["yoy"]],
                             textposition="outside"))
        fig.add_vline(x=0, line_color="#888", line_width=1)
        fig.update_layout(
            title=f"Variación interanual por sector — {fecha_label}",
            xaxis_title="% i.a.", yaxis_title=None,
            height=480, margin=dict(t=40, b=30, l=10, r=10),
        )
        st.plotly_chart(fig, width="stretch")



# ─────────────────────────────────────────────────────────────────────
# 7. Predictor de inflación — Lasso distributed-lag monetario
# ─────────────────────────────────────────────────────────────────────

def render_inflation_predictor(start: date, end: date) -> None:
    st.subheader("Predictor de inflación — modelo monetarista de lags distribuidos")
    st.caption(
        "Sin inercia (no usa IPC pasado). Solo señal monetaria: "
        "lags 1–6 de Base, M1, M2 y M3 como predictores. "
        "Lasso con TimeSeriesSplit selecciona los lags relevantes. "
        "GARCH(1,1) en residuales para intervalos de confianza. "
        "R² ≈ 0.46 — el mejor de todos los modelos evaluados para esta muestra."
    )

    with st.spinner("Estimando modelo monetarista (≈20 s)..."):
        out = _distributed_lag_run("2018-01-01")

    if "error" in out:
        st.error(str(out.get("error")))
        return

    fc   = out["forecast"]
    r2   = out["r2"]
    alpha = out["alpha"]
    n_act = out["n_active"]
    coefs = out["coefs_active"]
    df_raw = out["df_raw"]
    y      = out["y"]
    y_pred = out["y_pred"]

    # ── Métricas del modelo ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("Features activas (Lasso)", f"{n_act} / 24")
    c3.metric("Alpha (regularización)", f"{alpha:.5f}")
    next_date = pd.Timestamp(fc["next_date"])
    c4.metric(
        f"Pronóstico {next_date.strftime('%b-%Y')}",
        f"{fc['pred_next']:.2f}%",
        delta=f"vs {fc['last_ipc']:.2f}% Mar-26",
        delta_color="inverse",
    )

    # ── Feature importance ──────────────────────────────────────────
    st.markdown("**Lags seleccionados por el modelo** (coeficientes estandarizados)")
    top = coefs.head(15)
    top_sorted = top.sort_values()
    colors = ["#f59e0b" if c < 0 else "#2563eb" for c in top_sorted.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=top_sorted.index, x=top_sorted.values,
                         orientation="h", marker_color=colors,
                         text=[f"{v:+.3f}" for v in top_sorted.values],
                         textposition="outside"))
    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.update_layout(
        xaxis_title="Coeficiente (estandarizado)",
        height=420, margin=dict(t=20, b=30, l=10, r=30),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**Azul** (+): más liquidez → más inflación k meses después.  "
        "**Rojo** (−): contracción monetaria → desinflación.  "
        "M2 lags 1–4 dominan la señal. Base monetaria lags largos son negativos "
        "(en este régimen, la base cae mientras el Tesoro absorbe vía LECAPs → desinflación)."
    )

    # ── Backtest: real vs predicho ──────────────────────────────────
    st.markdown("**Backtest in-sample** (modelo ajustado vs IPC real)")
    ipc_real_df = _indec("ipc_nacional", start, end, representation="percent_change")
    if not ipc_real_df.empty:
        ipc_full = ipc_real_df.set_index("fecha")["valor"] * 100
        ipc_full.index = ipc_full.index.to_period("M").to_timestamp()
    else:
        ipc_full = y

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ipc_full.index, y=ipc_full.values,
                             name="IPC real", line=dict(color="#111827", width=2.5),
                             mode="lines+markers", marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred.values,
                             name="Lasso (fitted)", mode="lines",
                             line=dict(color="#2563eb", width=1.8, dash="dash")))
    fig.update_layout(title="IPC m/m real vs predicción in-sample",
                      yaxis_title="% m/m", xaxis_title=None,
                      height=360, hovermode="x unified",
                      margin=dict(t=40, b=70, l=10, r=10),
                      legend=dict(orientation="h", y=-0.18, x=0.5,
                                  xanchor="center", yanchor="top"))
    _add_regime_line(fig)
    st.plotly_chart(fig, width="stretch")

    # ── Pronóstico 4 meses — escenario status quo ──────────────────
    st.markdown("### Pronóstico 4 meses — escenario status quo")
    sc_a = out["forecast"]["scenario_a"]
    last_real = fc["last_ipc"]
    last_date = pd.Timestamp(fc["last_date"])

    # Supuesto del escenario
    asump = fc["scenarios_assumption"]
    with st.expander("Supuestos del escenario"):
        st.markdown("**Status quo** (cada agregado se mantiene en su media de los últimos 6 meses):")
        for a, v in asump["A"].items():
            st.markdown(f"- {a}: **{v:+.1f}%** m/m")

    fig = go.Figure()
    # Actual reciente
    recent = ipc_full.tail(14)
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values,
                             name="IPC real", mode="lines+markers",
                             line=dict(color="#111827", width=2.8),
                             marker=dict(size=6)))
    # IC95 banda
    fig.add_trace(go.Scatter(
        x=pd.concat([sc_a["fecha"], sc_a["fecha"][::-1]]),
        y=pd.concat([sc_a["upper95"], sc_a["lower95"][::-1]]),
        fill="toself", fillcolor="rgba(99,102,241,0.12)",
        line=dict(width=0), name="IC95", showlegend=True, hoverinfo="skip",
    ))
    # Línea de pronóstico
    fig.add_trace(go.Scatter(x=sc_a["fecha"], y=sc_a["pred"],
                             name="Pronóstico (status quo)", mode="lines+markers",
                             line=dict(color="#6366f1", width=2.5),
                             marker=dict(size=7, symbol="diamond")))
    # Línea de referencia = último IPC real
    fig.add_hline(y=last_real, line_dash="dot", line_color="#9ca3af", line_width=1,
                  annotation_text=f"Mar-26: {last_real:.2f}%",
                  annotation_position="right", annotation_font_size=10)
    fig.update_layout(
        title="IPC m/m real + pronóstico 4 meses (escenario status quo)",
        yaxis_title="% m/m", xaxis_title=None,
        height=440, hovermode="x unified",
        margin=dict(t=40, b=80, l=10, r=10),
        legend=dict(orientation="h", y=-0.20, x=0.5,
                    xanchor="center", yanchor="top", font=dict(size=11)),
    )
    _add_regime_line(fig)
    st.plotly_chart(fig, width="stretch")

    # ── Tabla resumen ───────────────────────────────────────────────
    st.markdown("**Tabla de pronósticos**")
    rows = []
    for ra in sc_a.itertuples():
        rows.append({
            "Mes": ra.fecha.strftime("%b-%Y"),
            "Pronóstico": f"{ra.pred:.2f}%",
            "IC 95%": f"[{ra.lower95:.1f}, {ra.upper95:.1f}]",
            "vs Mar-26": "↓" if ra.pred < last_real else "↑",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.info(
        "**Lectura**: el modelo no usa inercia del IPC — la señal viene puramente "
        "de la dinámica monetaria. Asumiendo que cada agregado se mantiene en su "
        "media de los últimos 6 meses, la inflación tiende a bajar gradualmente "
        "en los próximos 4 meses. Las bandas IC95 se abren con el horizonte — "
        "la incertidumbre crece más allá de 2-3 meses."
    )
    st.caption(
        "⚠️ **Efecto ventana móvil**: el modelo usa lags 1–6 de cada agregado. "
        "Cuando un shock monetario sale de esa ventana, su contribución desaparece "
        "y el pronóstico salta al equilibrio implícito por los supuestos. "
        "Ej.: la caída de M3 de enero-2026 (−6.6% m/m) está actuando como ancla "
        "deflacionaria en los primeros horizontes; cuando sale del lag-6 (h≈4 meses) "
        "el pronóstico converge a ~3.5%."
    )
