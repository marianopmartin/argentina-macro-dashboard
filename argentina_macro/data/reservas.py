"""Reservas netas: cálculo derivado del Estado Resumido del BCRA (XLS oficial semanal).

**Fuente primaria**: `bcra_balance.parse_balance_history` lee el XLS oficial
serieanuali.xls — actualización semanal, snapshots los días 7, 15, 23 y fin de mes,
todos los componentes en miles de pesos al TC del día. Se convierte a MUSD usando el
TC publicado en el mismo balance.

**Lo que el balance da limpio**:
  - Reservas brutas (Row 7)
  - Cuentas corrientes en otras monedas (Row 65) → encajes USD del sistema financiero
  - Obligaciones con organismos internacionales (Row 78) → BIS + IMF + multilaterales
  - Repos pasivos (Row 92) → SEDESA + bancos comerciales (sin desglose)
  - Depósitos del Gobierno Nacional (Row 67)
  - DEG neto (Row 73 menos contra-cuenta)

**Lo que NO está separable en el balance** (ajustes adicionales):
  - **Swap con China**: vive dentro de "Otros pasivos" (Row 96, agregado).
    Se calcula DINÁMICAMENTE como `130_000 M CNY × (1/CNY-USD del día)`.
    La línea total es 130 bn yuanes; su valor en USD varía con el tipo de cambio CNY/USD
    (de ahí que diferentes consultoras reporten $18-19 bn según el yuan del momento).
  - **Swap EEUU** (anuncio Bessent nov-2025): también dentro de "Otros pasivos".
    Manual (snapshot fijo ~$2.5 bn).
  - **Desembolsos netos FMI desde fecha base**: solo aplica para metodología FMI.
    Manual.

**Las tres versiones de netas** que devolvemos:

1. `netas_balance` (la más rigurosa, todo del XLS):
       brutas − ctas_ctes_otras_monedas − obligaciones_organismos
              − repos_pasivos − depositos_tesoro

2. `netas_tradicional` (alineada con press, agrega ajuste manual swap China):
       netas_balance − swap_china_estimado

3. `netas_fmi` (más estricta, agrega ajustes adicionales):
       netas_tradicional − swap_eeuu − desembolsos_fmi_neto
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from argentina_macro.data import bcra_balance


# Línea total swap PBoC: 130 bn yuanes (renovada 2023).
# Para 2023H1 se asume parcialmente activada (~$5bn equivalentes).
SWAP_CHINA_LINEA_CNY_BN = 130
SWAP_CHINA_ACTIVACION_PARCIAL: dict[str, float] = {
    "2023-06-30": 0.30,  # ~30% activado
    "2023-09-30": 0.95,  # casi total
    "2023-12-31": 1.00,  # 100% activado en adelante
}


# Ajustes manuales para componentes no separables del balance ni computables
# dinámicamente (swap EEUU, desembolso FMI neto desde fecha base).
AJUSTES_MANUALES: dict[str, dict[str, float]] = {
    "2023-06-30": {"swap_eeuu": 0, "fmi_desembolso_neto": 0},
    "2023-12-31": {"swap_eeuu": 0, "fmi_desembolso_neto": 0},
    "2024-12-31": {"swap_eeuu": 0, "fmi_desembolso_neto": 0},
    "2025-04-30": {"swap_eeuu": 0, "fmi_desembolso_neto": 7000},   # nuevo programa FMI
    "2025-09-30": {"swap_eeuu": 0, "fmi_desembolso_neto": 7500},
    "2025-11-25": {"swap_eeuu": 2500, "fmi_desembolso_neto": 8000},
    "2026-02-10": {"swap_eeuu": 2500, "fmi_desembolso_neto": 8300},
}


def _cny_usd_history(start: date, end: date) -> pd.Series:
    """Tipo de cambio CNY por USD desde Yahoo Finance (ticker CNY=X).
    Devuelve Series indexada por fecha."""
    df = yf.download("CNY=X", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].squeeze() if hasattr(df["Close"], "squeeze") else df["Close"]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _swap_china_usd_on_index(idx: pd.DatetimeIndex) -> pd.Series:
    """Computa swap China en MUSD para cada fecha del índice.

    swap_china_usd_M = 130_000 × activacion(t) / CNY_USD(t)
    """
    cny_full = _cny_usd_history(
        idx.min().date() if len(idx) > 0 else date(2023, 1, 1),
        (idx.max() + pd.Timedelta(days=2)).date() if len(idx) > 0 else date.today(),
    )
    if cny_full.empty:
        # Fallback: yuan ~7.0 si no hay data
        return pd.Series(SWAP_CHINA_LINEA_CNY_BN * 1000 / 7.0, index=idx)
    # Activación parcial interpolada
    act_snap = pd.Series({
        pd.Timestamp(f): v for f, v in SWAP_CHINA_ACTIVACION_PARCIAL.items()
    }).sort_index()
    # Para fechas posteriores a último snapshot, asumir 100%
    last_act = act_snap.iloc[-1]
    full_idx = idx.union(act_snap.index).sort_values()
    act = act_snap.reindex(full_idx).interpolate(method="time").ffill().bfill()
    act = act.reindex(idx)

    # CNY/USD: reindexar al idx, ffill (rates fines de semana usan el último viernes)
    cny_aligned = cny_full.reindex(idx.union(cny_full.index)).sort_index().ffill().bfill()
    cny_at_idx = cny_aligned.reindex(idx)

    swap_usd_m = SWAP_CHINA_LINEA_CNY_BN * 1000 * act / cny_at_idx
    return swap_usd_m.fillna(SWAP_CHINA_LINEA_CNY_BN * 1000 / 7.0)


def _ajustes_on_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpola ajustes manuales (swap EEUU, FMI) y computa swap China dinámicamente
    desde CNY/USD."""
    snap_rows = []
    for f, comp in AJUSTES_MANUALES.items():
        snap_rows.append({"fecha": pd.Timestamp(f), **comp})
    snap = pd.DataFrame(snap_rows).sort_values("fecha")
    out = pd.DataFrame(index=idx)
    out.index.name = "fecha"
    for col in ("swap_eeuu", "fmi_desembolso_neto"):
        s = snap.set_index("fecha")[col].reindex(idx.union(snap["fecha"])).sort_index()
        s = s.interpolate(method="time").ffill().bfill()
        out[col] = s.reindex(idx)
    # Swap China dinámico
    out["swap_china"] = _swap_china_usd_on_index(idx)
    return out


def reservas_netas_history(
    start: date, end: date,
) -> pd.DataFrame:
    """Serie semanal con reservas brutas + 3 versiones de netas + componentes.

    Columnas:
      fecha, brutas, ctas_ctes_otras_monedas, obligaciones_organismos, repos_pasivos,
      depositos_tesoro, deg_neto, swap_china (manual), swap_eeuu (manual),
      fmi_desembolso_neto (manual), netas_balance, netas_tradicional, netas_fmi.

    Frecuencia: semanal (snapshots BCRA del 7, 15, 23 y fin de mes).
    """
    years = tuple(range(start.year, end.year + 1))
    bal = bcra_balance.reservas_netas_balance(years=years)
    if bal.empty:
        return pd.DataFrame()
    bal = bal[(bal["fecha"] >= pd.Timestamp(start)) & (bal["fecha"] <= pd.Timestamp(end))]
    if bal.empty:
        return pd.DataFrame()
    # Sumar ajustes manuales
    ajustes = _ajustes_on_index(pd.DatetimeIndex(bal["fecha"]))
    df = bal.merge(ajustes.reset_index(), on="fecha", how="left")
    # Renombrar reservas_brutas → brutas para coherencia con código existente
    df = df.rename(columns={"reservas_brutas": "brutas"})
    df["netas_tradicional"] = df["netas_balance"] - df["swap_china"]
    df["netas_fmi"] = df["netas_tradicional"] - df["swap_eeuu"] - df["fmi_desembolso_neto"]
    return df.sort_values("fecha").reset_index(drop=True)
