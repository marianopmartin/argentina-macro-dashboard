"""INDEC pulls vía datos.gob.ar Series API + EMAE directo del XLS oficial.

API ref: https://apis.datos.gob.ar/series/api/series/?ids=<id>
Las series tienen IDs estables; se pueden buscar en https://datos.gob.ar/

Para EMAE: datos.gob.ar suele tener 1-2 meses de lag respecto a la publicación
oficial. Las funciones fetch_emae_* leen directo el XLS de INDEC para tener
el último mes publicado.
"""

from __future__ import annotations

import io
from datetime import date

import httpx
import pandas as pd

BASE_URL = "https://apis.datos.gob.ar/series/api/series/"

# Catálogo curado.
SERIES: dict[str, str] = {
    # IPC base dic-2016=100 (mensual)
    "ipc_nacional": "148.3_INIVELNAL_DICI_M_26",
    "ipc_nucleo": "148.3_INUCLEONAL_DICI_M_19",
    "ipc_regulados": "148.3_IREGULANAL_DICI_M_22",
    # EMAE base 2004=100 (mensual)
    "emae_general": "143.3_NO_PR_2004_A_21",
    "emae_desest": "143.3_NO_PR_2004_A_31",
    "emae_yoy": "143.3_ICE_SERVIA_2004_A_25",
    # EMAE por sector económico (índices base 2004=100, mensual)
    "emae_agricultura": "11.3_ISOM_2004_M_39",
    "emae_pesca": "11.3_VIPAA_2004_M_5",
    "emae_mineria": "11.3_ISD_2004_M_26",
    "emae_industria": "11.3_VMASD_2004_M_23",
    "emae_electricidad_gas_agua": "11.3_ITC_2004_M_21",
    "emae_construccion": "11.3_VMATC_2004_M_12",
    "emae_comercio": "11.3_AGCS_2004_M_41",
    "emae_hoteles_restaurantes": "11.3_P_2004_M_20",
    "emae_transporte_comunicaciones": "11.3_EMC_2004_M_25",
    "emae_intermediacion_financiera": "11.3_IM_2004_M_25",
    "emae_inmobiliarias": "11.3_SEGA_2004_M_48",
    "emae_admin_publica": "11.3_C_2004_M_60",
    "emae_ensenanza": "11.3_CMMR_2004_M_10",
    "emae_salud": "11.3_HR_2004_M_24",
    "emae_servicios_comunitarios": "11.3_TAC_2004_M_60",
    # Agregados monetarios alternativos (publicados por Sec. Programación Macro
    # en datos.gob.ar; complementan la API de BCRA que no expone M3 directo).
    "m3_pesos": "90.1_AMTM3_0_0_31",
    "m3_pesos_y_usd": "90.1_AMTMA_0_0_43",  # M3* incluye depósitos USD valuados en $
    # Sector externo y PBI
    "pbi_nominal": "166.2_PPIB_0_0_3",  # PBI corriente trimestral, millones $
    "exportaciones_total": "74.3_IET_0_M_16",  # mensual, millones USD
    "importaciones_total": "78.3_IIT_0_A_25",  # mensual, millones USD
    "cuenta_capital_financiera": "182.1_TOTAL_CUENRIA_0_M_41",  # mensual, millones USD
}


def fetch_series(
    series_id: str,
    start: date | None = None,
    end: date | None = None,
    representation: str = "value",
) -> pd.DataFrame:
    """Serie temporal desde datos.gob.ar.

    representation: 'value' (nivel) o 'percent_change' (var % período).
    """
    params: dict[str, str] = {
        "ids": series_id,
        "limit": "5000",
        "representation_mode": representation,
        "format": "json",
    }
    if start:
        params["start_date"] = start.isoformat()
    if end:
        params["end_date"] = end.isoformat()
    r = httpx.get(BASE_URL, params=params, timeout=30, follow_redirects=True)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame(columns=["fecha", "valor"])
    df = pd.DataFrame(data, columns=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.sort_values("fecha").reset_index(drop=True)


def fetch(
    name: str,
    start: date | None = None,
    end: date | None = None,
    representation: str = "value",
) -> pd.DataFrame:
    """Fetch por nombre canónico (clave de SERIES)."""
    if name not in SERIES:
        raise KeyError(f"'{name}' no está en SERIES. Disponibles: {list(SERIES)}")
    return fetch_series(SERIES[name], start, end, representation)


# ─────────────────────────────────────────────────────────────────────
# EMAE directo desde XLS de INDEC (más actualizado que datos.gob.ar)
# ─────────────────────────────────────────────────────────────────────

EMAE_URLS = {
    "mensual":   "https://www.indec.gob.ar/ftp/cuadros/economia/sh_emae_mensual_base2004.xls",
    "actividad": "https://www.indec.gob.ar/ftp/cuadros/economia/sh_emae_actividad_base2004.xls",
}

# Mapeo nombre canónico → letra de actividad CLANAE en el Excel de actividad
EMAE_SECTORES_LETRAS = {
    "agricultura": "A", "pesca": "B", "mineria": "C", "industria": "D",
    "electricidad_gas_agua": "E", "construccion": "F", "comercio": "G",
    "hoteles_restaurantes": "H", "transporte_comunicaciones": "I",
    "intermediacion_financiera": "J", "inmobiliarias": "K",
    "admin_publica": "L", "ensenanza": "M", "salud": "N",
    "servicios_comunitarios": "O",
}

_MES_NUM = {"Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,
            "Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12}


def _download_emae_xls(kind: str = "mensual") -> bytes:
    url = EMAE_URLS[kind]
    r = httpx.get(url, timeout=60, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content


def fetch_emae_general(serie: str = "desest") -> pd.DataFrame:
    """EMAE nivel general desde XLS oficial INDEC (sh_emae_mensual_base2004).

    serie:
      - 'orig'      → serie original
      - 'desest'    → desestacionalizada
      - 'tendencia' → tendencia-ciclo
      - 'yoy_orig'  → var % i.a. de la serie original
      - 'mm_desest' → var % m/m de desestacionalizada
      - 'mm_tend'   → var % m/m de tendencia-ciclo

    Devuelve DataFrame con columnas fecha (datetime) y valor.
    """
    raw = _download_emae_xls("mensual")
    df = pd.read_excel(io.BytesIO(raw), sheet_name="EMAE", header=None)
    # Filtrar filas con mes (col 1) y año en col 0 (forward-fill)
    df = df.iloc[:, :8].copy()
    df.columns = ["anio", "mes", "orig", "yoy_orig", "desest", "mm_desest",
                  "tendencia", "mm_tend"]
    df["anio"] = df["anio"].ffill()
    df = df[df["mes"].isin(_MES_NUM.keys())].copy()
    df["fecha"] = pd.to_datetime({
        "year": df["anio"].astype(int),
        "month": df["mes"].map(_MES_NUM),
        "day": 1,
    })
    col_map = {"orig":"orig", "desest":"desest", "tendencia":"tendencia",
               "yoy_orig":"yoy_orig", "mm_desest":"mm_desest", "mm_tend":"mm_tend"}
    if serie not in col_map:
        raise KeyError(f"serie '{serie}' inválida. Disponibles: {list(col_map)}")
    out = df[["fecha", col_map[serie]]].rename(columns={col_map[serie]: "valor"})
    return out.dropna().reset_index(drop=True)


def fetch_emae_sector(sector: str) -> pd.DataFrame:
    """EMAE por sector económico desde XLS oficial INDEC (sh_emae_actividad_base2004).

    sector: clave en EMAE_SECTORES_LETRAS o letra CLANAE directa (A..O).
    Devuelve nivel original (índice base 2004=100), mensual.
    """
    letra = EMAE_SECTORES_LETRAS.get(sector, sector).upper()
    raw = _download_emae_xls("actividad")
    df = pd.read_excel(io.BytesIO(raw), sheet_name="Tabla Letras", header=None)
    # Headers en fila 2: col 2 en adelante son sectores con prefix "X - ..."
    headers = df.iloc[2, 2:].astype(str).tolist()
    target_idx = None
    for i, h in enumerate(headers):
        h_up = h.strip().upper()
        if h_up.startswith(f"{letra} -") or h_up.startswith(f"{letra} - ") or h_up.startswith(f"{letra}-"):
            target_idx = i + 2
            break
    if target_idx is None:
        raise KeyError(f"sector '{sector}' (letra {letra}) no encontrado")

    body = df.iloc[5:].copy()
    body[0] = body[0].ffill()
    valid = body[1].isin(_MES_NUM.keys())
    body = body[valid]
    body["fecha"] = pd.to_datetime({
        "year": pd.to_numeric(body[0], errors="coerce").astype("Int64"),
        "month": body[1].map(_MES_NUM),
        "day": 1,
    }, errors="coerce")
    body = body.dropna(subset=["fecha"])
    out = body[["fecha", target_idx]].rename(columns={target_idx: "valor"})
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce")
    return out.dropna().reset_index(drop=True)
