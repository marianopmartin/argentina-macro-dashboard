"""BCRA pulls: monetarias (base, agregados, pasivos remunerados, reservas, tasas, FX).

API ref: https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias
"""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

BASE_URL = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"

# Catálogo curado de variables clave. Para descubrir más: fetch_catalog().
VARIABLES: dict[str, int] = {
    "base_monetaria": 15,
    "reservas_intl": 1,
    "fx_mayorista": 5,
    "badlar_privados": 7,
    "m2_yoy_privado_30d": 25,
    "m2_nivel": 109,
    "pases_pasivos_bcra": 152,
    "leliq_notalq": 155,
    "tasa_politica_monetaria": 161,
    "lefi_cartera_efs": 196,
    # Componentes para construir M3 diario (M3 = M2 + PF_privado_no_CER + PF_privado_CER)
    "plazo_fijo_priv_no_cer": 96,
    "plazo_fijo_priv_cer": 97,
}


def fetch_catalog() -> pd.DataFrame:
    """Catálogo completo de variables monetarias del BCRA."""
    r = httpx.get(BASE_URL, params={"limit": 1000}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["results"])


_PAGE_LIMIT = 3000  # tope por request del API BCRA


def fetch_series(variable_id: int, start: date, end: date) -> pd.DataFrame:
    """Serie temporal de una variable BCRA. Columnas: fecha (datetime), valor (float).

    Pagina con offset cuando el rango excede el tope por request.
    """
    url = f"{BASE_URL}/{variable_id}"
    chunks: list[pd.DataFrame] = []
    offset = 0
    while True:
        r = httpx.get(
            url,
            params={
                "desde": start.isoformat(),
                "hasta": end.isoformat(),
                "limit": _PAGE_LIMIT,
                "offset": offset,
            },
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        if not results or not results[0].get("detalle"):
            break
        df = pd.DataFrame(results[0]["detalle"])
        chunks.append(df)
        total = payload.get("metadata", {}).get("resultset", {}).get("count", len(df))
        offset += len(df)
        if offset >= total or len(df) < _PAGE_LIMIT:
            break
    if not chunks:
        return pd.DataFrame(columns=["fecha", "valor"])
    out = pd.concat(chunks, ignore_index=True)
    out["fecha"] = pd.to_datetime(out["fecha"])
    return out.sort_values("fecha").reset_index(drop=True)[["fecha", "valor"]]


def fetch(name: str, start: date, end: date) -> pd.DataFrame:
    """Fetch por nombre canónico (clave de VARIABLES)."""
    if name not in VARIABLES:
        raise KeyError(f"'{name}' no está en VARIABLES. Disponibles: {list(VARIABLES)}")
    return fetch_series(VARIABLES[name], start, end)


def fetch_m3(start: date, end: date) -> pd.DataFrame:
    """M3 diario = M2 + Plazo fijo privado (no CER) + Plazo fijo privado CER.

    Disponible en frecuencia diaria directo de la API del BCRA, a diferencia del M3
    de datos.gob.ar que solo publica mensual con lag de ~2 meses.
    """
    m2 = fetch_series(VARIABLES["m2_nivel"], start, end).set_index("fecha")["valor"]
    pf_no_cer = fetch_series(VARIABLES["plazo_fijo_priv_no_cer"], start, end).set_index("fecha")["valor"]
    pf_cer = fetch_series(VARIABLES["plazo_fijo_priv_cer"], start, end).set_index("fecha")["valor"]
    total = m2.add(pf_no_cer, fill_value=0).add(pf_cer, fill_value=0)
    return total.dropna().reset_index().rename(columns={"valor": "valor"})
