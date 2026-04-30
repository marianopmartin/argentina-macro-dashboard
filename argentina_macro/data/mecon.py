"""MECON / Tesoro pulls vía datos.gob.ar Series API.

API ref: misma que indec.py (https://apis.datos.gob.ar/series/api/series/).

LIMITACIÓN: el detalle del stock de deuda en pesos por instrumento (LECAPs, BONCAPs,
BONCERes) no está expuesto en la API de series — se publica como Excel en
https://www.argentina.gob.ar/economia/finanzas/deudapublica. Si más adelante se necesita,
implementar descarga + parseo del Excel oficial.
"""

from __future__ import annotations

from datetime import date

from argentina_macro.data.indec import fetch_series

SERIES: dict[str, str] = {
    "resultado_primario_imig": "452.3_RESULTADO_RIO_0_M_18_54",
    "recaudacion_total": "172.3_TL_RECAION_M_0_0_17",
}


def fetch(
    name: str,
    start: date | None = None,
    end: date | None = None,
    representation: str = "value",
):
    if name not in SERIES:
        raise KeyError(f"'{name}' no está en SERIES. Disponibles: {list(SERIES)}")
    return fetch_series(SERIES[name], start, end, representation)
