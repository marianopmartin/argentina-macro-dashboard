"""BCRA Estado Resumido de Activos y Pasivos — parser del XLS oficial.

Fuente: https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/serieanuali.xls
- Una hoja por año (2002–presente)
- Cada hoja tiene snapshots semanales del balance: 7, 15, 23 y fin de mes
- Valores en MILES de pesos a precios de cada fecha (no constantes)
- Incluye TC USD/$ del día para convertir a USD

Mapeo de filas clave (verificado para sheet 2026):
  Row 7  → Reservas brutas (gold + currency + deposits + others)
  Row 8  → Oro (incluido en brutas)
  Row 9  → Foreign Currency
  Row 10 → Deposits to be realized (encajes USD efectivamente disponibles)
  Row 65 → Cuentas corrientes en otras monedas (encajes USD del sistema financiero)
  Row 67 → Depósitos del Gobierno Nacional en USD
  Row 73 → DEG (IMF SDR — neto de contra-cuenta)
  Row 78 → Obligaciones con Organismos Internacionales (incluye BIS y otros)
  Row 92 → Pasivos por operaciones de repo
  Row 96 → Otros pasivos (incluye swap con China en parte, BOPREAL, etc.)
  Row 51/106 → TC USD/$

LIMITACIÓN: Algunas componentes de la "metodología tradicional" usada en prensa NO
son directamente extraíbles del XLS porque vienen agregadas (e.g., swap China + encajes
juntos en Row 65 según interpretación). Combinar XLS + snapshots manuales para mejor
fidelidad cuando se necesite el desglose preciso.
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path("data/raw/bcra_balance")
URL = "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/serieanuali.xls"

# Mapeo de etiquetas (subcadena en el cell de la columna 0) → nombre canónico
# Robusto a cambios menores en el archivo.
ROW_MATCHERS: dict[str, str] = {
    "GOLD, CURRENCY, DEPOSITS TO BE REALIZED": "reservas_brutas",
    "Gold (net of allowances)": "oro",
    "CURRENT ACCOUNTS IN  OTHERS CURRENCIES": "ctas_ctes_otras_monedas",
    "DEPOSITS FROM NATIONAL ARGENTINE": "depositos_tesoro",
    "IMF SPECIAL DRAWING RIGHTS": "deg_neto",
    "OBLIGATIONS WITH INTERNATIONAL AGENCIES": "obligaciones_organismos",
    "DUE TO REPO TRANSACTIONS": "repos_pasivos",
    "OTHER LIABILITIES": "otros_pasivos",
    "Rate of Exchange": "tc_usd",
}


def _download() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / "serieanuali.xls"
    if cached.exists():
        # Cache válido por 24h (el archivo se actualiza semanal pero damos margen)
        import time
        age_hours = (time.time() - cached.stat().st_mtime) / 3600
        if age_hours < 24:
            return cached
    r = httpx.get(URL, timeout=120, follow_redirects=True, verify=False,
                  headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    cached.write_bytes(r.content)
    return cached


def _parse_year(xls: pd.ExcelFile, year: int) -> pd.DataFrame:
    """Parse single year sheet. Returns long DataFrame: fecha, concept, value_musd."""
    sheet = str(year)
    if sheet not in xls.sheet_names:
        return pd.DataFrame()
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    # Encontrar la fila de fechas (típicamente fila 3 o cerca)
    fechas_row_idx = None
    for i in range(10):
        row = df.iloc[i]
        # detectar fila de fechas: muchos datetimes/timestamps en columnas 1+
        date_count = sum(
            1 for v in row.iloc[1:]
            if isinstance(v, (pd.Timestamp, dt.datetime, dt.date))
        )
        if date_count >= 4:
            fechas_row_idx = i
            break
    if fechas_row_idx is None:
        return pd.DataFrame()
    fechas = [pd.Timestamp(v) if isinstance(v, (dt.datetime, dt.date)) else v
              for v in df.iloc[fechas_row_idx, 1:].tolist()]

    # Para cada matcher, encontrar la fila y extraer valores
    rows_out = []
    label_col = df.iloc[:, 0].astype(str)
    for matcher, canonical in ROW_MATCHERS.items():
        # Match: primera fila donde label contenga el matcher
        mask = label_col.str.contains(matcher, regex=False, na=False)
        if not mask.any():
            continue
        idx = mask.idxmax()
        values = df.iloc[idx, 1:].tolist()
        for fecha, val in zip(fechas, values):
            if not isinstance(fecha, pd.Timestamp) or pd.isna(val):
                continue
            try:
                rows_out.append({
                    "fecha": fecha,
                    "concepto": canonical,
                    "valor_kpesos": float(val) if canonical != "tc_usd" else None,
                    "tc_usd": float(val) if canonical == "tc_usd" else None,
                })
            except (ValueError, TypeError):
                continue
    return pd.DataFrame(rows_out)


def parse_balance_history(years: tuple[int, ...] = tuple(range(2018, 2027))) -> pd.DataFrame:
    """Parsea múltiples años. Devuelve long DataFrame con fecha, concepto, valor_musd.

    valor_musd = valor_kpesos / (tc_usd × 1000) — conversión a millones de USD usando
    el TC mayorista del mismo día publicado en el balance.
    """
    path = _download()
    xls = pd.ExcelFile(path)
    frames = [_parse_year(xls, y) for y in years]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        return df

    # Pivotear: para cada fecha, separar TC del resto
    tc = df[df["concepto"] == "tc_usd"][["fecha", "tc_usd"]].drop_duplicates("fecha")
    main = df[df["concepto"] != "tc_usd"][["fecha", "concepto", "valor_kpesos"]]
    out = main.merge(tc, on="fecha", how="left")
    out["valor_musd"] = out["valor_kpesos"] / (out["tc_usd"] * 1000)
    return out[["fecha", "concepto", "valor_kpesos", "tc_usd", "valor_musd"]].sort_values("fecha")


def latest_snapshot_musd() -> pd.DataFrame:
    """Snapshot más reciente convertido a MUSD: una fila por concepto."""
    df = parse_balance_history(years=(2026,))
    if df.empty:
        return df
    last_date = df["fecha"].max()
    return df[df["fecha"] == last_date].copy()


def reservas_netas_balance(years: tuple[int, ...] = tuple(range(2018, 2027))) -> pd.DataFrame:
    """Computa reservas netas usando exclusivamente filas del balance del BCRA.

    netas_balance = reservas_brutas
                  − ctas_ctes_otras_monedas
                  − obligaciones_organismos
                  − repos_pasivos
                  − depositos_tesoro

    Esta versión es self-consistent (todo del mismo balance, mismo TC).
    Difiere del cálculo "tradicional" de prensa porque ese desagrega encajes vs swap
    China dentro de ctas_ctes_otras_monedas, y suele incluir items adicionales que
    están dispersos en 'Otros pasivos' del balance.

    Devuelve DataFrame con fecha, brutas, deducciones por concepto, netas — todo en MUSD.
    """
    df = parse_balance_history(years)
    if df.empty:
        return df
    wide = df.pivot_table(index="fecha", columns="concepto",
                          values="valor_musd", aggfunc="first").reset_index()
    # Asegurar columnas existentes
    for c in ("reservas_brutas", "ctas_ctes_otras_monedas", "obligaciones_organismos",
              "repos_pasivos", "depositos_tesoro", "deg_neto", "otros_pasivos"):
        if c not in wide.columns:
            wide[c] = 0.0
    wide["deducciones_balance"] = (
        wide["ctas_ctes_otras_monedas"]
        + wide["obligaciones_organismos"]
        + wide["repos_pasivos"]
        + wide["depositos_tesoro"]
    )
    wide["netas_balance"] = wide["reservas_brutas"] - wide["deducciones_balance"]
    return wide.sort_values("fecha").reset_index(drop=True)
