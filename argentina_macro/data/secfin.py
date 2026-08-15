"""Secretaría de Finanzas: colocaciones de deuda XLSX scraper.

Datos oficiales de emisiones del Tesoro Nacional. Permite reconstruir el stock vivo
de LECAPs y la tasa promedio ponderada — insumo para:
  - panel de deuda absorbente consolidada (BCRA pasivos + LECAPs)
  - cálculo de intereses capitalizados (resultado primario ajustado)

Fuente: https://www.argentina.gob.ar/economia/finanzas/deudapublica/colocacionesdedeuda
URL pattern: https://www.argentina.gob.ar/sites/default/files/colocaciones_31-12-{year}{suffix}.xlsx

Cobertura: archivo anual completo. Para datos del año en curso (parciales) habría
que scrapear los anuncios individuales de licitación — pendiente.

Cache local en data/raw/secfin/ (git-ignored).
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/raw/secfin")
URL_BASE = "https://www.argentina.gob.ar/sites/default/files"
NOTICIAS_BASE = "https://www.argentina.gob.ar/noticias"

# Sec. Finanzas a veces re-publica con sufijo (_0, _1) tras revisión. Mapeo verificado.
KNOWN_FILES: dict[int, str] = {
    2024: "colocaciones_31-12-2024.xlsx",
    2025: "colocaciones_31-12-2025_0.xlsx",
}

# Datos trimestrales de la deuda pública (todas las categorías: Tesoro + Org. Internacionales
# + FMI + BCRA Adelantos). Fuente: Sec. Finanzas. URL pattern: deuda_publica_DD-MM-YYYY.xlsx
KNOWN_DEUDA_FILES: dict[tuple[int, int], str] = {
    (2022, 4): "deuda_publica_31-12-2022.xlsx",
    (2023, 1): "deuda_publica_31-03-2023.xlsx",
    (2023, 2): "deuda_publica_30-06-2023.xlsx",
    (2023, 3): "deuda_publica_30-09-2023.xlsx",
    (2023, 4): "deuda_publica_31-12-2023.xlsx",
    (2024, 1): "deuda_publica_31-03-2024_0.xlsx",
    (2024, 2): "deuda_publica_30-06-2024.xlsx",
    (2024, 3): "deuda_publica_30-09-2024.xlsx",
    (2024, 4): "deuda_publica_31-12-2024.xlsx",
    (2025, 1): "deuda_publica_31-03-2025.xlsx",
    (2025, 2): "deuda_publica_30-06-2025.xlsx",
    (2025, 3): "deuda_publica_30-09-2025.xlsx",
    (2025, 4): "deuda_publica_31-12-2025.xlsx",
}

# Slugs de los anuncios oficiales de licitaciones 2026 (años posteriores se agregan acá).
# El XLSX oficial cierra en diciembre, así que las del año en curso se scrapean post-hoc.
LICITACIONES_SLUGS_2026: list[str] = [
    "resultado-de-la-licitacion-de-instrumentos-del-tesoro-nacional-denominados-en-pesos-y-0",
    "resultado-de-la-licitacion-de-instrumentos-del-tesoro-nacional-en-pesos-y-en-dolares",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-en",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-0",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-5",
    "resultado-de-licitacion-de-instrumentos-del-tesoro-nacional-denominados-en-pesos-y-dolares",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-2",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-3",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-4",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-8",
    "resultado-de-la-licitacion-por-efectivo-de-instrumentos-del-tesoro-nacional-denominados-10",
]

_TASA_RE = re.compile(r"capitalizable\s+([\d,\.]+)\s*%", re.IGNORECASE)
_FECHA_RE = re.compile(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(20\d\d)", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b([STM]\d{1,2}[A-Z]\d)\b")
_VENC_RE = re.compile(
    r"VENCIMIENTO\s+(\d{1,2})\s+DE\s+(\w+)\s+(?:DE\s+)?(20\d\d)", re.IGNORECASE
)
_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def _download_xlsx(year: int) -> Path:
    """Descarga el XLSX si no está cacheado. Devuelve el path local."""
    fname = KNOWN_FILES.get(year)
    if not fname:
        raise ValueError(f"No conozco URL del XLSX para {year}. Agregar a KNOWN_FILES.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / fname
    if cached.exists():
        return cached
    r = httpx.get(f"{URL_BASE}/{fname}", timeout=60, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    cached.write_bytes(r.content)
    return cached


def _read_sheet(year: int, sheet: str) -> pd.DataFrame:
    """Lee una sheet limpiando título + notas finales."""
    path = _download_xlsx(year)
    df = pd.read_excel(path, sheet_name=sheet, header=2)
    df = df.dropna(subset=[df.columns[0]])
    df = df[~df.iloc[:, 0].astype(str).str.startswith("(", na=False)]
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    rename = {
        "Nombre del Instrumento": "instrumento",
        "Fecha de emisión": "fecha_emision",
        "Vencimiento": "fecha_vencimiento",
        "Cupón": "cupon",
        "Amortización": "amortizacion",
        "Tipo Moneda (*)": "tipo_moneda",
        "Valor Nominal": "valor_nominal_mm",
        "Valor Efectivo": "valor_efectivo_mm",
        "Precio de emisión c/1000": "precio_c1000",
        "Fecha colocación/      liquidación": "fecha_colocacion",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("fecha_emision", "fecha_vencimiento", "fecha_colocacion"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _parse_tasa_mensual(cupon: object) -> float | None:
    """Extrae la tasa efectiva mensual capitalizable del campo Cupón.
    Devuelve fracción (0.055 para 5,5%) o None si no es tasa fija."""
    if not isinstance(cupon, str):
        return None
    m = _TASA_RE.search(cupon)
    if not m:
        return None
    return float(m.group(1).replace(",", ".")) / 100


def parse_letras(years: tuple[int, ...] = (2024, 2025)) -> pd.DataFrame:
    """Concatena las 'Letras' (LECAPs y variantes) de varios años."""
    return pd.concat([_read_sheet(y, "Letras") for y in years], ignore_index=True)


def parse_bonos(years: tuple[int, ...] = (2024, 2025)) -> pd.DataFrame:
    """Concatena los 'Bonos' (BONCAPs, BONCERes) de varios años."""
    return pd.concat([_read_sheet(y, "Bonos") for y in years], ignore_index=True)


def _categorize_instrumento(name: str) -> str:
    """Asigna categoría. Maneja tanto el formato compacto del XLSX ('LECAP/$/...')
    como el formato largo de los anuncios ('LETRA DEL TESORO NACIONAL CAPITALIZABLE...').

    Orden importa: las variantes más específicas (TAMAR, CER, Dólar Linked) primero.
    """
    s = (name or "").upper()
    if not s:
        return "Otro"
    # Variantes más específicas primero
    if "TAMAR" in s:
        return "TAMAR-linked"
    if "CER" in s or s.startswith("BONCER") or s.startswith("LECER"):
        return "CER-linked"
    if "DOLAR LINKED" in s or "DÓLAR LINKED" in s or "VINCULAD" in s or s.startswith("LELINK"):
        return "Dollar-linked"
    # LECAP / BONCAP en formato largo o corto
    if s.startswith("LECAP") or (s.startswith("LETRA") and "CAPITALIZABLE" in s and "PESOS" in s):
        return "LECAP"
    if s.startswith("BONCAP") or (s.startswith("BONO") and "CAPITALIZABLE" in s and "PESOS" in s):
        return "BONCAP"
    if s.startswith("BONTE") or s.startswith("BONO"):
        return "BONTE"
    if s.startswith("LETRA"):
        return "Letra otra"
    return "Otro"


def lecaps_pesos(years: tuple[int, ...] = (2024, 2025)) -> pd.DataFrame:
    """Solo LECAPs en pesos, con tasa mensual parseada."""
    df = parse_letras(years)
    df = df[df["instrumento"].astype(str).str.startswith("LECAP", na=False)]
    df = df[df.get("tipo_moneda", "") == "MONEDA NACIONAL"]
    df = df.dropna(subset=["fecha_emision", "fecha_vencimiento", "valor_nominal_mm"])
    df["tasa_mensual"] = df["cupon"].map(_parse_tasa_mensual)
    return df.reset_index(drop=True)


def all_tesoro_pesos(years: tuple[int, ...] = (2024, 2025)) -> pd.DataFrame:
    """Todos los instrumentos del Tesoro en pesos (Letras + Bonos), categorizados.

    Excluye Letras ISP (intra sector público — no constituyen deuda con el público)
    y dolarizados (van por separado en all_tesoro_usd).
    """
    letras = parse_letras(years).assign(_origen="letra")
    bonos = parse_bonos(years).assign(_origen="bono")
    df = pd.concat([letras, bonos], ignore_index=True)
    df = df[df.get("tipo_moneda", "") == "MONEDA NACIONAL"]
    df = df.dropna(subset=["fecha_emision", "fecha_vencimiento", "valor_nominal_mm"])
    df["categoria"] = df["instrumento"].astype(str).map(_categorize_instrumento)
    df["tasa_mensual"] = df["cupon"].map(_parse_tasa_mensual)
    return df.reset_index(drop=True)


def all_tesoro_usd(years: tuple[int, ...] = (2024, 2025)) -> pd.DataFrame:
    """Instrumentos del Tesoro en USD o dólar linked emitidos en el período."""
    letras = parse_letras(years).assign(_origen="letra")
    bonos = parse_bonos(years).assign(_origen="bono")
    df = pd.concat([letras, bonos], ignore_index=True)
    df = df[df.get("tipo_moneda", "") != "MONEDA NACIONAL"]
    return df.dropna(subset=["fecha_emision", "fecha_vencimiento", "valor_nominal_mm"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Scraping de anuncios individuales (gap del año en curso)
# ─────────────────────────────────────────────────────────────────────

def _parse_money(s: str) -> float | None:
    """'$ 11.077,06' → 11077.06; 'USD 282' → ignorado (USD no procesa acá)."""
    if not s or "USD" in s:
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", s).replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_pct(s: str) -> float | None:
    """'2,16%' → 0.0216; '40,53%' → 0.4053."""
    if not s:
        return None
    m = re.search(r"([\d,\.]+)\s*%", s)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ".")) / 100
    except ValueError:
        return None


def _venc_from_instrumento(text: str) -> pd.Timestamp | None:
    m = _VENC_RE.search(text)
    if not m:
        return None
    d, mes, y = m.group(1), m.group(2).lower(), m.group(3)
    if mes not in _MESES:
        return None
    return pd.Timestamp(year=int(y), month=_MESES[mes], day=int(d))


def parse_licitacion(slug: str, only_lecap: bool = False) -> pd.DataFrame:
    """Scrapea un anuncio de licitación. Devuelve instrumentos en pesos del evento.

    only_lecap=True → filtra solo LECAPs (compatibilidad histórica con lecaps_pesos).
    only_lecap=False → todos los instrumentos en pesos categorizados.
    """
    url = f"{NOTICIAS_BASE}/{slug}"
    cached = CACHE_DIR / f"licit_{slug}.html"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        html = cached.read_text(encoding="utf-8")
    else:
        r = httpx.get(url, timeout=30, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        html = r.text
        cached.write_text(html, encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")
    body_text = soup.get_text(" ", strip=True)
    fm = _FECHA_RE.search(body_text)
    fecha_lic: pd.Timestamp | None = None
    if fm:
        d, mes, y = fm.group(1), fm.group(2).lower(), fm.group(3)
        if mes in _MESES:
            fecha_lic = pd.Timestamp(year=int(y), month=_MESES[mes], day=int(d))

    rows = []
    for tabla in soup.find_all("table"):
        trs = tabla.find_all("tr")
        if not trs:
            continue
        header_cells = [c.get_text(" ", strip=True) for c in trs[0].find_all(["td", "th"])]
        header_str = " | ".join(header_cells).lower()
        if "circulación" not in header_str:
            continue  # no es tabla de instrumentos
        # Identificar columna VNO Adjudicado (presente en todas)
        try:
            idx_vno_adj = next(i for i, h in enumerate(header_cells) if "VNO Adjudicado" in h)
        except StopIteration:
            continue
        # TEM solo en tablas de tasa fija; otras tablas no entregan tasa parseable
        idx_tem = next((i for i, h in enumerate(header_cells) if "TEM" in h), None)

        for tr in trs[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) <= idx_vno_adj:
                continue
            instrumento = cells[0]
            instr_norm = instrumento.upper()
            # Excluir USD-puros (BONAR, GLOBALES, instrumentos en USD del Tesoro)
            if "DOLARES ESTADOUNIDENSES" in instr_norm and "VINCULAD" not in instr_norm:
                continue
            categoria = _categorize_instrumento(instrumento)
            if only_lecap and categoria != "LECAP":
                continue
            # Filtros adicionales: la fila debe tener un VENCIMIENTO explícito
            venc = _venc_from_instrumento(instrumento)
            vno_adj = _parse_money(cells[idx_vno_adj])
            tem = _parse_pct(cells[idx_tem]) if idx_tem is not None else None
            if vno_adj is None or venc is None or fecha_lic is None:
                continue
            ticker_m = _TICKER_RE.search(instrumento)
            rows.append({
                "instrumento": categoria,
                "ticker": ticker_m.group(1) if ticker_m else None,
                "fecha_emision": fecha_lic,
                "fecha_vencimiento": venc,
                "tipo_moneda": "MONEDA NACIONAL",
                "valor_nominal_mm": vno_adj,
                "tasa_mensual": tem,
                "categoria": categoria,
                "fuente": "scrape",
            })
    return pd.DataFrame(rows)


def lecaps_pesos_extended(
    years: tuple[int, ...] = (2024, 2025),
    extra_slugs: list[str] | None = None,
) -> pd.DataFrame:
    """LECAPs combinando XLSX (años cerrados) + scrape de licitaciones del año en curso."""
    df_xlsx = lecaps_pesos(years).assign(fuente="xlsx")
    slugs = extra_slugs if extra_slugs is not None else LICITACIONES_SLUGS_2026
    extra_dfs = []
    for slug in slugs:
        try:
            extra_dfs.append(parse_licitacion(slug, only_lecap=True))
        except Exception:
            continue
    if not extra_dfs:
        return df_xlsx
    df_extra = pd.concat(extra_dfs, ignore_index=True)
    cols = ["instrumento", "fecha_emision", "fecha_vencimiento", "tipo_moneda",
            "valor_nominal_mm", "tasa_mensual", "fuente"]
    return pd.concat([df_xlsx[cols], df_extra[cols]], ignore_index=True)


def all_tesoro_pesos_extended(
    years: tuple[int, ...] = (2024, 2025),
    extra_slugs: list[str] | None = None,
) -> pd.DataFrame:
    """Todos los instrumentos en pesos del Tesoro: XLSX + scrape de 2026 (todos los tipos)."""
    df_xlsx = all_tesoro_pesos(years).assign(fuente="xlsx")
    slugs = extra_slugs if extra_slugs is not None else LICITACIONES_SLUGS_2026
    extra_dfs = []
    for slug in slugs:
        try:
            extra_dfs.append(parse_licitacion(slug, only_lecap=False))
        except Exception:
            continue
    if not extra_dfs:
        return df_xlsx
    df_extra = pd.concat(extra_dfs, ignore_index=True)
    cols = ["instrumento", "fecha_emision", "fecha_vencimiento", "tipo_moneda",
            "valor_nominal_mm", "tasa_mensual", "categoria", "fuente"]
    df_xlsx = df_xlsx[cols]
    df_extra = df_extra.reindex(columns=cols)
    return pd.concat([df_xlsx, df_extra], ignore_index=True)


def _download_deuda_xlsx(year: int, quarter: int) -> Path:
    fname = KNOWN_DEUDA_FILES.get((year, quarter))
    if not fname:
        raise ValueError(f"No conozco XLSX de deuda para {year}-Q{quarter}.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / fname
    if cached.exists():
        return cached
    r = httpx.get(f"{URL_BASE}/{fname}", timeout=60, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    cached.write_bytes(r.content)
    return cached


def deuda_bruta_trimestral(year: int = 2025, quarter: int = 4) -> pd.DataFrame:
    """Lee la sheet A.2.1 del XLSX de deuda pública. Devuelve serie trimestral
    de saldos por categoría (en miles de USD al TC del trimestre).

    Categorías relevantes:
      - "I- DEUDA BRUTA + VALORES NEGOCIABLES VINCULADOS AL PBI" (total amplio)
      - "II- DEUDA BRUTA" (sin TVPP)
      - "TÍTULOS PÚBLICOS" (subdividido en moneda nacional / extranjera)
      - "PRÉSTAMOS" (FMI, BID, BIRF, CAF, etc.)
      - "LETRAS DEL TESORO" (LECAPs/BONCAPs/etc.)
      - "ADELANTOS TRANSITORIOS BCRA"
    """
    path = _download_deuda_xlsx(year, quarter)
    df = pd.read_excel(path, sheet_name="A.2.1", header=None)
    # Filas con concepto en columna 1, valores trimestrales en columnas 2-6.
    # Columnas 2-6 corresponden a 5 trimestres consecutivos terminando en (year,quarter).
    sub = df.iloc[12:70, [1, 2, 3, 4, 5, 6]].copy()
    sub.columns = ["concepto", "q-4", "q-3", "q-2", "q-1", "q-0"]
    sub = sub[sub["concepto"].notna()]
    # Convertir a long format
    end_q = pd.Period(f"{year}Q{quarter}", freq="Q")
    fechas = [(end_q + (i - 4)).end_time.normalize() for i in range(5)]
    long = sub.melt(id_vars=["concepto"], value_vars=["q-4", "q-3", "q-2", "q-1", "q-0"],
                    var_name="q_offset", value_name="valor_kusd")
    q_to_fecha = {f"q-{4-i}": fechas[i] for i in range(5)}
    long["fecha"] = long["q_offset"].map(q_to_fecha)
    long["concepto"] = long["concepto"].astype(str).str.strip()
    long = long.dropna(subset=["valor_kusd"])
    return long[["fecha", "concepto", "valor_kusd"]].reset_index(drop=True)


def deuda_total_bruta_history(latest: tuple[int, int] = (2025, 4)) -> pd.DataFrame:
    """Serie trimestral de deuda bruta total (II) en millones USD."""
    df = deuda_bruta_trimestral(*latest)
    total = df[df["concepto"].str.startswith("II- DEUDA BRUTA")]
    return total.assign(valor_musd=total["valor_kusd"] / 1e3)[["fecha", "valor_musd"]]


def deuda_total_bruta_extendida() -> pd.DataFrame:
    """Combina múltiples XLSX trimestrales para serie larga de deuda bruta.

    Cada XLSX trae 5 trimestres; concatenamos todos los disponibles y dedupeamos por fecha
    (manteniendo la última versión publicada).
    """
    frames = []
    for (y, q) in sorted(KNOWN_DEUDA_FILES.keys()):
        try:
            df = deuda_bruta_trimestral(y, q)
            total = df[df["concepto"].str.startswith("II- DEUDA BRUTA")]
            frames.append(total.assign(
                valor_musd=total["valor_kusd"] / 1e3,
                source_xlsx=f"{y}Q{q}",
            )[["fecha", "valor_musd", "source_xlsx"]])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Dedup: mantener la última versión publicada de cada fecha
    df = df.sort_values(["fecha", "source_xlsx"]).drop_duplicates("fecha", keep="last")
    return df.sort_values("fecha").reset_index(drop=True)


def deuda_breakdown_extendido() -> pd.DataFrame:
    """Breakdown por categoría combinando múltiples XLSX (serie larga)."""
    frames = []
    for (y, q) in sorted(KNOWN_DEUDA_FILES.keys()):
        try:
            br = deuda_breakdown_categorias(latest=(y, q))
            br["source_xlsx"] = f"{y}Q{q}"
            frames.append(br)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Dedup: mantener la versión más reciente publicada de cada fecha
    df = df.sort_values(["fecha", "source_xlsx"]).drop_duplicates("fecha", keep="last")
    return df.drop(columns=["source_xlsx"]).sort_values("fecha").reset_index(drop=True)


def deuda_pbi_oficial(latest: tuple[int, int] = (2025, 4)) -> pd.DataFrame:
    """Lee A.4.7 (Indicadores de sostenibilidad) — devuelve serie ANUAL de
    deuda/PBI computada por Sec. Finanzas. Mucho más robusto que recompilarlo
    nosotros porque ya usa la metodología INDEC oficial.

    Devuelve DataFrame con: anio, deuda_bruta_pct_pbi, deuda_externa_pct_pbi.
    """
    path = _download_deuda_xlsx(*latest)
    df = pd.read_excel(path, sheet_name="A.4.7", header=None)
    # Fila 7 tiene los años (col 3+); fila 9 = "Deuda Bruta de la Administración Central"
    # fila 10 = "Deuda Externa de la Administración Central"
    years_row = df.iloc[7, 3:].tolist()
    deuda_bruta_row = df.iloc[9, 3:].tolist()
    deuda_externa_row = df.iloc[10, 3:].tolist()
    rows = []
    for y, db, de in zip(years_row, deuda_bruta_row, deuda_externa_row):
        # El año puede venir como int (2000) o como string "2022 (1)".
        anio = None
        if isinstance(y, (int, float)) and not pd.isna(y):
            anio = int(y)
        elif isinstance(y, str):
            m = re.search(r"(20\d\d|19\d\d)", y)
            if m:
                anio = int(m.group(1))
        if anio is None:
            continue
        rows.append({
            "anio": anio,
            "deuda_bruta_pct_pbi": float(db) if isinstance(db, (int, float)) and not pd.isna(db) else None,
            "deuda_externa_pct_pbi": float(de) if isinstance(de, (int, float)) and not pd.isna(de) else None,
        })
    return pd.DataFrame(rows)


def deuda_breakdown_categorias(latest: tuple[int, int] = (2025, 4)) -> pd.DataFrame:
    """Breakdown principal de la deuda bruta: Títulos $, Títulos USD, FMI, otros Org. Int.,
    Letras corto plazo, Adelantos BCRA. Wide-format con columna por categoría, en MUSD."""
    df = deuda_bruta_trimestral(*latest)
    bag: dict[str, pd.Series] = {}

    def grab(label: str, contains: str, exact: bool = False) -> None:
        if exact:
            sub = df[df["concepto"] == contains]
        else:
            sub = df[df["concepto"].str.contains(contains, regex=False)]
        if sub.empty:
            return
        # Algunas categorías aparecen 2x (LP + CP). Sumar por fecha.
        s = sub.groupby("fecha")["valor_kusd"].sum() / 1e3
        bag[label] = s

    grab("Títulos $ (Long Term)", "- Moneda nacional", exact=True)
    grab("Títulos USD (Long Term)", "- Moneda extranjera", exact=True)
    grab("Letras del Tesoro (LP+CP)", "LETRAS DEL TESORO")
    grab("FMI", "- FMI", exact=True)
    grab("BID", "- BID", exact=True)
    grab("BIRF", "- BIRF", exact=True)
    grab("CAF", "- CAF", exact=True)
    grab("Adelantos BCRA", "ADELANTOS TRANSITORIOS BCRA")
    grab("Avales", "AVALES")
    grab("Pago diferido + reestructuración", "DEUDA EN SITUACIÓN DE PAGO")

    if not bag:
        return pd.DataFrame()
    out = pd.concat(bag, axis=1).reset_index().rename(columns={"index": "fecha"})
    return out.fillna(0)


def total_pesos_stock_breakdown(
    years: tuple[int, ...] = (2024, 2025),
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Stock fin-de-mes por categoría (LECAP, BONCAP, CER-linked, TAMAR-linked, etc.)."""
    df = all_tesoro_pesos_extended(years)
    end = end_date or pd.Timestamp.today().normalize()
    fechas = pd.date_range("2024-01-01", end, freq="ME")
    rows = []
    for fecha in fechas:
        alive = df[(df["fecha_emision"] <= fecha) & (df["fecha_vencimiento"] > fecha)]
        breakdown = alive.groupby("categoria")["valor_nominal_mm"].sum().to_dict()
        breakdown["fecha"] = fecha
        breakdown["total"] = alive["valor_nominal_mm"].sum()
        rows.append(breakdown)
    return pd.DataFrame(rows).fillna(0).set_index("fecha").reset_index()


def lecap_stock_history(
    years: tuple[int, ...] = (2024, 2025),
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Stock mensual fin-de-mes de LECAPs vivos en pesos (valor nominal).

    Para cada fin de mes t, suma valor_nominal de LECAPs con
    emision <= t y vencimiento > t.
    """
    df = lecaps_pesos_extended(years)
    end = end_date or pd.Timestamp.today().normalize()
    fechas = pd.date_range("2024-01-01", end, freq="ME")
    rows = []
    for fecha in fechas:
        alive = df[(df["fecha_emision"] <= fecha) & (df["fecha_vencimiento"] > fecha)]
        rows.append({
            "fecha": fecha,
            "stock_lecap_nominal_mm": float(alive["valor_nominal_mm"].sum()),
            "n_lecaps_vivas": int(len(alive)),
        })
    return pd.DataFrame(rows)


def lecap_avg_tasa_history(
    years: tuple[int, ...] = (2024, 2025),
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tasa efectiva mensual promedio ponderada (por nominal vivo) en cada fin de mes.

    Solo considera LECAPs de tasa fija (excluye TAMAR-linked y variantes sin tasa parseable).
    """
    df = lecaps_pesos_extended(years)
    df = df.dropna(subset=["tasa_mensual"])  # solo tasa fija
    end = end_date or pd.Timestamp.today().normalize()
    fechas = pd.date_range("2024-01-01", end, freq="ME")
    rows = []
    for fecha in fechas:
        alive = df[(df["fecha_emision"] <= fecha) & (df["fecha_vencimiento"] > fecha)]
        peso_total = alive["valor_nominal_mm"].sum()
        if peso_total > 0:
            tasa_w = float((alive["tasa_mensual"] * alive["valor_nominal_mm"]).sum() / peso_total)
        else:
            tasa_w = float("nan")
        rows.append({
            "fecha": fecha,
            "tasa_mensual_pond": tasa_w,
            "stock_tasa_fija_mm": float(peso_total),
        })
    return pd.DataFrame(rows)


def intereses_capitalizados_history(
    years: tuple[int, ...] = (2024, 2025),
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Estimación mensual de intereses LECAP capitalizados (no contabilizados como gasto fiscal).

    intereses_capitalizados_t ≈ stock_LECAPs_(t-1) × tasa_mensual_pond_t

    Retorna DataFrame con: fecha, intereses_capitalizados_mm.
    """
    stock = lecap_stock_history(years, end_date).set_index("fecha")
    tasa = lecap_avg_tasa_history(years, end_date).set_index("fecha")
    df = stock.join(tasa, how="inner").reset_index()
    df["stock_lag1"] = df["stock_lecap_nominal_mm"].shift(1)
    df["intereses_capitalizados_mm"] = df["stock_lag1"] * df["tasa_mensual_pond"]
    return df[["fecha", "intereses_capitalizados_mm", "stock_lag1", "tasa_mensual_pond"]]
