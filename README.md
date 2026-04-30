# Argentina · Liquidez & Política Económica

Dashboard interactivo de coyuntura macroeconómica argentina con foco en la
relación liquidez → inflación. Datos en tiempo real desde fuentes oficiales
públicas (BCRA, INDEC, MECON, Sec. de Finanzas) y de mercado (dolarapi, Yahoo
Finance).

## Secciones

- **Liquidez monetaria**: Base, M2, M3, M3* — vista nominal, real (deflactada)
  y en USD.
- **Inflación y tasas**: IPC headline y núcleo, tasa de política, BADLAR, tasa
  real ex-post.
- **Sector externo / cambiario**: TC nominal vs real, brecha cambiaria, reservas
  brutas y netas (3 metodologías), balanza comercial, cuenta capital.
- **Deuda y fiscal**: deuda Tesoro consolidada por instrumento, ratio deuda/PBI,
  resultado primario oficial vs ajustado por capitalización LECAPs.
- **Sector real**: EMAE general + 15 sectores económicos.
- **Predictor de inflación**: modelo Lasso de lags distribuidos sobre agregados
  monetarios (sin inercia), con GARCH(1,1) en residuales.

## Run local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy en Streamlit Cloud

1. Subir este repositorio a GitHub (público).
2. Ir a https://share.streamlit.io y autenticar con GitHub.
3. **New app** → seleccionar el repo, branch `main`, main file `streamlit_app.py`.
4. Deploy. URL queda como `<app-name>.streamlit.app`.

El primer arranque tarda 1-2 minutos descargando los XLS oficiales del BCRA y
Sec. Finanzas a `data/raw/` (cache local).

## Fuentes de datos

| Fuente | Datos | Frecuencia |
|---|---|---|
| BCRA API v4.0 | Reservas, base, M2, agregados, tasas | Diaria |
| BCRA XLS oficial | Estado Resumido (balance) | Semanal |
| INDEC datos.gob.ar | IPC, EMAE, balanza comercial, cuenta capital | Mensual |
| INDEC XLS oficial | EMAE detallado por sector | Mensual |
| Sec. de Finanzas | Stock de deuda + composición + colocaciones | Trimestral |
| dolarapi.com | Cotizaciones blue/MEP/CCL | Tiempo real |
| Yahoo Finance | CNY/USD para valuación swap China | Diario |

## Limitaciones documentadas

- Reservas netas: las deducciones (swap China, swap EEUU, FMI desembolsos) no
  están separadas en el balance público — se aplican ajustes manuales con
  snapshots curados.
- BOPREAL: no se resta de reservas hoy (es pasivo a futuro).
- EMAE: lag de ~2 meses post-publicación.
- Deuda Tesoro: la API no expone stock, se reconstruye desde XLS de
  colocaciones + scrape de licitaciones del año en curso.

## Disclaimer

Dashboard de research, NO de trading. Sin recomendaciones de inversión. Datos
sujetos a revisión por las fuentes oficiales.
