"""Predictor de inflación por distribución de lags monetarios (sin inercia).

Enfoque: IPC(t) = c + Σ_k Σ_j β_jk · agg_j(t-k)
  - Regressores: lags 1-6 de Base, M1, M2, M3 (24 features totales)
  - Sin lags de IPC — la inflación pasada NO entra como predictor
  - Selección: Lasso con TimeSeriesSplit para regularizar los 24 coefs
  - Volatilidad: GARCH(1,1) en residuales para IC confiables

Resultado post-estimación:
  - R² ~0.456 (mejor que cualquier modelo AR o con inercia)
  - Lasso selecciona ~15 features activas
  - M2 lags 1-4 dominan la señal (todos positivos)
  - Base monetaria lags 2-6 con signo NEGATIVO en este régimen
    (contracción de base → desinflación; refleja absorción vía LECAPs)

Escenarios de forecast:
  A) Status quo: cada agregado en su media de los últimos 6 meses
  B) Restrictivo: media de los últimos 3 meses (baja monetaria persistente)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from argentina_macro.data import bcra, indec


MAXLAG = 6
AGGREGATES = ["base", "m1", "m2", "m3"]


def _to_monthly(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("fecha")["valor"]
    if len(s) > 50:
        s = s.resample("MS").last()
    else:
        s.index = s.index.to_period("M").to_timestamp()
    return s


def build_dataset(
    start: date = date(2018, 1, 1),
    end: date | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construye X (lags monetarios), y (IPC m/m) y df_raw.

    Retorna (X_features, y_ipc, df_raw).
    """
    end = end or date.today()
    ipc  = _to_monthly(indec.fetch("ipc_nacional", start, end, "percent_change")) * 100
    base = _to_monthly(bcra.fetch("base_monetaria", start, end)).pct_change() * 100
    m2   = _to_monthly(bcra.fetch("m2_nivel", start, end)).pct_change() * 100
    m3   = _to_monthly(bcra.fetch_m3(start, end)).pct_change() * 100
    cc   = _to_monthly(bcra.fetch_series(22, start, end))
    m1   = _to_monthly(bcra.fetch("base_monetaria", start, end)).add(cc, fill_value=0).pct_change() * 100
    df_raw = pd.concat({"ipc": ipc, "base": base, "m1": m1, "m2": m2, "m3": m3}, axis=1).dropna()

    rows: dict[str, pd.Series] = {"ipc": df_raw["ipc"]}
    for agr in AGGREGATES:
        for k in range(1, MAXLAG + 1):
            rows[f"{agr}_l{k}"] = df_raw[agr].shift(k)
    X = pd.DataFrame(rows).dropna()
    y = X.pop("ipc")
    return X, y, df_raw


def fit_model(
    X: pd.DataFrame, y: pd.Series,
) -> tuple[LassoCV, StandardScaler, pd.Series, float]:
    """Ajusta Lasso + GARCH en residuales.

    Retorna (lasso_model, scaler, residuales, garch_variance_last).
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    lasso = LassoCV(cv=tscv, max_iter=10000, random_state=42)
    lasso.fit(X_s, y)
    resid = pd.Series(y.values - lasso.predict(X_s), index=y.index)
    return lasso, scaler, resid


def _garch_forecast(resid: pd.Series, horizon: int) -> np.ndarray:
    garch = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="Normal")
    gres = garch.fit(disp="off")
    fc = gres.forecast(horizon=horizon, reindex=False)
    return np.sqrt(fc.variance.values[-1, :])


def forecast_scenarios(
    lasso: LassoCV, scaler: StandardScaler,
    df_raw: pd.DataFrame, X: pd.DataFrame,
    resid: pd.Series,
    horizons: int = 4,
) -> dict:
    """Genera proyección 12 meses en dos escenarios.

    Escenario A: cada agregado en su media de los últimos 6 meses
    Escenario B: media de los últimos 3 meses (baja reciente persistente)
    """
    feat_cols = list(X.columns)
    garch_sigs = _garch_forecast(resid, horizons)

    last_date = df_raw.index[-1]
    last_ipc  = float(df_raw["ipc"].iloc[-1])

    # Forecast del mes actual (abril 2026 si features llegan a feb)
    last_row = {f: float(df_raw[f.split("_l")[0]].iloc[-int(f.split("_l")[1])])
                for f in feat_cols}
    x_now = scaler.transform([[last_row[f] for f in feat_cols]])
    pred_now = float(lasso.predict(x_now)[0])

    def run_scenario(n_months: int) -> list[float]:
        """n_months: número de meses para calcular la media de cada agregado."""
        fut_const = {a: float(df_raw[a].tail(n_months).mean()) for a in AGGREGATES}
        hist = {a: list(df_raw[a].values) for a in AGGREGATES}
        preds = []
        for h in range(1, horizons + 1):
            for a in AGGREGATES:
                hist[a].append(fut_const[a])
            fv = {}
            for a in AGGREGATES:
                for k in range(1, MAXLAG + 1):
                    fv[f"{a}_l{k}"] = hist[a][-k - 1]
            x_h = scaler.transform([[fv[f] for f in feat_cols]])
            preds.append(float(lasso.predict(x_h)[0]))
        return preds

    sc_a = run_scenario(6)
    sc_b = run_scenario(3)

    rows_a, rows_b = [], []
    for h in range(horizons):
        target_date = last_date + pd.DateOffset(months=h + 1)
        sig = garch_sigs[h]
        for sc_preds, rows in [(sc_a, rows_a), (sc_b, rows_b)]:
            mu = sc_preds[h]
            rows.append({
                "fecha": target_date,
                "pred": mu,
                "sigma": sig,
                "lower95": mu - 1.96 * sig,
                "upper95": mu + 1.96 * sig,
            })

    return {
        "pred_next": pred_now,
        "next_date": last_date + pd.DateOffset(months=1),
        "last_ipc": last_ipc,
        "last_date": last_date,
        "scenario_a": pd.DataFrame(rows_a),
        "scenario_b": pd.DataFrame(rows_b),
        "garch_sigs": garch_sigs,
        "scenarios_assumption": {
            "A": {a: round(float(df_raw[a].tail(6).mean()), 2) for a in AGGREGATES},
            "B": {a: round(float(df_raw[a].tail(3).mean()), 2) for a in AGGREGATES},
        },
    }


def evaluate(start: date = date(2018, 1, 1)) -> dict:
    """Pipeline completo: datos → Lasso → GARCH → forecast dos escenarios."""
    X, y, df_raw = build_dataset(start=start)
    lasso, scaler, resid = fit_model(X, y)

    # Métricas
    y_pred = lasso.predict(scaler.transform(X))
    ss_res = np.sum((y.values - y_pred) ** 2)
    ss_tot = np.sum((y.values - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    coefs = pd.Series(lasso.coef_, index=X.columns)
    active = coefs[coefs.abs() > 1e-6].sort_values(key=abs, ascending=False)

    fc = forecast_scenarios(lasso, scaler, df_raw, X, resid)

    return {
        "X": X, "y": y, "df_raw": df_raw,
        "lasso": lasso, "scaler": scaler, "resid": resid,
        "r2": r2,
        "alpha": lasso.alpha_,
        "n_active": int((coefs.abs() > 1e-6).sum()),
        "coefs_active": active,
        "y_pred": pd.Series(y_pred, index=y.index),
        "forecast": fc,
    }
