from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from favorita_forecasting.utils import ensure_dir

SERIES_KEYS = ["store_nbr", "family"]


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    clean = series.astype(float)
    return float(clean.autocorr(lag=lag)) if len(clean) > lag and clean.std() > 0 else np.nan


def _trend_slope(series: pd.Series) -> float:
    y = series.astype(float).to_numpy()
    if len(y) < 3 or np.nanstd(y) == 0:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(stats.theilslopes(y, x).slope)


def series_profiles(panel: pd.DataFrame) -> pd.DataFrame:
    historical = panel.loc[~panel["is_test"]].copy()
    rows = []
    for keys, group in historical.groupby(SERIES_KEYS, sort=False):
        sales = group.sort_values("date")["sales"].fillna(0).astype(float)
        positive = sales[sales > 0]
        zero_rate = float((sales == 0).mean())
        adi = float(len(sales) / max(len(positive), 1))
        cv2 = float((positive.std(ddof=1) / positive.mean()) ** 2) if len(positive) > 1 and positive.mean() > 0 else np.nan
        if adi < 1.32 and (np.isnan(cv2) or cv2 < 0.49):
            demand_class = "smooth"
        elif adi >= 1.32 and (np.isnan(cv2) or cv2 < 0.49):
            demand_class = "intermittent"
        elif adi < 1.32:
            demand_class = "erratic"
        else:
            demand_class = "lumpy"
        rows.append(
            {
                "store_nbr": keys[0],
                "family": keys[1],
                "n_days": len(sales),
                "mean_sales": sales.mean(),
                "median_sales": sales.median(),
                "std_sales": sales.std(ddof=1),
                "zero_rate": zero_rate,
                "adi": adi,
                "cv2": cv2,
                "demand_class": demand_class,
                "lag7_autocorrelation": _safe_autocorr(sales, 7),
                "trend_slope_theilsen": _trend_slope(sales),
                "promo_rate": group["any_promotion"].mean(),
            }
        )
    return pd.DataFrame(rows)


def promotion_effects(panel: pd.DataFrame) -> pd.DataFrame:
    historical = panel.loc[~panel["is_test"]].copy()
    rows = []
    for family, group in historical.groupby("family", sort=False):
        promo = group.loc[group["any_promotion"], "sales"].dropna().astype(float)
        nonpromo = group.loc[~group["any_promotion"], "sales"].dropna().astype(float)
        if len(promo) < 2 or len(nonpromo) < 2:
            continue
        welch = stats.ttest_ind(promo, nonpromo, equal_var=False, nan_policy="omit")
        pooled = np.sqrt((promo.var(ddof=1) + nonpromo.var(ddof=1)) / 2)
        effect_size = (promo.mean() - nonpromo.mean()) / pooled if pooled > 0 else np.nan
        rows.append(
            {
                "family": family,
                "promo_n": len(promo),
                "nonpromo_n": len(nonpromo),
                "promo_mean": promo.mean(),
                "nonpromo_mean": nonpromo.mean(),
                "absolute_uplift": promo.mean() - nonpromo.mean(),
                "relative_uplift": promo.mean() / nonpromo.mean() - 1 if nonpromo.mean() > 0 else np.nan,
                "welch_t_stat": welch.statistic,
                "welch_p_value": welch.pvalue,
                "cohens_d_approx": effect_size,
                "interpretation": "association_not_causal",
            }
        )
    return pd.DataFrame(rows)


def day_of_week_effects(panel: pd.DataFrame) -> pd.DataFrame:
    historical = panel.loc[~panel["is_test"]].copy()
    historical["day_of_week"] = pd.to_datetime(historical["date"]).dt.day_name()
    return (
        historical.groupby(["family", "day_of_week"], as_index=False)
        .agg(mean_sales=("sales", "mean"), median_sales=("sales", "median"), observations=("sales", "size"))
    )


def holiday_effects(panel: pd.DataFrame) -> pd.DataFrame:
    historical = panel.loc[~panel["is_test"]].copy()
    return (
        historical.groupby(["family", "is_holiday"], as_index=False)
        .agg(mean_sales=("sales", "mean"), median_sales=("sales", "median"), observations=("sales", "size"))
    )


def oil_sales_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    historical = panel.loc[~panel["is_test"]].groupby("date", as_index=False).agg(
        total_sales=("sales", "sum"), dcoilwtico=("dcoilwtico", "first")
    )
    valid = historical.dropna()
    if len(valid) < 3:
        rho = p = np.nan
    else:
        rho, p = stats.spearmanr(valid["total_sales"], valid["dcoilwtico"])
    return pd.DataFrame([{"spearman_rho": rho, "p_value": p, "n_dates": len(valid)}])


def run_statistical_analysis(panel: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    ensure_dir(output_dir)
    tables = {
        "series_profiles": series_profiles(panel),
        "promotion_effects": promotion_effects(panel),
        "day_of_week_effects": day_of_week_effects(panel),
        "holiday_effects": holiday_effects(panel),
        "oil_sales_correlation": oil_sales_correlation(panel),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables
