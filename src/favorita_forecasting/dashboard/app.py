from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from favorita_forecasting.storage import frame_exists, read_frame
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = Path(os.getenv("FAVORITA_OUTPUTS", ROOT / "outputs"))
ARTIFACTS = Path(os.getenv("FAVORITA_ARTIFACTS", ROOT / "artifacts"))

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Promotion-aware retail forecasting")
st.caption("Ventes observées utilisées comme proxy de la demande — horizon 16 jours")


def read_optional(path: Path, kind: str = "csv"):
    if kind == "csv":
        return pd.read_csv(path) if path.exists() else None
    return read_frame(path) if frame_exists(path) else None


metrics = read_optional(OUTPUTS / "holdout_metrics.csv")
forecast = read_optional(OUTPUTS / "holdout_predictions.parquet", "parquet")
inventory = read_optional(OUTPUTS / "inventory_comparison.csv")

if metrics is None or forecast is None:
    st.warning("Exécute d'abord les scripts 06 à 09 pour générer les résultats.")
    st.stop()

m = metrics.iloc[0]
cols = st.columns(4)
cols[0].metric("WAPE", f"{m['wape']:.2%}")
cols[1].metric("MAE", f"{m['mae']:.1f}")
cols[2].metric("Biais", f"{m['bias']:.2%}")
cols[3].metric("RMSLE", f"{m['rmsle']:.3f}")

forecast["date"] = pd.to_datetime(forecast["date"])
stores = sorted(forecast["store_nbr"].unique())
families = sorted(forecast["family"].unique())
left, right = st.columns(2)
store = left.selectbox("Magasin", stores)
family = right.selectbox("Famille", families)
view = forecast[forecast["store_nbr"].eq(store) & forecast["family"].eq(family)].copy()
long = view.melt(
    id_vars="date",
    value_vars=["actual", "prediction"],
    var_name="series",
    value_name="sales",
)
st.plotly_chart(
    px.line(long, x="date", y="sales", color="series", markers=True, title="Prévision vs réalisé"),
    use_container_width=True,
)

promo = forecast.groupby("any_promotion", as_index=False).apply(
    lambda g: pd.Series({
        "wape": (g["actual"] - g["prediction"]).abs().sum() / g["actual"].abs().sum(),
        "bias": (g["prediction"] - g["actual"]).sum() / g["actual"].abs().sum(),
    }),
    include_groups=False,
).reset_index(drop=True)
st.subheader("Performance promotionnelle")
st.dataframe(promo, use_container_width=True)

if inventory is not None:
    st.subheader("Simulation supply chain")
    st.dataframe(inventory, use_container_width=True)
    st.plotly_chart(
        px.bar(inventory, x="policy", y="total_cost", title="Coût total simulé par politique"),
        use_container_width=True,
    )

model_card_path = ARTIFACTS / "model_card.json"
if model_card_path.exists():
    st.subheader("Model card")
    st.json(json.loads(model_card_path.read_text(encoding="utf-8")))
