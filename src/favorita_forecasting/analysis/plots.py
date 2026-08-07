from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from favorita_forecasting.utils import ensure_dir


def save_core_plots(panel: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    historical = panel.loc[~panel["is_test"]].copy()

    daily = historical.groupby("date", as_index=False)["sales"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["sales"])
    ax.set(title="Ventes quotidiennes totales du périmètre", xlabel="Date", ylabel="Unités")
    fig.tight_layout()
    fig.savefig(output_dir / "daily_sales_total.png", dpi=150)
    plt.close(fig)

    family = historical.groupby("family", as_index=False)["sales"].sum().sort_values("sales")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(family["family"], family["sales"])
    ax.set(title="Volume par famille", xlabel="Unités", ylabel="Famille")
    fig.tight_layout()
    fig.savefig(output_dir / "sales_by_family.png", dpi=150)
    plt.close(fig)

    promo = historical.groupby("any_promotion", as_index=False)["sales"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(promo["any_promotion"].astype(str), promo["sales"])
    ax.set(title="Vente moyenne avec et sans promotion", xlabel="Promotion", ylabel="Vente moyenne")
    fig.tight_layout()
    fig.savefig(output_dir / "promotion_mean_sales.png", dpi=150)
    plt.close(fig)
