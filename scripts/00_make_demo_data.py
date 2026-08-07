from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(output: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    train_dates = pd.date_range("2020-01-01", "2021-03-31", freq="D")
    test_dates = pd.date_range("2021-04-01", periods=16, freq="D")

    stores = pd.DataFrame(
        [
            {"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1},
            {"store_nbr": 2, "city": "Guayaquil", "state": "Guayas", "type": "B", "cluster": 2},
            {"store_nbr": 3, "city": "Cuenca", "state": "Azuay", "type": "C", "cluster": 3},
        ]
    )
    stores.to_csv(output / "stores.csv", index=False)

    families = ["BEVERAGES", "GROCERY I", "PRODUCE", "PERSONAL CARE"]
    item_rows = []
    item_nbr = 1000
    for family_idx, family in enumerate(families):
        for local in range(3):
            item_rows.append(
                {
                    "item_nbr": item_nbr,
                    "family": family,
                    "class": 100 + family_idx * 10 + local,
                    "perishable": int(family == "PRODUCE" or (family == "BEVERAGES" and local == 0)),
                }
            )
            item_nbr += 1
    items = pd.DataFrame(item_rows)
    items.to_csv(output / "items.csv", index=False)

    holidays = pd.DataFrame(
        [
            {"date": "2020-12-25", "type": "Holiday", "locale": "National", "locale_name": "Ecuador", "description": "Navidad", "transferred": False},
            {"date": "2021-01-01", "type": "Holiday", "locale": "National", "locale_name": "Ecuador", "description": "Primer dia del ano", "transferred": False},
            {"date": "2021-02-12", "type": "Event", "locale": "Local", "locale_name": "Quito", "description": "Fiesta Quito", "transferred": False},
            {"date": "2021-03-01", "type": "Holiday", "locale": "Regional", "locale_name": "Guayas", "description": "Fiesta Guayas", "transferred": False},
            {"date": "2021-04-02", "type": "Holiday", "locale": "National", "locale_name": "Ecuador", "description": "Viernes Santo", "transferred": False},
        ]
    )
    holidays.to_csv(output / "holidays_events.csv", index=False)

    all_dates = train_dates.union(test_dates)
    oil = pd.DataFrame({"date": all_dates})
    oil["dcoilwtico"] = 50 + np.linspace(0, 8, len(oil)) + rng.normal(0, 1.5, len(oil))
    oil.loc[oil.index % 29 == 0, "dcoilwtico"] = np.nan
    oil.to_csv(output / "oil.csv", index=False)

    family_base = {"BEVERAGES": 9.0, "GROCERY I": 7.0, "PRODUCE": 5.5, "PERSONAL CARE": 3.0}
    store_factor = {1: 1.25, 2: 1.0, 3: 0.8}
    train_rows = []
    test_rows = []
    test_id = 1
    holiday_dates = set(pd.to_datetime(holidays["date"]))

    for date in all_dates:
        dow_factor = 1.22 if date.dayofweek in [4, 5] else (0.88 if date.dayofweek == 1 else 1.0)
        holiday_factor = 1.3 if date in holiday_dates else 1.0
        for store in stores["store_nbr"]:
            for item in items.itertuples(index=False):
                promo_probability = 0.06 + 0.08 * (item.family in ["BEVERAGES", "PERSONAL CARE"])
                onpromotion = bool(rng.random() < promo_probability)
                if date in test_dates:
                    test_rows.append(
                        {"id": test_id, "date": date.date(), "store_nbr": store, "item_nbr": item.item_nbr, "onpromotion": onpromotion}
                    )
                    test_id += 1
                    continue
                item_factor = 0.75 + 0.25 * ((item.item_nbr % 3) + 1)
                promo_factor = 1.65 if onpromotion else 1.0
                annual = 1 + 0.12 * np.sin(2 * np.pi * date.dayofyear / 365)
                mean = family_base[item.family] * store_factor[store] * item_factor * dow_factor * holiday_factor * promo_factor * annual
                units = float(rng.poisson(max(mean, 0.1)))
                if rng.random() < 0.003:
                    units = -float(rng.integers(1, 3))
                # Mimic sparse Favorita: omit some zero rows.
                if units == 0 and rng.random() < 0.8:
                    continue
                train_rows.append(
                    {"id": len(train_rows) + 1, "date": date.date(), "store_nbr": store, "item_nbr": item.item_nbr, "unit_sales": units, "onpromotion": onpromotion}
                )

    pd.DataFrame(train_rows).to_csv(output / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output / "test.csv", index=False)
    pd.DataFrame({"id": [row["id"] for row in test_rows], "unit_sales": 0.0}).to_csv(
        output / "sample_submission.csv", index=False
    )

    transaction_rows = []
    for date in train_dates:
        for store in stores["store_nbr"]:
            base = 700 * store_factor[store] * (1.15 if date.dayofweek in [4, 5] else 1.0)
            transaction_rows.append(
                {"date": date.date(), "store_nbr": store, "transactions": int(max(rng.normal(base, 40), 100))}
            )
    pd.DataFrame(transaction_rows).to_csv(output / "transactions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo_raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_demo_data(Path(args.output), args.seed)
    print(f"Demo dataset created in {args.output}")
