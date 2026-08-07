import pandas as pd

from favorita_forecasting.evaluation.hierarchy import bottom_up_coherence, hierarchy_metrics


def test_bottom_up_is_coherent():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01"] * 4),
            "state": ["A", "A", "B", "B"],
            "city": ["X", "X", "Y", "Y"],
            "store_nbr": [1, 1, 2, 2],
            "family": ["F1", "F2", "F1", "F2"],
            "actual": [10, 20, 5, 15],
            "prediction": [11, 19, 6, 14],
        }
    )
    result = bottom_up_coherence(predictions)
    assert result["coherent"] is True
    levels = hierarchy_metrics(predictions)
    assert {"total", "store", "family", "store_family"}.issubset(
    set(levels["level"])
)
