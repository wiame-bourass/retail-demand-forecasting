import numpy as np
import pandas as pd

from favorita_forecasting.monitoring.drift import population_stability_index


def test_psi_is_small_for_similar_samples_and_large_for_shift():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 5000))
    similar = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(3, 1, 5000))
    assert population_stability_index(reference, similar) < 0.1
    assert population_stability_index(reference, shifted) > 0.25
