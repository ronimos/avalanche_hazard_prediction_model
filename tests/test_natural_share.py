"""Checks the natural share model on simulated record-days."""
import numpy as np
import pandas as pd
import pytest

import likelihood_data as ld
import natural_share as ns


@pytest.mark.slow
def test_share_pools_a_polygon_with_few_all_natural_records():
    rng = np.random.default_rng(2)
    polygons = ld.polygon_regions().index[:30]
    # 40 record-days in every polygon but the first, which has 4, all natural
    rows = [(polygons[0], 1)] * 4 + [(p, int(rng.uniform() < 0.6)) for p in polygons[1:] for _ in range(40)]
    events = pd.DataFrame(rows, columns=["polygon", "natural"])
    events["date"] = pd.Timestamp("2025-01-06") + pd.to_timedelta(rng.integers(0, 90, len(events)), unit="D")
    events["season"] = 2024
    events["day_of_season"] = (events["date"] - pd.Timestamp("2024-10-01")).dt.days
    for name in ld.DRIVERS:
        events[name] = rng.normal(size=len(events))
    events["Sn_capped"] = events["Ss_capped"] = 0

    model = ns.NaturalShareModel(draws=300, tune=300, chains=2, n_predict_draws=100).fit(events)
    share = model.predict_draws(events)

    assert share.shape == (100, len(events))
    assert ((share > 0) & (share < 1)).all()
    assert abs(share.mean() - events["natural"].mean()) < 0.05
    # Four natural records alone don't make the polygon's share 100%
    assert share[:, 0].mean() < 0.8
