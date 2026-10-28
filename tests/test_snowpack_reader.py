import pandas as pd

from conftest import FIXTURE_PRO
from snowpack_reader import read_snowpack


def test_read_pro_file():
    profile = read_snowpack(str(FIXTURE_PRO))

    assert profile is not None
    assert len(profile) > 0
    timestamps = pd.to_datetime(profile.data.timestamp.values)
    assert timestamps.min() >= pd.Timestamp("2026-10-14")
    assert timestamps.max() <= pd.Timestamp("2026-11-01")


def test_read_does_not_write_netcdf_cache_by_default():
    read_snowpack(str(FIXTURE_PRO))

    assert not FIXTURE_PRO.with_suffix(".nc").exists()


def test_slice_limits_profile_to_date_range():
    profile = read_snowpack(str(FIXTURE_PRO))
    sliced = profile.slice(start_date="2026-10-20", end_date="2026-10-22")

    timestamps = pd.to_datetime(sliced.data.timestamp.values)
    assert 0 < len(sliced) < len(profile)
    assert timestamps.min() >= pd.Timestamp("2026-10-20")
    assert timestamps.max() < pd.Timestamp("2026-10-23")
