import numpy as np
import pandas as pd
import pytest

import feature_store as fs
import travel
from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking

BOS, LAL, DEN = 1610612738, 1610612747, 1610612743


def _sched(rows):
    return pd.DataFrame(rows, columns=["GAME_ID", "SEASON", "GAME_DATE_EST", "TIP_UTC", "HOME_TEAM_ID",
                                       "AWAY_TEAM_ID", "HOME_TRICODE", "AWAY_TRICODE"])


SCHED = _sched([
    ("0012500001", "2025-26", "2025-10-15", "2025-10-15T23:00:00Z", BOS, LAL, "BOS", "LAL"),  # preseason: ignored
    ("0022500001", "2025-26", "2025-11-01", "2025-11-01T23:00:00Z", BOS, LAL, "BOS", "LAL"),
    ("0022500002", "2025-26", "2025-11-03", "2025-11-04T02:00:00Z", DEN, LAL, "DEN", "LAL"),
    ("0022500003", "2025-26", "2025-11-04", "2025-11-05T03:00:00Z", LAL, DEN, "LAL", "DEN"),
])


def test_travel_features_values():
    tf = travel.travel_features(SCHED).set_index(["TEAM_ID", "GAME_ID"])
    lal = tf.xs(LAL)
    assert lal.loc["0022500001", "travel_km"] == 0.0           # first game of season
    assert lal.loc["0022500002", "travel_km"] == pytest.approx(2840, rel=0.03)  # Boston -> Denver
    assert lal.loc["0022500002", "tz_shift"] == -2.0           # ET -> MT
    assert lal.loc["0022500002", "high_altitude_venue"] == 1
    assert lal.loc["0022500003", "high_altitude_venue"] == 0
    assert lal.loc["0022500003", "games_last_7"] == 2
    assert lal.loc["0022500003", "sched_rest_days"] == 1
    assert lal.loc["0022500001", "sched_rest_days"] == 10      # season opener
    assert "0012500001" not in tf.index.get_level_values("GAME_ID")


def test_unknown_venue_gives_nan_distance():
    s = SCHED.copy()
    s.loc[s["GAME_ID"] == "0022500002", "HOME_TRICODE"] = "XXX"
    tf = travel.travel_features(s).set_index(["TEAM_ID", "GAME_ID"])
    assert np.isnan(tf.loc[(LAL, "0022500002"), "travel_km"])


def test_travel_group_columns_in_feature_table():
    gl = _synthetic_gamelogs(n_days=10)
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False,
                                groups=["travel"], schedule=SCHED)
    for c in fs.FEATURE_GROUPS["travel"]:
        assert c in df.columns  # synthetic team ids aren't in SCHED: NaN, no KeyError
    assert df["travel_km"].isna().all()
