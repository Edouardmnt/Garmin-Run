"""Tests de l'harmonisation multi-sources et de l'expérience de transfert."""

import numpy as np
import pandas as pd

from ml.features import FEATURES, TARGET_REL, build_features, garmin_to_common, lifesnaps_to_common, load_lifesnaps, usable_rows
from ml.train_transfer import evaluate_transfer


def test_harmonisation_lifesnaps():
    raw = pd.DataFrame({
        "id": ["a", "a"], "date": ["2021-05-01", "2021-05-01"],  # doublon le même jour
        "rmssd": [80.0, 90.0], "resting_hr": [60, 60], "minutesAsleep": [420, 420],
        "minutes_in_default_zone_1": [10, 10], "minutes_in_default_zone_2": [2, 2], "minutes_in_default_zone_3": [0, 0],
    })
    common = lifesnaps_to_common(raw)
    assert len(common) == 1  # une ligne par personne et par jour
    row = common.iloc[0]
    assert row["hrv"] == 85.0 and row["sleep_h"] == 7.0
    assert row["load"] == 10 * 1.5 + 2 * 3.5  # TRIMP d'Edwards


def make_common(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "person_id": "p", "source": "test",
        "date": pd.date_range("2026-01-01", periods=n), "hrv": rng.normal(60, 5, n),
        "resting_hr": rng.normal(50, 2, n), "sleep_h": rng.normal(7, 0.5, n), "load": rng.uniform(0, 100, n),
    })


def test_aucune_fuite_de_l_avenir():
    common = make_common()
    before = build_features(common)
    modified = common.copy()
    modified.loc[40:, ["hrv", "resting_hr", "sleep_h", "load"]] *= 3  # on change radicalement l'avenir
    after = build_features(modified)
    # Les variables des jours 0 à 39 ne doivent pas bouger d'un iota
    pd.testing.assert_frame_equal(before.loc[:39, FEATURES], after.loc[:39, FEATURES])


def test_calendrier_complet_le_decalage_vise_le_lendemain():
    common = make_common(30).drop(index=[10, 11])  # deux jours sans montre
    feats = build_features(common)
    assert len(feats) == 30  # les trous sont réintroduits comme jours vides
    day9 = feats[feats["date"] == "2026-01-10"].iloc[0]
    assert np.isnan(day9["hrv_next"])  # le lendemain (jour 10) est inconnu, pas le jour 12


def test_experience_de_transfert(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    me = usable_rows(build_features(garmin_to_common(gold)))
    others = usable_rows(build_features(load_lifesnaps(data_dir)))
    assert others["person_id"].nunique() > 1 and me[TARGET_REL].notna().all()

    results = evaluate_transfer(me, others, n_splits=3)
    expected = {f"{s}/{m}" for s in ["personnel", "global", "global_plus_personnel"] for m in ["ridge", "gradient_boosting"]}
    assert expected <= set(results.columns)
    assert np.isfinite(results.drop(columns=["n_train_moi", "n_test"]).to_numpy()).all()
