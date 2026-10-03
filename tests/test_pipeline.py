"""Test de bout en bout : données synthétiques -> bronze -> silver -> gold."""

import pandas as pd


def test_les_trois_couches_existent(data_dir):
    assert (data_dir / "raw" / "activities.json").exists()
    assert (data_dir / "silver" / "activities.parquet").exists()
    assert (data_dir / "silver" / "daily.parquet").exists()
    assert (data_dir / "gold" / "daily_features.parquet").exists()


def test_silver_harmonise_les_sports(data_dir):
    act = pd.read_parquet(data_dir / "silver" / "activities.parquet")
    assert not act.empty
    # Les noms Garmin (tennis_v2, treadmill_running...) sont harmonisés
    assert set(act["sport"]) <= {"running", "tennis", "strength", "walking", "hiking", "other"}
    # L'allure n'existe que pour la course
    assert act.loc[act["sport"] != "running", "pace_min_km"].isna().all()


def test_silver_une_ligne_par_jour(data_dir):
    daily = pd.read_parquet(data_dir / "silver" / "daily.parquet")
    assert daily["date"].is_unique


def test_gold_charge_positive_et_jours_sans_activite_a_zero(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    load_cols = [c for c in gold.columns if c.startswith("load_")]
    assert gold[load_cols].notna().all().all()
    assert (gold[load_cols] >= 0).all().all()
    assert (gold["load_total"] == 0).any()


def test_gold_nuit_absente_n_est_pas_une_nuit_a_zero(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    absentes = gold[~gold["night_tracked"]]
    assert not absentes.empty
    assert absentes["sleep_s"].isna().all()
    assert (gold["sleep_s"].dropna() > 0).all()


def test_gold_cible_est_la_nuit_suivante(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet").reset_index(drop=True)
    pd.testing.assert_series_equal(
        gold["hrv_next"].iloc[:-1],
        gold["hrv_last_night"].iloc[1:].reset_index(drop=True),
        check_names=False,
    )


def test_gold_contexte_des_seances(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    # L'heure de la dernière séance n'existe que les jours d'entraînement
    assert gold.loc[gold["is_rest_day"], "last_session_hour"].isna().all()
    assert gold.loc[~gold["is_rest_day"], "last_session_hour"].between(0, 24).all()
    assert gold["weekday"].between(0, 6).all()
