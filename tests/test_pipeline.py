"""Test de bout en bout : données synthétiques -> bronze -> silver -> gold."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/generate_sample_data.py",
    "processing/build_silver.py",
    "processing/build_gold.py",
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Exécute tout le pipeline une fois, dans un dossier temporaire."""
    out = tmp_path_factory.mktemp("data")
    env = {**os.environ, "RUNLAB_DATA_DIR": str(out)}
    for script in SCRIPTS:
        subprocess.run([sys.executable, script], cwd=ROOT, env=env, check=True)
    return out


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
