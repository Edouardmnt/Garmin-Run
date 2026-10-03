"""Tests du modèle de récupération (sans MLflow)."""

import numpy as np
import pandas as pd

from ml.train_recovery import FEATURES, TARGET, evaluate, make_dataset


def test_dataset_exclut_les_jours_inexploitables(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    df = make_dataset(gold)
    assert df[TARGET].notna().all()
    assert not df["next_night_suspect"].any()
    assert set(FEATURES) <= set(df.columns)
    assert len(df) < len(gold)  # les nuits absentes ont bien été retirées


def test_evaluation_temporelle_sans_fuite(data_dir):
    df = make_dataset(pd.read_parquet(data_dir / "gold" / "daily_features.parquet"))
    results = evaluate(df, n_splits=3)  # l'absence de fuite est vérifiée dans evaluate()
    assert len(results) == 3
    assert results["n_train"].is_monotonic_increasing  # fenêtre d'entraînement croissante
    scores = results.drop(columns=["n_train", "n_test"])
    expected = {"naive_today", "naive_7d_mean", "ridge", "gradient_boosting", "ridge_residual", "gradient_boosting_residual"}
    assert expected <= set(scores.columns)
    assert np.isfinite(scores.to_numpy()).all()


def test_modele_residuel_sans_signal_retombe_sur_la_reference(data_dir):
    """Avec une régularisation extrême, la correction est nulle : prédiction = moyenne des 7 jours."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    from ml.models import ResidualModel

    df = make_dataset(pd.read_parquet(data_dir / "gold" / "daily_features.parquet"))
    X = df[FEATURES]
    model = ResidualModel(make_pipeline(SimpleImputer(), Ridge(alpha=1e12))).fit(X, df[TARGET])
    baseline = df["hrv_7d_mean"].fillna(df["hrv_last_night"])
    correction = df[TARGET] - baseline
    assert np.allclose(model.predict(X), baseline + correction.mean(), atol=0.01)
