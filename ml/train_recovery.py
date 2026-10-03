"""Modèle de récupération : prédire la VFC de la nuit suivante.

Entrée  : $RUNLAB_DATA_DIR/gold/daily_features.parquet
Lancement, depuis la racine du dépôt : python -m ml.train_recovery

Sorties : métriques affichées, et si MLflow est installé, une expérience suivie dans
          $RUNLAB_DATA_DIR/mlflow (paramètres, métriques, modèle). Désactivable avec RUNLAB_MLFLOW=0.

Principes :
- validation TEMPORELLE (walk-forward) : on entraîne toujours sur le passé, on teste sur le futur ;
- chaque modèle est comparé à deux références naïves. Un modèle qui ne les bat pas ne sert à rien.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.models import ResidualModel
from ml.tracking import TRUSTED_TYPES, get_mlflow

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
GOLD_FILE = DATA_DIR / "gold" / "daily_features.parquet"
N_SPLITS = 5
TARGET = "hrv_next"

FEATURES = [
    # État de récupération actuel
    "hrv_last_night", "hrv_7d_mean", "resting_hr", "sleep_h", "sleep_score", "avg_stress",
    "body_battery_max", "body_battery_min",
    # Charge d'entraînement
    "load_total", "load_running", "load_tennis", "load_strength", "atl", "ctl", "tsb",
    # Contexte
    "is_rest_day", "last_session_hour", "is_weekend",
]


def make_dataset(gold: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables et ne garde que les jours exploitables pour l'apprentissage."""
    df = gold.sort_values("date").copy()
    df["hrv_7d_mean"] = df["hrv_last_night"].rolling(7, min_periods=4).mean()
    df["is_weekend"] = df["weekday"] >= 5
    for col in FEATURES:  # un sport jamais pratiqué n'a pas de colonne : charge nulle
        if col not in df.columns:
            df[col] = 0.0
    df[["is_rest_day", "is_weekend"]] = df[["is_rest_day", "is_weekend"]].astype(float)

    # Lignes utilisables : cible connue, nuit suivante fiable, VFC du jour connue (pour la référence)
    usable = df[TARGET].notna() & ~df["next_night_suspect"] & df["hrv_last_night"].notna()
    return df.loc[usable].reset_index(drop=True)


def candidate_models() -> dict:
    return {
        # Régression linéaire régularisée : simple, interprétable
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        # Gradient boosting : capte les effets non linéaires, gère nativement les valeurs manquantes
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, random_state=0
        ),
        # Versions résiduelles : apprennent seulement la correction de la moyenne des 7 jours
        "ridge_residual": ResidualModel(
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
        ),
        "gradient_boosting_residual": ResidualModel(
            HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=15, random_state=0)
        ),
    }


def evaluate(df: pd.DataFrame, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Validation walk-forward : MAE de chaque modèle et des références, sur chaque période de test."""
    X, y = df[FEATURES], df[TARGET]
    rows = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
        # Garantie anti-fuite : toutes les dates d'entraînement précèdent les dates de test
        assert df["date"].iloc[train_idx].max() < df["date"].iloc[test_idx].min()
        y_test = y.iloc[test_idx]
        scores = {
            # Référence 1 : "la VFC de demain sera celle d'aujourd'hui"
            "naive_today": mean_absolute_error(y_test, df["hrv_last_night"].iloc[test_idx]),
            # Référence 2 : "la VFC de demain sera ma moyenne des 7 derniers jours"
            "naive_7d_mean": mean_absolute_error(
                y_test, df["hrv_7d_mean"].iloc[test_idx].fillna(df["hrv_last_night"].iloc[test_idx])
            ),
        }
        for name, model in candidate_models().items():
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores[name] = mean_absolute_error(y_test, model.predict(X.iloc[test_idx]))
        rows.append({"fold": fold, "n_train": len(train_idx), "n_test": len(test_idx), **scores})
    return pd.DataFrame(rows).set_index("fold")


def main() -> None:
    df = make_dataset(pd.read_parquet(GOLD_FILE))
    print(f"{len(df)} jours exploitables, du {df['date'].min():%Y-%m-%d} au {df['date'].max():%Y-%m-%d}")

    results = evaluate(df)
    print("\nMAE par période de test (ms de VFC, plus bas = meilleur) :")
    print(results.round(2).to_string())

    summary = results.drop(columns=["n_train", "n_test"]).mean().sort_values()
    best_naive = summary[["naive_today", "naive_7d_mean"]].min()
    print("\nMAE moyenne :")
    for name, mae in summary.items():
        gain = (1 - mae / best_naive) * 100
        print(f"  {name:<18} {mae:6.2f}   ({gain:+5.1f} % vs meilleure référence)")

    best = summary.drop(["naive_today", "naive_7d_mean"]).idxmin()
    verdict = "bat" if summary[best] < best_naive else "ne bat PAS"
    print(f"\nMeilleur modèle : {best}, qui {verdict} la meilleure référence naïve.")

    # Modèle final : réentraîné sur toutes les données, puis enregistré dans MLflow
    final = candidate_models()[best].fit(df[FEATURES], df[TARGET])
    log_to_mlflow(df, results, summary, best, final)


def log_to_mlflow(df, results, summary, best, final) -> None:
    mlflow = get_mlflow(DATA_DIR, "recovery")
    if mlflow is None:
        return
    with mlflow.start_run(run_name=best):
        mlflow.log_params({
            "best_model": best,
            "target": TARGET,
            "n_days": len(df),
            "n_splits": N_SPLITS,
            "features": ",".join(FEATURES),
        })
        mlflow.log_metrics({f"mae_{name}": float(mae) for name, mae in summary.items()})
        best_naive = summary[["naive_today", "naive_7d_mean"]].min()
        mlflow.log_metric("skill_vs_naive", float(1 - summary[best] / best_naive))
        mlflow.sklearn.log_model(
            final, name="model", input_example=df[FEATURES].head(3), skops_trusted_types=TRUSTED_TYPES
        )
        print(f"\nExpérience enregistrée dans MLflow ({DATA_DIR / 'mlflow'}).")


if __name__ == "__main__":
    np.random.seed(0)
    main()
