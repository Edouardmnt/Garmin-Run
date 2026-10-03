"""Expérience de transfert : les données d'autres personnes aident-elles à prédire MA récupération ?

Lancement, depuis la racine du dépôt : python -m ml.train_transfer

Trois stratégies d'entraînement, toutes évaluées sur MES données, avec la même validation temporelle :
- personnel           : mon historique passé uniquement
- global              : les participants de LifeSnaps uniquement (aucune de mes données)
- global_plus_personnel : LifeSnaps + mon historique passé, mes jours ayant un poids renforcé

Le modèle prédit une variation RELATIVE de VFC (comparable entre montres et entre personnes),
convertie ensuite en millisecondes pour être comparée aux références naïves.

Source publique : Yfantidou et al. (2022), LifeSnaps, doi:10.5281/zenodo.7229547 (CC BY 4.0).
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

from ml.features import FEATURES, TARGET_REL, build_features, garmin_to_common, load_lifesnaps, usable_rows
from ml.tracking import TRUSTED_TYPES, get_mlflow

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
N_SPLITS = 5
PERSONAL_WEIGHT = 5.0  # un de mes jours compte comme 5 jours d'un participant LifeSnaps
NAIVE = ["naive_today", "naive_7d_mean"]


def candidate_models() -> dict:
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0
        ),
    }


def fit(model, X, y, weights=None):
    """Entraîne un modèle avec des poids par ligne, que ce soit un pipeline ou un estimateur simple."""
    if weights is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": weights})
    return model.fit(X, y, sample_weight=weights)


def to_ms(rows: pd.DataFrame, pred_rel) -> pd.Series:
    """Convertit une variation relative prédite en VFC (ms)."""
    return rows["hrv_7d"] + np.asarray(pred_rel) * rows["hrv_base28"]


def training_sets(me_train: pd.DataFrame, others: pd.DataFrame | None) -> dict:
    sets = {"personnel": (me_train[FEATURES], me_train[TARGET_REL], None)}
    if others is not None and len(others):
        sets["global"] = (others[FEATURES], others[TARGET_REL], None)
        X = pd.concat([others[FEATURES], me_train[FEATURES]], ignore_index=True)
        y = pd.concat([others[TARGET_REL], me_train[TARGET_REL]], ignore_index=True)
        w = np.r_[np.ones(len(others)), np.full(len(me_train), PERSONAL_WEIGHT)]
        sets["global_plus_personnel"] = (X, y, w)
    return sets


def evaluate_transfer(me: pd.DataFrame, others: pd.DataFrame | None, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """MAE (ms) sur mes données, par période de test, pour chaque stratégie x modèle et les références."""
    me = me.sort_values("date").reset_index(drop=True)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(me)):
        train, test = me.iloc[train_idx], me.iloc[test_idx]
        assert train["date"].max() < test["date"].min()  # jamais de fuite de mon propre avenir
        y_ms = test["hrv_next"]
        scores = {
            "naive_today": mean_absolute_error(y_ms, test["hrv"]),
            "naive_7d_mean": mean_absolute_error(y_ms, test["hrv_7d"]),
        }
        for strategy, (X, y, w) in training_sets(train, others).items():
            for name, model in candidate_models().items():
                fit(model, X, y, w)
                scores[f"{strategy}/{name}"] = mean_absolute_error(y_ms, to_ms(test, model.predict(test[FEATURES])))
        rows.append({"fold": fold, "n_train_moi": len(train), "n_test": len(test), **scores})
    return pd.DataFrame(rows).set_index("fold")


def main() -> None:
    gold = pd.read_parquet(DATA_DIR / "gold" / "daily_features.parquet")
    me = usable_rows(build_features(garmin_to_common(gold)))
    print(f"Mes données : {len(me)} jours utilisables")

    lifesnaps = load_lifesnaps(DATA_DIR)
    if lifesnaps is None:
        others = None
        print("LifeSnaps absent : seule la stratégie 'personnel' est évaluée.")
    else:
        others = usable_rows(build_features(lifesnaps))
        print(f"LifeSnaps : {len(others)} jours utilisables, {others['person_id'].nunique()} participants")

    results = evaluate_transfer(me, others)
    scores = results.drop(columns=["n_train_moi", "n_test"])
    table = scores.T
    table["moyenne"] = scores.mean()
    table["2 dernières"] = scores.tail(2).mean()  # périodes avec le plus d'historique personnel
    table = table.sort_values("moyenne")
    print("\nMAE (ms de VFC, plus bas = meilleur), par période de test :")
    print(table.round(2).to_string())

    best_naive = table.loc[NAIVE, "moyenne"].min()
    best = table.drop(index=NAIVE)["moyenne"].idxmin()
    gain = (1 - table.loc[best, "moyenne"] / best_naive) * 100
    print(f"\nMeilleure combinaison : {best} ({gain:+.1f} % vs meilleure référence naïve)")

    log_to_mlflow(me, others, table, best)


def log_to_mlflow(me, others, table, best) -> None:
    mlflow = get_mlflow(DATA_DIR, "recovery-transfer")
    if mlflow is None:
        return
    strategy, name = best.split("/")
    # Modèle final : la meilleure combinaison, réentraînée sur toutes les données disponibles
    X, y, w = training_sets(me, others)[strategy]
    final = fit(candidate_models()[name], X, y, w)
    with mlflow.start_run(run_name=best):
        mlflow.log_params({
            "best": best,
            "n_days_me": len(me),
            "n_days_lifesnaps": 0 if others is None else len(others),
            "n_people_lifesnaps": 0 if others is None else others["person_id"].nunique(),
            "personal_weight": PERSONAL_WEIGHT,
            "features": ",".join(FEATURES),
        })
        for combo, row in table.iterrows():
            key = combo.replace("/", "__")
            mlflow.log_metric(f"mae_{key}", float(row["moyenne"]))
            mlflow.log_metric(f"mae_recent_{key}", float(row["2 dernières"]))
        mlflow.sklearn.log_model(
            final, name="model", input_example=me[FEATURES].head(3), skops_trusted_types=TRUSTED_TYPES
        )
    print(f"Expérience enregistrée dans MLflow ({DATA_DIR / 'mlflow'}, expérience 'recovery-transfer').")


if __name__ == "__main__":
    main()
