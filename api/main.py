"""API Garmin-Run : la source de vérité que le futur coach IA interrogera.

Lancement local, depuis la racine du dépôt :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs

Toutes les réponses sont calculées à la demande à partir des couches silver et gold.
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from processing.performance import (
    DISTANCES_M,
    collect_performances,
    day_adjustment,
    day_state,
    estimate_vdot,
    format_time,
    garmin_predictions,
    hr_speed_vo2max,
    personal_training_paces,
    predict_time_s,
    recent_runs,
    vo2max_history,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))

app = FastAPI(
    title="Garmin-Run API",
    description="Forme du jour, temps prédits et allures d'entraînement, à partir des données Garmin.",
    version="1.0.0",
)


# --- Accès aux données ---------------------------------------------------------------------------

def read_parquet(relative: str) -> pd.DataFrame:
    path = DATA_DIR / relative
    if not path.exists():
        raise HTTPException(503, f"Données indisponibles : {relative}. Le pipeline a-t-il tourné ?")
    return pd.read_parquet(path)


def read_labels() -> pd.DataFrame | None:
    path = DATA_DIR / "labels" / "run_labels.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=";", decimal=",", dtype={"label": str})


def reference_day(gold: pd.DataFrame) -> date:
    """Jour de référence : le dernier jour présent dans les données (et non l'horloge du serveur)."""
    return pd.to_datetime(gold["date"]).max().date()


def current_estimate():
    gold = read_parquet("gold/daily_features.parquet")
    activities = read_parquet("silver/activities.parquet")
    labels = read_labels()
    perf = collect_performances(activities, labels)
    hr_max = float(activities["max_hr"].max())
    hr_rest = float(gold["resting_hr"].median())
    physio = {
        "vo2max_montre": vo2max_history(activities),
        "relation_fc_vitesse": hr_speed_vo2max(activities, labels, hr_rest, hr_max),
    }
    estimate = estimate_vdot(perf, physio, reference_day(gold))
    if estimate is None:
        raise HTTPException(404, "Ni VO2 max récente ni performance exploitable : impossible d'estimer ta forme.")
    return gold, perf, estimate


def read_garmin_predictions() -> dict:
    path = DATA_DIR / "raw" / "race_predictions.json"
    if not path.exists():
        return {}
    return garmin_predictions(json.loads(path.read_text(encoding="utf-8")))


def pace_txt(seconds_per_km: float) -> str:
    return f"{format_time(seconds_per_km)}/km"


# --- Points d'accès ------------------------------------------------------------------------------

@app.get("/health", tags=["technique"])
def health() -> dict:
    """Sonde de disponibilité utilisée par Kubernetes."""
    return {"status": "ok"}


@app.get("/forme", tags=["forme"])
def forme() -> dict:
    """État du dernier jour connu : VFC et sommeil par rapport à ta normale, charge et fraîcheur."""
    state = day_state(read_parquet("gold/daily_features.parquet"))
    adj, reasons = day_adjustment(state)
    return {**state, "ajustement_chrono_pct": round(adj * 100, 1), "explications": reasons}


@app.get("/predictions", tags=["performance"])
def predictions(
    distance: Literal["5k", "10k", "semi", "marathon", "toutes"] = "toutes",
    ajuster_au_jour: bool = Query(True, description="Appliquer l'état du jour (nuit, fatigue accumulée)"),
) -> dict:
    """Temps prédits : temps de base (forme de fond) et temps ajusté à la forme du jour."""
    gold, perf, estimate = current_estimate()
    state = day_state(gold)
    adj, reasons = day_adjustment(state) if ajuster_au_jour else (0.0, ["Ajustement du jour désactivé"])
    garmin = read_garmin_predictions()

    targets = DISTANCES_M if distance == "toutes" else {distance: DISTANCES_M[distance]}
    results = {}
    for name, meters in targets.items():
        base = predict_time_s(estimate["vdot"], meters)
        adjusted = base * (1 + adj)
        results[name] = {
            "temps_base": format_time(base),
            "temps_ajuste": format_time(adjusted),
            "allure_course": pace_txt(adjusted / (meters / 1000)),
            "temps_base_s": round(base),
            "temps_ajuste_s": round(adjusted),
            "prediction_montre": format_time(garmin[name]) if name in garmin else None,
        }

    warnings = []
    if estimate["courses_etiquetees"] == 0:
        warnings.append("Aucune course étiquetée : les estimations physiologiques ne sont pas calibrées sur tes chronos réels.")
    recent = perf[perf["date"] >= pd.Timestamp(reference_day(gold)) - pd.Timedelta(days=56)]
    if distance in ("semi", "marathon", "toutes") and (recent.empty or recent["distance_m"].max() < 15000):
        warnings.append("Peu d'efforts longs récents : le semi et le marathon supposent une endurance spécifique.")

    return {
        "vdot": round(estimate["vdot"], 1),
        "estimation": estimate["composantes"],
        "ajustement_du_jour_pct": round(adj * 100, 1),
        "explications_ajustement": reasons,
        "predictions": results,
        "avertissements": warnings,
    }


@app.get("/allures", tags=["performance"])
def allures(fenetre_jours: int = Query(120, ge=14, le=365, description="Période d'observation des séances")) -> dict:
    """Allures d'entraînement personnelles, calculées à partir de tes séances réelles.

    Pour l'EF, le tempo et le fractionné : allures et FC observées dans tes séances étiquetées,
    ton modèle FC -> allure, et la valeur théorique (VDOT) pour comparaison. La recommandation
    retient l'observation quand il y a assez de séances, sinon le modèle, sinon la théorie.
    """
    gold, _, estimate = current_estimate()
    activities = read_parquet("silver/activities.parquet")
    hr_max = float(activities["max_hr"].max())
    runs = recent_runs(activities, read_labels(), reference_day(gold), fenetre_jours)
    paces = personal_training_paces(runs, hr_max, estimate["vdot"])
    return {"fenetre_jours": fenetre_jours, "fc_max": round(hr_max), "vdot": round(estimate["vdot"], 1), **paces}


@app.get("/seances", tags=["historique"])
def seances(limite: int = Query(10, ge=1, le=100)) -> dict:
    """Dernières sorties de course, avec leur type : ton étiquette, sinon la suggestion des règles."""
    labels = read_labels()
    if labels is None:
        raise HTTPException(404, "Aucun fichier d'étiquetage : lance scripts/make_run_labels.py.")
    labels = labels.sort_values("date", ascending=False).head(limite)
    out = []
    for _, r in labels.iterrows():
        labelled = isinstance(r["label"], str) and r["label"].strip() != ""
        out.append({
            "date": r["date"],
            "distance_km": float(r["distance_km"]),
            "duree_min": float(r["duration_min"]),
            "allure": None if pd.isna(r["pace_min_km"]) else pace_txt(float(r["pace_min_km"]) * 60),
            "fc_moyenne": None if pd.isna(r["avg_hr"]) else float(r["avg_hr"]),
            "type": r["label"] if labelled else r["suggestion"],
            "type_source": "etiquette" if labelled else "regles",
        })
    return {"seances": out}
