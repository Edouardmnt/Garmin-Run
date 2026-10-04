"""API Garmin-Run : la source de vérité que le futur coach IA interrogera.

Lancement local, depuis la racine du dépôt :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs

Toutes les réponses sont calculées à la demande à partir des couches silver et gold.
"""

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query

from processing.feedback import (
    apply_feedback_labels,
    load_feedback,
    pending_run,
    questions_for,
    recent_signals,
    save_feedback,
    summary,
    validate,
)
from processing.insights import build_analysis
from processing.nutrition import race_nutrition
from processing.performance import (
    DISTANCES_M,
    collect_performances,
    day_adjustment,
    day_state,
    estimate_vdot,
    flat_equivalent_m,
    format_time,
    garmin_predictions,
    hr_speed_vo2max,
    personal_training_paces,
    predict_time_s,
    recent_runs,
    vo2max_history,
)
from processing.planning import build_plan

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
    """Étiquettes saisies à la main, complétées par le type déclaré dans les questionnaires."""
    path = DATA_DIR / "labels" / "run_labels.csv"
    labels = pd.read_csv(path, sep=";", decimal=",", dtype={"label": str}) if path.exists() else None
    return apply_feedback_labels(labels, load_feedback(DATA_DIR))


def full_state(gold: pd.DataFrame) -> dict:
    """État du jour enrichi : stress de la veille, Body Battery, douleur signalée récemment."""
    state = day_state(gold)
    g = gold.sort_values("date")
    last = g.iloc[-1]
    stress = g["avg_stress"].dropna() if "avg_stress" in g else pd.Series(dtype=float)
    state["stress_veille"] = None if stress.empty else float(stress.iloc[-1])
    battery = last.get("body_battery_max")
    state["body_battery_max"] = None if battery is None or pd.isna(battery) else float(battery)
    state.update(recent_signals(load_feedback(DATA_DIR), reference_day(gold)))
    return state


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
    denivele_m: int = Query(0, ge=0, le=5000, description="D+ total du parcours visé, en mètres"),
) -> dict:
    """Temps prédits : temps de base (forme de fond) et temps ajusté à la forme du jour.

    Avec `denivele_m`, le temps tient compte du D+ du parcours ; `temps_plat` reste fourni pour comparaison.
    """
    gold, perf, estimate = current_estimate()
    state = day_state(gold)
    adj, reasons = day_adjustment(state) if ajuster_au_jour else (0.0, ["Ajustement du jour désactivé"])
    garmin = read_garmin_predictions()

    targets = DISTANCES_M if distance == "toutes" else {distance: DISTANCES_M[distance]}
    results = {}
    for name, meters in targets.items():
        flat = predict_time_s(estimate["vdot"], meters)
        # Parcours vallonné : temps d'un parcours plat de distance équivalente (équivalence de Scarf)
        base = predict_time_s(estimate["vdot"], flat_equivalent_m(meters, denivele_m)) if denivele_m else flat
        adjusted = base * (1 + adj)
        results[name] = {
            "temps_plat": format_time(flat),
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

    if denivele_m:
        warnings.append(f"D+ de {denivele_m} m converti en distance de plat équivalente (1 m de montée = 7,92 m de plat).")
        warnings.append("La prédiction de la montre suppose un parcours plat.")
    return {
        "vdot": round(estimate["vdot"], 1),
        "denivele_m": denivele_m,
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


@app.get("/historique", tags=["historique"])
def historique(jours: int = Query(90, ge=7, le=730)) -> dict:
    """Série quotidienne pour les graphiques : charge par sport, ATL/CTL/TSB, VFC, FC de repos, sommeil."""
    gold = read_parquet("gold/daily_features.parquet").sort_values("date").copy()
    gold["date"] = pd.to_datetime(gold["date"])
    gold["hrv_7j"] = gold["hrv_last_night"].rolling(7, min_periods=3).mean()
    gold = gold[gold["date"] >= gold["date"].max() - pd.Timedelta(days=jours - 1)]
    sports = [c for c in gold.columns if c.startswith("load_") and c != "load_total"]
    extra = ["deep_sleep_s", "light_sleep_s", "rem_sleep_s", "awake_s", "sleep_score", "bedtime_h", "wake_h",
             "avg_stress", "max_stress", "high_stress_min", "body_battery_max", "body_battery_min", "steps"]
    cols = ["load_total", *sports, "atl", "ctl", "tsb", "hrv_last_night", "hrv_7j", "resting_hr", "sleep_h",
            *[c for c in extra if c in gold.columns]]
    out = gold[["date", *cols]].astype({"date": str})
    out["date"] = out["date"].str[:10]
    records = out.astype(object).where(out.notna(), None).to_dict(orient="records")  # NaN -> null en JSON
    return {"jours": jours, "sports": [c.removeprefix("load_") for c in sports], "series": records}


@app.get("/analyse", tags=["forme"])
def analyse() -> dict:
    """Verdict du jour et analyses rédigées : charge, forme, récupération, sommeil, nuit, journée, stress, activités."""
    gold = read_parquet("gold/daily_features.parquet")
    sports = [c.removeprefix("load_") for c in gold.columns if c.startswith("load_") and c != "load_total"]
    return build_analysis(gold, sports, full_state(gold), read_parquet("silver/activities.parquet"))


def personalisation(gold: pd.DataFrame, state: dict) -> tuple[float, int, list[str]]:
    """Adapte le plan aux signaux récents : douleur, sommeil, stress, ressenti des footings."""
    factor, ef_shift, notes = 1.0, 0, []
    week = gold.sort_values("date").tail(7)
    if state.get("douleur_forte_recente"):
        factor *= 0.75
        notes.append("Douleur forte signalée cette semaine : volume réduit de 25 %. Si elle persiste, consulte "
                     "un médecin ou un kinésithérapeute avant de reprendre les séances dures.")
    elif state.get("douleur_recente"):
        factor *= 0.85
        notes.append("Gêne signalée cette semaine : volume réduit de 15 %, surveille-la à chaque sortie.")
    sleep = week["sleep_h"].mean()
    if pd.notna(sleep) and sleep < 6.5:
        factor *= 0.9
        notes.append(f"Sommeil moyen de {sleep:.1f} h cette semaine : volume réduit de 10 %.".replace(".", ",", 1))
    stress = week["avg_stress"].mean() if "avg_stress" in week else None
    if stress is not None and pd.notna(stress) and stress > 45:
        factor *= 0.9
        notes.append(f"Stress moyen élevé cette semaine ({stress:.0f}) : volume réduit de 10 %.")
    if state.get("footings_trop_durs"):
        ef_shift = 15
        notes.append("Tu as ressenti plusieurs footings comme difficiles : allures d'endurance ralenties de 15 s/km.")
    return max(factor, 0.6), ef_shift, notes


@app.get("/planning", tags=["planning"])
def planning(
    distance: Literal["5k", "10k", "semi", "marathon"] = "10k",
    date_course: date | None = Query(None, description="Date de la course visée (AAAA-MM-JJ)"),
    seances_par_semaine: int = Query(3, ge=2, le=6, description="Nombre de sorties de course par semaine"),
    jours_tennis: str = Query("", description="Jours de tennis, 0 = lundi ... 6 = dimanche, ex. 1,3"),
    jour_sortie_longue: int = Query(6, ge=0, le=6, description="0 = lundi ... 6 = dimanche"),
    denivele_m: int = Query(0, ge=0, le=5000, description="D+ du parcours de la course"),
) -> dict:
    """Plan d'entraînement jusqu'à la course : phases, séances détaillées, allures et FC personnelles."""
    gold, _, estimate = current_estimate()
    today = reference_day(gold)
    if date_course is not None and date_course <= today:
        raise HTTPException(422, "La date de course doit être postérieure au dernier jour de données.")
    try:
        tennis = sorted({int(d) for d in jours_tennis.split(",") if d.strip() != ""})
    except ValueError:
        raise HTTPException(422, "jours_tennis doit être une liste de chiffres de 0 à 6, ex. 1,3") from None
    if any(d not in range(7) for d in tennis):
        raise HTTPException(422, "jours_tennis doit contenir des chiffres de 0 (lundi) à 6 (dimanche).")

    activities = read_parquet("silver/activities.parquet")
    hr_max = float(activities["max_hr"].max())
    runs = recent_runs(activities, read_labels(), today, 120)
    zones = personal_training_paces(runs, hr_max, estimate["vdot"])["zones"]
    paces = {k: {"rapide_s": z["recommandation"]["allure_rapide_s"], "lente_s": z["recommandation"]["allure_lente_s"],
                 "fc_cible": z["recommandation"]["fc_cible"]} for k, z in zones.items()}
    recent = runs[runs["date"] >= pd.Timestamp(today - timedelta(days=28))]
    base_km = float(recent["distance_m"].sum() / 1000 / 4)
    meters = DISTANCES_M[distance]
    race_time = predict_time_s(estimate["vdot"], flat_equivalent_m(meters, denivele_m))
    state = full_state(gold)
    verdict = build_analysis(gold, [], state)["verdict"]
    factor, ef_shift, adaptations = personalisation(gold, state)
    plan = build_plan(today, distance, date_course, seances_par_semaine, tennis, jour_sortie_longue,
                      base_km, paces, race_time / (meters / 1000), verdict["niveau"], factor, ef_shift)
    return {
        "objectif": {"distance": distance, "date_course": None if date_course is None else date_course.isoformat(),
                     "temps_vise": format_time(race_time), "allure_course": pace_txt(race_time / (meters / 1000)),
                     "denivele_m": denivele_m},
        "volume_actuel_km_semaine": round(base_km, 1),
        "verdict_du_jour": verdict,
        "personnalisation": adaptations,
        **plan,
    }


@app.get("/questionnaire", tags=["questionnaire"])
def questionnaire() -> dict:
    """Questionnaire de la dernière sortie de course sans réponse (7 derniers jours), ou rien si tout est à jour."""
    gold = read_parquet("gold/daily_features.parquet")
    runs = recent_runs(read_parquet("silver/activities.parquet"), read_labels(), reference_day(gold), 7)
    run = pending_run(runs, load_feedback(DATA_DIR), reference_day(gold))
    if run is None:
        return {"en_attente": None}
    kind = run["kind"] if isinstance(run["kind"], str) else None
    return {"en_attente": {
        "activity_id": int(run["activity_id"]), "date": run["date"].date().isoformat(),
        "distance_km": round(float(run["distance_m"]) / 1000, 2), "duree_min": round(float(run["duration_s"]) / 60),
        "allure": pace_txt(float(run["raw_pace_s"])), "type_suppose": kind, "questions": questions_for(kind),
    }}


@app.post("/questionnaire/{activity_id}", tags=["questionnaire"])
def repondre(activity_id: int, reponses: dict = Body(..., embed=True)) -> dict:
    """Enregistre les réponses au questionnaire d'une sortie."""
    gold = read_parquet("gold/daily_features.parquet")
    runs = recent_runs(read_parquet("silver/activities.parquet"), read_labels(), reference_day(gold), 30)
    match = runs[runs["activity_id"] == activity_id]
    if match.empty:
        raise HTTPException(404, "Sortie introuvable parmi les 30 derniers jours.")
    run = match.iloc[0]
    if any(f["activity_id"] == activity_id for f in load_feedback(DATA_DIR)):
        raise HTTPException(409, "Ce questionnaire a déjà été rempli.")
    questions = questions_for(run["kind"] if isinstance(run["kind"], str) else None)
    errors = validate(reponses, questions)
    if errors:
        raise HTTPException(422, errors)
    record = save_feedback(DATA_DIR, activity_id, run["date"].date().isoformat(), reponses)
    return {"enregistre": record, "message": "Merci ! Tes réponses affinent tes allures, ton planning et tes prédictions."}


@app.get("/questionnaire/bilan", tags=["questionnaire"])
def bilan_questionnaires() -> dict:
    """Ce que disent les questionnaires sur la justesse des prédictions, des allures et du verdict du jour."""
    return summary(load_feedback(DATA_DIR))


@app.get("/nutrition", tags=["planning"])
def nutrition(
    distance: Literal["5k", "10k", "semi", "marathon"] = "semi",
    temperature_c: float = Query(15, ge=-10, le=45, description="Température prévue le jour de la course"),
    denivele_m: int = Query(0, ge=0, le=5000),
) -> dict:
    """Nutrition et hydratation avant, pendant et après la course, selon ta durée prévue et la température."""
    _, _, estimate = current_estimate()
    meters = DISTANCES_M[distance]
    duration_s = predict_time_s(estimate["vdot"], flat_equivalent_m(meters, denivele_m))
    label = {"5k": "5 km", "10k": "10 km", "semi": "semi-marathon", "marathon": "marathon"}[distance]
    plan = race_nutrition(label, duration_s / 60, duration_s / (meters / 1000), temperature_c)
    return {"distance": distance, "temps_prevu": format_time(duration_s), **plan}


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
