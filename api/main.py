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
    efforts_by_activity,
    load_feedback,
    pending_run,
    prediction_bias,
    questions_for,
    recent_signals,
    save_feedback,
    summary,
    validate,
)
from processing.goals import (
    active_goal,
    create_goal,
    delete_goal,
    goal_km,
    goal_label,
    load_goals,
    set_active,
    validate_goal,
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
from processing.planning import build_plan, category_for
from processing.watch import load_history, send_session, to_garmin_workout

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


def estimation_inputs():
    """Données nécessaires à l'estimation du niveau : performances et sources physiologiques."""
    gold = read_parquet("gold/daily_features.parquet")
    activities = read_parquet("silver/activities.parquet")
    labels = read_labels()
    perf = collect_performances(activities, labels, efforts_by_activity(load_feedback(DATA_DIR)))
    hr_max = float(activities["max_hr"].max())
    hr_rest = float(gold["resting_hr"].median())
    physio = {
        "vo2max_montre": vo2max_history(activities),
        "relation_fc_vitesse": hr_speed_vo2max(activities, labels, hr_rest, hr_max),
    }
    return gold, perf, physio


def estimate_on(day: date, perf: pd.DataFrame, physio: dict) -> dict | None:
    """Niveau estimé tel qu'il était connu à une date : seules les données antérieures sont utilisées."""
    limit = pd.Timestamp(day)
    past = {name: series[series["date"] <= limit] for name, series in physio.items()}
    return estimate_vdot(perf[perf["date"] <= limit], past, day)


def current_estimate():
    gold, perf, physio = estimation_inputs()
    estimate = estimate_vdot(perf, physio, reference_day(gold))
    if estimate is None:
        raise HTTPException(404, "Ni VO2 max récente ni performance exploitable : impossible d'estimer ta forme.")
    return gold, perf, estimate


def feedback_bias() -> tuple[float, int]:
    """Correction des temps prédits déclarée dans les questionnaires d'après-course (en %)."""
    return prediction_bias(load_feedback(DATA_DIR))


def race_time_s(vdot: float, meters: float, denivele_m: int = 0, bias_pct: float | None = None) -> float:
    """Temps prédit sur une distance : VDOT, D+ (équivalence de Scarf), correction des questionnaires."""
    bias = feedback_bias()[0] if bias_pct is None else bias_pct
    return predict_time_s(vdot, flat_equivalent_m(meters, denivele_m)) * (1 + bias / 100)


def distance_from(distance: str, distance_km: float | None) -> tuple[str, float, str]:
    """(clé, mètres, libellé) : distance classique, ou distance libre en km si elle est fournie."""
    if distance_km:
        return "personnalisee", distance_km * 1000, f"{distance_km:g} km".replace(".", ",")
    labels = {"5k": "5 km", "10k": "10 km", "semi": "semi-marathon", "marathon": "marathon"}
    return distance, DISTANCES_M[distance], labels[distance]


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
    distance_km: float | None = Query(None, ge=1, le=100, description="Distance libre en km (remplace `distance`)"),
) -> dict:
    """Temps prédits : temps de base (forme de fond) et temps ajusté à la forme du jour.

    Avec `denivele_m`, le temps tient compte du D+ du parcours ; `temps_plat` reste fourni pour comparaison.
    """
    gold, perf, estimate = current_estimate()
    state = day_state(gold)
    adj, reasons = day_adjustment(state) if ajuster_au_jour else (0.0, ["Ajustement du jour désactivé"])
    garmin = read_garmin_predictions()

    bias, n_answers = feedback_bias()
    if distance_km:
        targets = {"personnalisee": distance_km * 1000}
    else:
        targets = DISTANCES_M if distance == "toutes" else {distance: DISTANCES_M[distance]}
    results = {}
    for name, meters in targets.items():
        flat = race_time_s(estimate["vdot"], meters, 0, bias)
        # Parcours vallonné : temps d'un parcours plat de distance équivalente (équivalence de Scarf)
        base = race_time_s(estimate["vdot"], meters, denivele_m, bias)
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

    if bias:
        sens = "allongés" if bias > 0 else "raccourcis"
        warnings.append(f"Temps {sens} de {abs(bias):.1f} % d'après tes réponses aux questionnaires après course "
                        f"({n_answers} réponse(s) sur la justesse des prédictions).".replace(".", ",", 1))
    if denivele_m:
        warnings.append(f"D+ de {denivele_m} m converti en distance de plat équivalente (1 m de montée = 7,92 m de plat).")
        warnings.append("La prédiction de la montre suppose un parcours plat.")
    return {
        "vdot": round(estimate["vdot"], 1),
        "denivele_m": denivele_m,
        "correction_questionnaires_pct": bias,
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
    distance_km: float | None = Query(None, ge=1, le=100, description="Distance libre en km (remplace `distance`)"),
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
    key, meters, label = distance_from(distance, distance_km)
    family = category_for(meters / 1000) if distance_km else distance
    race_time = race_time_s(estimate["vdot"], meters, denivele_m)
    state = full_state(gold)
    verdict = build_analysis(gold, [], state)["verdict"]
    factor, ef_shift, adaptations = personalisation(gold, state)
    plan = build_plan(today, family, date_course, seances_par_semaine, tennis, jour_sortie_longue,
                      base_km, paces, race_time / (meters / 1000), verdict["niveau"], factor, ef_shift,
                      race_km=meters / 1000, race_label=label)
    return {
        "objectif": {"distance": key, "distance_km": round(meters / 1000, 3), "libelle": label,
                     "date_course": None if date_course is None else date_course.isoformat(),
                     "temps_vise": format_time(race_time), "allure_course": pace_txt(race_time / (meters / 1000)),
                     "denivele_m": denivele_m},
        "volume_actuel_km_semaine": round(base_km, 1),
        "verdict_du_jour": verdict,
        "personnalisation": adaptations,
        **plan,
    }


# --- Objectifs ------------------------------------------------------------------------------------

def goal_status(goal: dict, estimate: dict, today: date) -> dict:
    """Où en est un objectif : jours restants, temps prédit aujourd'hui, écart au temps visé."""
    predicted = race_time_s(estimate["vdot"], goal_km(goal) * 1000, goal["denivele_m"])
    days_left = (date.fromisoformat(goal["date_course"]) - today).days
    out = {**goal, "libelle": goal_label(goal), "distance_km": goal_km(goal), "jours_restants": days_left,
           "temps_predit": format_time(predicted),
           "temps_predit_s": round(predicted), "temps_vise": None, "ecart_s": None, "statut": "sans temps visé"}
    if goal.get("temps_vise_s"):
        gap = predicted - goal["temps_vise_s"]
        out.update({"temps_vise": format_time(goal["temps_vise_s"]), "ecart_s": round(gap),
                    "ecart_pct": round(gap / goal["temps_vise_s"] * 100, 1),
                    "statut": "dans les temps" if gap <= 0 else "proche" if gap <= 0.02 * goal["temps_vise_s"]
                    else "en retard"})
    if days_left < 0:
        out["statut"] = "course passée"
    return out


@app.get("/objectifs", tags=["objectifs"])
def objectifs() -> dict:
    """Tous tes objectifs, avec où tu en es pour chacun."""
    gold, _, estimate = current_estimate()
    today = reference_day(gold)
    goals = sorted(load_goals(DATA_DIR), key=lambda g: g["date_course"])
    return {"objectifs": [goal_status(g, estimate, today) for g in goals]}


@app.post("/objectifs", tags=["objectifs"])
def nouvel_objectif(objectif: dict = Body(..., embed=True)) -> dict:
    """Crée un objectif (nom, distance, date_course, temps_vise, denivele_m, seances_par_semaine,
    jours_tennis, jour_sortie_longue). Il devient l'objectif actif."""
    gold = read_parquet("gold/daily_features.parquet")
    errors = validate_goal(objectif, reference_day(gold))
    if errors:
        raise HTTPException(422, errors)
    return {"objectif": create_goal(DATA_DIR, objectif)}


@app.post("/objectifs/{goal_id}/activer", tags=["objectifs"])
def activer_objectif(goal_id: str) -> dict:
    goal = set_active(DATA_DIR, goal_id)
    if goal is None:
        raise HTTPException(404, "Objectif introuvable.")
    return {"objectif": goal}


@app.delete("/objectifs/{goal_id}", tags=["objectifs"])
def supprimer_objectif(goal_id: str) -> dict:
    if not delete_goal(DATA_DIR, goal_id):
        raise HTTPException(404, "Objectif introuvable.")
    return {"supprime": goal_id}


@app.get("/objectifs/{goal_id}/suivi", tags=["objectifs"])
def suivi_objectif(goal_id: str, semaines: int = Query(12, ge=2, le=52)) -> dict:
    """Évolution, semaine après semaine, de ton temps prédit sur la course visée, face au temps visé."""
    goal = next((g for g in load_goals(DATA_DIR) if g["id"] == goal_id), None)
    if goal is None:
        raise HTTPException(404, "Objectif introuvable.")
    gold, perf, physio = estimation_inputs()
    today = reference_day(gold)
    bias = feedback_bias()[0]
    points = []
    for weeks_ago in range(semaines - 1, -1, -1):
        day = today - timedelta(weeks=weeks_ago)
        estimate = estimate_on(day, perf, physio)
        if estimate:
            t = race_time_s(estimate["vdot"], goal_km(goal) * 1000, goal["denivele_m"], bias)
            points.append({"date": day.isoformat(), "temps_predit_s": round(t), "temps_predit": format_time(t)})
    return {"objectif": goal_status(goal, current_estimate()[2], today), "evolution": points}


# --- Planning de l'objectif actif et montre -------------------------------------------------------

@app.get("/planning/actif", tags=["planning"])
def planning_actif() -> dict:
    """Planning de l'objectif actif ; sans objectif, deux semaines de développement général."""
    gold = read_parquet("gold/daily_features.parquet")
    goal = active_goal(DATA_DIR, reference_day(gold))
    if goal is None:
        return {**planning(distance="10k", date_course=None, seances_par_semaine=3, jours_tennis="",
                           jour_sortie_longue=6, denivele_m=0, distance_km=None), "objectif_actif": None}
    preset = goal.get("distance") in DISTANCES_M
    plan = planning(distance=goal["distance"] if preset else "10k", date_course=date.fromisoformat(goal["date_course"]),
                    seances_par_semaine=goal["seances_par_semaine"],
                    jours_tennis=",".join(str(d) for d in goal["jours_tennis"]),
                    jour_sortie_longue=goal["jour_sortie_longue"], denivele_m=goal["denivele_m"],
                    distance_km=None if preset else goal_km(goal))
    return {**plan, "objectif_actif": goal}


def today_date() -> date:
    """La vraie date du jour (remplaçable dans les tests)."""
    return date.today()


def todays_session() -> dict | None:
    """Séance prévue à la VRAIE date du jour, même si la dernière synchronisation date d'hier."""
    today = today_date().isoformat()
    plan = planning_actif()
    return next((s for w in plan["semaines"] for s in w["seances"] if s["date"] == today), None)


@app.get("/montre/seance-du-jour", tags=["montre"])
def seance_du_jour() -> dict:
    """La séance du jour, telle qu'elle sera envoyée à la montre, et l'historique des envois."""
    session = todays_session()
    return {"seance": session, "entrainement_garmin": None if session is None else to_garmin_workout(session),
            "envois": load_history(DATA_DIR)}


@app.post("/montre/envoyer", tags=["montre"])
def envoyer_a_la_montre() -> dict:
    """Place la séance du jour dans ton calendrier Garmin ; ta montre l'affiche à sa prochaine synchronisation."""
    session = todays_session()
    if session is None:
        return {"statut": "repos", "message": "Pas de séance de course prévue aujourd'hui."}
    from ingestion.garmin_export import connect

    try:
        client = connect()
    except Exception as exc:
        raise HTTPException(502, f"Connexion à Garmin Connect impossible : {exc}") from exc
    result = send_session(client, session, DATA_DIR)
    messages = {"envoyee": "Séance envoyée dans ton calendrier Garmin.",
                "remplacee": "La séance du jour a changé : la précédente a été remplacée.",
                "deja_envoyee": "Cette séance est déjà dans ton calendrier Garmin."}
    return {**result, "message": messages[result["statut"]]}


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
    distance_km: float | None = Query(None, ge=1, le=100, description="Distance libre en km (remplace `distance`)"),
) -> dict:
    """Nutrition et hydratation avant, pendant et après la course, selon ta durée prévue et la température."""
    _, _, estimate = current_estimate()
    _, meters, label = distance_from(distance, distance_km)
    duration_s = race_time_s(estimate["vdot"], meters, denivele_m)
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
