"""Tests du questionnaire après sortie, de la nutrition et des nouvelles analyses."""

from datetime import date

import pandas as pd
import pytest

from processing.feedback import (
    apply_feedback_labels,
    load_feedback,
    pending_run,
    questions_for,
    recent_signals,
    save_feedback,
    validate,
)
from processing.insights import build_analysis
from processing.nutrition import race_nutrition

TODAY = date(2026, 10, 1)


def answers(kind, **overrides):
    ans = {q["id"]: q["options"][0] for q in questions_for(kind)}
    return {**ans, **overrides}


def test_trois_a_cinq_questions_selon_la_seance():
    assert len(questions_for("ef")) == 4
    assert len(questions_for("fractionne")) == 5
    ids = [q["id"] for q in questions_for("course")]
    assert "prediction" in ids and "allure" not in ids  # après une course, on évalue le temps prédit
    assert all(3 <= len(questions_for(k)) <= 5 for k in (None, "ef", "tempo", "fractionne", "course"))


def test_validation_des_reponses():
    assert validate(answers("ef"), questions_for("ef")) == {}
    assert "effort" in validate(answers("ef", effort="Inventé"), questions_for("ef"))


def test_enregistrement_et_effets(tmp_path):
    save_feedback(tmp_path, 42, "2026-09-29", answers("ef", type="Fractionné", effort="Difficile (7-8/10)",
                                                      douleur="Légère gêne"))
    feedback = load_feedback(tmp_path)
    assert feedback[0]["label"] == "fractionne" and feedback[0]["rpe"] == 7.5 and feedback[0]["douleur"] == 1
    # Le type déclaré devient l'étiquette, sans écraser une étiquette saisie à la main
    labels = pd.DataFrame({"activity_id": [42, 43], "label": [None, "tempo"], "suggestion": ["ef", "ef"]})
    merged = apply_feedback_labels(labels, feedback).set_index("activity_id")["label"]
    assert merged[42] == "fractionne" and merged[43] == "tempo"
    assert recent_signals(feedback, TODAY)["douleur_recente"] is True
    assert recent_signals(feedback, date(2026, 10, 20))["douleur_recente"] is False  # plus d'une semaine après


def test_sortie_en_attente():
    runs = pd.DataFrame({"activity_id": [1, 2, 3], "start_time": ["2026-09-20", "2026-09-28", "2026-09-30"],
                         "date": pd.to_datetime(["2026-09-20", "2026-09-28", "2026-09-30"])})
    assert pending_run(runs, [], TODAY)["activity_id"] == 3
    assert pending_run(runs, [{"activity_id": 3}], TODAY)["activity_id"] == 2
    assert pending_run(runs, [{"activity_id": 3}, {"activity_id": 2}], TODAY) is None  # la 1re a plus de 7 jours


@pytest.mark.parametrize("minutes,carbs,sodium", [(45, [0, 0], False), (100, [30, 60], False), (200, [60, 90], True)])
def test_nutrition_selon_la_duree(minutes, carbs, sodium):
    plan = race_nutrition("course", minutes, 300, 15)
    assert plan["glucides_g_par_heure"] == carbs and plan["sodium"] is sodium
    assert plan["avant"] and plan["apres"] and "entraînement" in plan["avertissement"]


def test_nutrition_chaleur_et_reperes():
    hot, mild = race_nutrition("semi", 100, 285, 28), race_nutrition("semi", 100, 285, 12)
    assert hot["boisson_ml_par_heure"][1] > mild["boisson_ml_par_heure"][1]
    assert hot["sodium"]  # effort de plus d'une heure par forte chaleur
    gels = [r for r in mild["reperes"] if "gel" in r["action"]]
    assert gels and gels[0]["minute"] == 35 and all(r["minute"] < 100 for r in mild["reperes"])


def test_analyses_nuit_journee_stress_activites(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    acts = pd.read_parquet(data_dir / "silver" / "activities.parquet")
    analysis = build_analysis(gold, [], {"stress_veille": 60, "body_battery_max": 40, "douleur_recente": True}, acts)
    assert analysis["nuit"]["nuit"]["profond_pct"] > 0 and analysis["nuit"]["nuit"]["coucher"]
    assert analysis["stress"]["points"] and analysis["journee"]["points"]
    assert "activites" in analysis
    verdict = analysis["verdict"]
    assert verdict["score"] <= 60 and "douleur" in verdict["explication"]  # douleur signalée : pas de feu vert
