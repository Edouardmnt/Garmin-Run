"""Questionnaire après chaque sortie de course : 3 à 5 questions dont les réponses améliorent l'application.

Ce que les réponses changent :
- type de séance       -> devient l'étiquette de la sortie (classification, allures, chronos de référence) ;
- effort ressenti      -> repère les footings courus trop vite ;
- douleur              -> verdict du jour plafonné et planning allégé ;
- temps / allure prévus -> mesure la justesse des prédictions et des allures conseillées ;
- forme avant de partir -> vérifie que le verdict du jour correspond au ressenti.

Stockage : $RUNLAB_DATA_DIR/feedback/feedback.jsonl (une réponse par ligne, jamais publié).
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

Q_EFFORT = {
    "id": "effort", "question": "Comment as-tu ressenti l'effort ?",
    "options": ["Très facile (1-3/10)", "Modéré (4-6/10)", "Difficile (7-8/10)", "Maximal (9-10/10)"],
}
Q_TYPE = {
    "id": "type", "question": "Quel type de séance était-ce ?",
    "options": ["Endurance fondamentale", "Tempo", "Fractionné", "Course"],
}
Q_PAIN = {
    "id": "douleur", "question": "As-tu ressenti une douleur ?",
    "options": ["Aucune", "Légère gêne", "Douleur qui m'a fait ralentir ou arrêter"],
}
Q_PREDICTION = {
    "id": "prediction", "question": "Par rapport à ton chrono, le temps prédit par l'application était…",
    "options": ["Trop optimiste", "Juste", "Trop pessimiste", "Je n'avais pas regardé"],
}
Q_PACE = {
    "id": "allure", "question": "L'allure conseillée pour cette séance était…",
    "options": ["Trop lente", "Adaptée", "Trop rapide", "Je ne l'ai pas suivie"],
}
Q_READINESS = {
    "id": "forme", "question": "Comment te sentais-tu avant de partir ?",
    "options": ["En pleine forme", "Normal", "Fatigué"],
}
RPE = {0: 2, 1: 5, 2: 7.5, 3: 9.5}
TYPE_TO_LABEL = {"Endurance fondamentale": "ef", "Tempo": "tempo", "Fractionné": "fractionne", "Course": "course"}
PENDING_WINDOW_DAYS = 7


def feedback_file(data_dir: Path) -> Path:
    return data_dir / "feedback" / "feedback.jsonl"


def load_feedback(data_dir: Path) -> list[dict]:
    path = feedback_file(data_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def questions_for(kind: str | None) -> list[dict]:
    """4 questions pour un footing, 5 pour une séance de qualité ou une course."""
    if kind == "course":
        return [Q_EFFORT, Q_TYPE, Q_PAIN, Q_PREDICTION, Q_READINESS]
    if kind in ("tempo", "fractionne"):
        return [Q_EFFORT, Q_TYPE, Q_PAIN, Q_PACE, Q_READINESS]
    return [Q_EFFORT, Q_TYPE, Q_PAIN, Q_READINESS]


def pending_run(runs: pd.DataFrame, feedback: list[dict], today: date) -> pd.Series | None:
    """Dernière sortie de course des 7 derniers jours sans réponse au questionnaire."""
    answered = {f["activity_id"] for f in feedback}
    recent = runs[(runs["date"] > pd.Timestamp(today - timedelta(days=PENDING_WINDOW_DAYS)))
                  & ~runs["activity_id"].isin(answered)]
    return None if recent.empty else recent.sort_values("start_time").iloc[-1]


def validate(answers: dict, questions: list[dict]) -> dict:
    """Vérifie que chaque question a une réponse parmi les options proposées."""
    errors = {}
    for q in questions:
        if answers.get(q["id"]) not in q["options"]:
            errors[q["id"]] = f"Réponse attendue parmi : {', '.join(q['options'])}"
    return errors


def save_feedback(data_dir: Path, activity_id: int, run_date: str, answers: dict) -> dict:
    effort_index = Q_EFFORT["options"].index(answers["effort"])
    record = {
        "activity_id": int(activity_id),
        "date_sortie": run_date,
        "repondu_le": datetime.now().isoformat(timespec="seconds"),
        "reponses": answers,
        "rpe": RPE[effort_index],
        "label": TYPE_TO_LABEL[answers["type"]],
        "douleur": Q_PAIN["options"].index(answers["douleur"]),  # 0 aucune, 1 gêne, 2 douleur
    }
    path = feedback_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def apply_feedback_labels(labels: pd.DataFrame | None, feedback: list[dict]) -> pd.DataFrame | None:
    """Le type déclaré dans le questionnaire sert d'étiquette quand aucune étiquette n'a été saisie à la main."""
    if not feedback:
        return labels
    from_feedback = {f["activity_id"]: f["label"] for f in feedback}
    if labels is None:
        return pd.DataFrame({"activity_id": list(from_feedback), "label": list(from_feedback.values()), "suggestion": "?"})
    labels = labels.copy()
    empty = labels["label"].isna() | (labels["label"].astype(str).str.strip() == "")
    labels.loc[empty, "label"] = labels.loc[empty, "activity_id"].map(from_feedback)
    missing = [a for a in from_feedback if a not in set(labels["activity_id"])]
    extra = pd.DataFrame({"activity_id": missing, "label": [from_feedback[a] for a in missing], "suggestion": "?"})
    return pd.concat([labels, extra], ignore_index=True)


def efforts_by_activity(feedback: list[dict]) -> dict[int, float]:
    """Effort ressenti (RPE) déclaré pour chaque sortie."""
    return {f["activity_id"]: f["rpe"] for f in feedback}


PREDICTION_SCORE = {"Trop optimiste": 1, "Juste": 0, "Trop pessimiste": -1}
BIAS_STEP_PCT, BIAS_MAX_PCT, BIAS_WINDOW = 1.5, 3.0, 5


def prediction_bias(feedback: list[dict]) -> tuple[float, int]:
    """Correction des temps prédits d'après les réponses « temps prédit » des dernières courses.

    « Trop optimiste » : la prédiction était trop rapide -> on allonge les temps ; « Trop pessimiste » :
    on les raccourcit. Moyenne des 5 dernières réponses, 1,5 % par réponse, plafonnée à ±3 %.
    """
    scores = [PREDICTION_SCORE[f["reponses"]["prediction"]] for f in sorted(feedback, key=lambda f: f["date_sortie"])
              if f["reponses"].get("prediction") in PREDICTION_SCORE][-BIAS_WINDOW:]
    if not scores:
        return 0.0, 0
    pct = sum(scores) / len(scores) * BIAS_STEP_PCT * len(scores) ** 0.5  # plus de réponses = plus de confiance
    return round(max(-BIAS_MAX_PCT, min(BIAS_MAX_PCT, pct)), 2), len(scores)


def recent_signals(feedback: list[dict], today: date) -> dict:
    """Signaux récents utilisés par le verdict du jour et le planning."""
    def within(days):
        limit = (today - timedelta(days=days)).isoformat()
        return [f for f in feedback if f["date_sortie"] >= limit]

    last_week, two_weeks = within(7), within(14)
    hard_easy_runs = [f for f in two_weeks if f["label"] == "ef" and f["rpe"] >= 7]
    return {
        "douleur_recente": any(f["douleur"] >= 1 for f in last_week),
        "douleur_forte_recente": any(f["douleur"] == 2 for f in last_week),
        "footings_trop_durs": len(hard_easy_runs) >= 2,
    }


def summary(feedback: list[dict]) -> dict:
    """Ce que les questionnaires disent de la justesse de l'application."""
    def count(question_id):
        values = [f["reponses"].get(question_id) for f in feedback if question_id in f["reponses"]]
        return {v: values.count(v) for v in dict.fromkeys(values)} if values else {}

    return {"questionnaires": len(feedback), "predictions": count("prediction"), "allures": count("allure"),
            "forme_avant_depart": count("forme")}
