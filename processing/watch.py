"""Envoi de la séance du jour sur la montre, via le calendrier Garmin Connect.

1. La séance du planning (étapes structurées) est convertie en entraînement Garmin : échauffement,
   répétitions, récupérations, retour au calme, avec une cible d'allure à chaque étape.
2. L'entraînement est créé dans le compte (upload_workout) puis placé à la date du jour (schedule_workout).
   La montre l'affiche comme entraînement du jour à sa prochaine synchronisation.
3. Les envois sont mémorisés : une séance inchangée n'est jamais renvoyée ; une séance modifiée
   (allégée à cause de la forme du jour, par exemple) remplace la précédente.

Format reproduit d'après garminconnect.workout (types d'étapes, conditions de fin, cibles "pace.zone").
"""

import hashlib
import json
from pathlib import Path

STEP_TYPES = {"echauffement": (1, "warmup"), "retour_au_calme": (2, "cooldown"), "effort": (3, "interval"),
              "recuperation": (4, "recovery"), "repetition": (6, "repeat")}
RUNNING = {"sportTypeId": 1, "sportTypeKey": "running", "displayOrder": 1}
NO_TARGET = {"workoutTargetTypeId": 1, "workoutTargetTypeKey": "no.target", "displayOrder": 1}
PACE_TARGET = {"workoutTargetTypeId": 6, "workoutTargetTypeKey": "pace.zone", "displayOrder": 6}
NAME_PREFIX = "Foulée"


def _executable(step: dict, order: int) -> dict:
    type_id, type_key = STEP_TYPES[step["type"]]
    if step.get("distance_m"):
        condition = {"conditionTypeId": 3, "conditionTypeKey": "distance", "displayOrder": 3, "displayable": True}
        value = float(step["distance_m"])
    elif step.get("duree_s"):
        condition = {"conditionTypeId": 2, "conditionTypeKey": "time", "displayOrder": 2, "displayable": True}
        value = float(step["duree_s"])
    else:  # sans durée ni distance : on passe à l'étape suivante avec le bouton Lap
        condition = {"conditionTypeId": 1, "conditionTypeKey": "lap.button", "displayOrder": 1, "displayable": True}
        value = None
    dto = {"type": "ExecutableStepDTO", "stepOrder": order,
           "stepType": {"stepTypeId": type_id, "stepTypeKey": type_key, "displayOrder": type_id},
           "endCondition": condition, "endConditionValue": value, "targetType": NO_TARGET}
    pace = step.get("allure")
    if pace:  # cible d'allure en m/s : la limite basse est l'allure la plus LENTE
        dto.update({"targetType": PACE_TARGET, "targetValueOne": round(1000 / pace["lente_s"], 4),
                    "targetValueTwo": round(1000 / pace["rapide_s"], 4)})
    return {k: v for k, v in dto.items() if v is not None}


def _steps(steps: list[dict], counter: list[int]) -> list[dict]:
    out = []
    for step in steps:
        counter[0] += 1
        order = counter[0]
        if step["type"] == "repetition":
            out.append({
                "type": "RepeatGroupDTO", "stepOrder": order,
                "stepType": {"stepTypeId": 6, "stepTypeKey": "repeat", "displayOrder": 6},
                "numberOfIterations": step["repetitions"],
                "endCondition": {"conditionTypeId": 7, "conditionTypeKey": "iterations", "displayOrder": 7,
                                 "displayable": False},
                "endConditionValue": float(step["repetitions"]), "smartRepeat": False,
                "workoutSteps": _steps(step["etapes"], counter),
            })
        else:
            out.append(_executable(step, order))
    return out


def estimated_duration_s(session: dict) -> int:
    return int((session.get("duree_min") or 30) * 60)


def to_garmin_workout(session: dict) -> dict:
    """Convertit une séance du planning en entraînement Garmin (dictionnaire prêt pour upload_workout)."""
    day = session["date"][8:10] + "/" + session["date"][5:7]
    return {
        "workoutName": f"{NAME_PREFIX} {day} {session['titre']}"[:80],
        "description": f"{session['description']} {session['objectif']}"[:500],
        "sportType": RUNNING,
        "estimatedDurationInSecs": estimated_duration_s(session),
        "workoutSegments": [{"segmentOrder": 1, "sportType": RUNNING,
                             "workoutSteps": _steps(session["etapes"], [0])}],
    }


def fingerprint(session: dict) -> str:
    """Empreinte du contenu d'une séance : elle change si la séance est modifiée."""
    content = json.dumps({k: session.get(k) for k in ("titre", "description", "etapes")}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def history_file(data_dir: Path) -> Path:
    return data_dir / "montre" / "envois.json"


def load_history(data_dir: Path) -> dict:
    path = history_file(data_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def send_session(client, session: dict, data_dir: Path) -> dict:
    """Envoie une séance au calendrier Garmin, sans doublon. `client` : objet garminconnect.Garmin connecté."""
    history = load_history(data_dir)
    previous = history.get(session["date"])
    print_ = fingerprint(session)
    if previous and previous["empreinte"] == print_:
        return {"statut": "deja_envoyee", "workout_id": previous["workout_id"], "date": session["date"]}
    if previous:  # la séance a changé : on remplace l'ancienne
        try:
            client.delete_workout(previous["workout_id"])
        except Exception:
            pass  # déjà supprimée à la main dans Garmin Connect
    created = client.upload_workout(to_garmin_workout(session))
    workout_id = created["workoutId"]
    client.schedule_workout(workout_id, session["date"])
    history[session["date"]] = {"workout_id": workout_id, "empreinte": print_, "titre": session["titre"]}
    path = history_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"statut": "remplacee" if previous else "envoyee", "workout_id": workout_id, "date": session["date"],
            "titre": session["titre"]}


def remove_session(client, day: str, data_dir: Path) -> dict:
    """Retire du calendrier Garmin la séance envoyée pour ce jour (devenu un jour de repos ou séance déplacée)."""
    history = load_history(data_dir)
    previous = history.pop(day, None)
    if previous is None:
        return {"statut": "rien_a_retirer", "date": day}
    try:
        client.delete_workout(previous["workout_id"])
    except Exception:
        pass  # déjà supprimée à la main dans Garmin Connect
    path = history_file(data_dir)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"statut": "retiree", "date": day}
