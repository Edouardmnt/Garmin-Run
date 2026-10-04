"""Objectifs de course : enregistrés dans $RUNLAB_DATA_DIR/objectifs.json (jamais publié).

Un seul objectif est actif à la fois : c'est lui qui pilote le planning et l'envoi des séances à la montre.
"""

import json
import uuid
from datetime import date, datetime
from pathlib import Path

DISTANCES = {"5k": 5.0, "10k": 10.0, "semi": 21.0975, "marathon": 42.195}
MIN_KM, MAX_KM = 1.0, 100.0


def goal_km(goal: dict) -> float:
    """Distance de l'objectif en km (distance libre, ou distance classique pour les anciens objectifs)."""
    return float(goal.get("distance_km") or DISTANCES[goal["distance"]])


def goal_label(goal: dict) -> str:
    names = {"5k": "5 km", "10k": "10 km", "semi": "semi-marathon", "marathon": "marathon"}
    return names.get(goal.get("distance")) or f"{goal_km(goal):g} km".replace(".", ",")


def goals_file(data_dir: Path) -> Path:
    return data_dir / "objectifs.json"


def load_goals(data_dir: Path) -> list[dict]:
    path = goals_file(data_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_goals(data_dir: Path, goals: list[dict]) -> None:
    path = goals_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(goals, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_duration(text: str | None) -> int | None:
    """"47:30" -> 2850 s ; "1:45:00" -> 6300 s ; vide -> None."""
    if not text:
        return None
    parts = [int(p) for p in str(text).replace("h", ":").replace("'", ":").strip(":").split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError("Temps visé attendu au format mm:ss ou h:mm:ss, par exemple 47:30 ou 1:45:00")


def validate_goal(goal: dict, today: date) -> dict:
    errors = {}
    if not str(goal.get("nom", "")).strip():
        errors["nom"] = "Donne un nom à ta course."
    if goal.get("distance") not in DISTANCES:
        try:
            km = float(goal.get("distance_km") or 0)
        except (TypeError, ValueError):
            km = 0
        if not MIN_KM <= km <= MAX_KM:
            errors["distance"] = "Choisis 5 km, 10 km, semi ou marathon, ou indique une distance entre 1 et 100 km."
    try:
        race_day = date.fromisoformat(str(goal.get("date_course")))
        if race_day <= today:
            errors["date_course"] = "La date de la course doit être dans le futur."
    except ValueError:
        errors["date_course"] = "Date attendue au format AAAA-MM-JJ."
    try:
        parse_duration(goal.get("temps_vise"))
    except ValueError as exc:
        errors["temps_vise"] = str(exc)
    if not 2 <= int(goal.get("seances_par_semaine", 3)) <= 6:
        errors["seances_par_semaine"] = "Entre 2 et 6 sorties par semaine."
    if any(d not in range(7) for d in goal.get("jours_tennis", [])):
        errors["jours_tennis"] = "Jours de 0 (lundi) à 6 (dimanche)."
    return errors


def create_goal(data_dir: Path, goal: dict) -> dict:
    """Ajoute un objectif et le rend actif."""
    goals = load_goals(data_dir)
    for g in goals:
        g["actif"] = False
    record = {
        "id": uuid.uuid4().hex[:8],
        "nom": str(goal["nom"]).strip(),
        "distance": goal["distance"] if goal.get("distance") in DISTANCES else None,
        "distance_km": DISTANCES.get(goal.get("distance")) or round(float(goal["distance_km"]), 3),
        "date_course": str(goal["date_course"]),
        "temps_vise_s": parse_duration(goal.get("temps_vise")),
        "denivele_m": int(goal.get("denivele_m", 0)),
        "seances_par_semaine": int(goal.get("seances_par_semaine", 3)),
        "jours_tennis": sorted(int(d) for d in goal.get("jours_tennis", [])),
        "jour_sortie_longue": int(goal.get("jour_sortie_longue", 6)),
        "actif": True,
        "cree_le": datetime.now().date().isoformat(),
    }
    save_goals(data_dir, goals + [record])
    return record


def set_active(data_dir: Path, goal_id: str) -> dict | None:
    goals = load_goals(data_dir)
    found = None
    for g in goals:
        g["actif"] = g["id"] == goal_id
        found = g if g["actif"] else found
    if found:
        save_goals(data_dir, goals)
    return found


def delete_goal(data_dir: Path, goal_id: str) -> bool:
    goals = load_goals(data_dir)
    remaining = [g for g in goals if g["id"] != goal_id]
    if len(remaining) == len(goals):
        return False
    save_goals(data_dir, remaining)
    return True


def active_goal(data_dir: Path, today: date) -> dict | None:
    """L'objectif actif, s'il n'est pas déjà passé."""
    for g in load_goals(data_dir):
        if g["actif"] and date.fromisoformat(g["date_course"]) >= today:
            return g
    return None
