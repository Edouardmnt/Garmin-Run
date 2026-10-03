"""Prépare le fichier d'étiquetage de mes sorties de course à pied.

Sortie : $RUNLAB_DATA_DIR/labels/run_labels.csv (dans data/, donc jamais publié)

Chaque sortie reçoit une SUGGESTION calculée par des règles expertes. La colonne `label`, elle,
est à remplir à la main : ef, tempo, fractionne ou course. C'est la vérité terrain du projet.
Relancer le script ajoute les nouvelles sorties sans jamais effacer les étiquettes déjà saisies.
"""

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
OUT = DATA_DIR / "labels" / "run_labels.csv"
RUN_TYPES = {"running", "treadmill_running", "trail_running", "track_running"}
RACE_DISTANCES_KM = [5.0, 10.0, 21.0975, 42.195]
LABELS = ["ef", "tempo", "fractionne", "course"]


def interval_reps(activity: dict) -> int:
    """Nombre de portions d'effort détectées par Garmin dans le résumé des splits."""
    reps = 0
    for split in activity.get("splitSummaries") or []:
        split_type = str(split.get("splitType", "")).upper()
        if "INTERVAL_ACTIVE" in split_type:
            reps += int(split.get("noOfSplits") or 1)
    return reps


def summarize(activity: dict) -> dict:
    zones = [activity.get(f"hrTimeInZone_{i}") or 0 for i in range(1, 6)]
    total = sum(zones) or None
    share = [z / total if total else None for z in zones]
    distance_km = (activity.get("distance") or 0) / 1000
    duration_min = (activity.get("duration") or 0) / 60
    return {
        "activity_id": activity["activityId"],
        "date": str(activity.get("startTimeLocal", ""))[:16],
        "type": activity.get("activityType", {}).get("typeKey"),
        "distance_km": round(distance_km, 2),
        "duration_min": round(duration_min, 1),
        "pace_min_km": round(duration_min / distance_km, 2) if distance_km > 0.5 else None,
        "avg_hr": activity.get("averageHR"),
        "max_hr": activity.get("maxHR"),
        "z1_z2": round(share[0] + share[1], 2) if total else None,
        "z3": round(share[2], 2) if total else None,
        "z4_z5": round(share[3] + share[4], 2) if total else None,
        "has_intervals": bool(activity.get("hasIntensityIntervals")),
        "interval_reps": interval_reps(activity),
        "lap_count": activity.get("lapCount"),
    }


def suggest(row: pd.Series) -> str:
    """Règles expertes (zones Garmin par défaut, en % de FC max). Une suggestion, pas une vérité."""
    if row["z4_z5"] is None or pd.isna(row["z4_z5"]):
        return "?"
    near_race = any(abs(row["distance_km"] - d) / d <= 0.03 for d in RACE_DISTANCES_KM)
    if near_race and row["z4_z5"] >= 0.6:
        return "course"
    if row["has_intervals"] or row["interval_reps"] >= 3:
        return "fractionne"
    if row["z4_z5"] >= 0.35:
        return "tempo"
    return "ef"


def main() -> None:
    activities = json.loads((DATA_DIR / "raw" / "activities.json").read_text(encoding="utf-8"))
    runs = pd.DataFrame([summarize(a) for a in activities if a.get("activityType", {}).get("typeKey") in RUN_TYPES])
    runs["suggestion"] = runs.apply(suggest, axis=1)
    runs["label"] = ""

    if OUT.exists():  # conserver les étiquettes déjà saisies
        existing = pd.read_csv(OUT, sep=";", decimal=",", dtype={"label": str}).set_index("activity_id")["label"].dropna()
        runs["label"] = runs["activity_id"].map(existing).fillna("")

    runs = runs.sort_values("date").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Format adapté à Excel en français : séparateur ";", virgule décimale, accents corrects (utf-8-sig)
    runs.to_csv(OUT, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    print(runs[["date", "distance_km", "pace_min_km", "avg_hr", "z4_z5", "has_intervals", "suggestion", "label"]].to_string())
    print(f"\nSuggestions : {runs['suggestion'].value_counts().to_dict()}")
    todo = (runs["label"] == "").sum()
    print(f"{todo} sorties à étiqueter dans {OUT}")
    print(f"Valeurs possibles pour la colonne label : {', '.join(LABELS)}")


if __name__ == "__main__":
    main()
