"""Inspection des données disponibles pour classer les sorties de course à pied
(endurance fondamentale, tempo, fractionné, course).

1. Hors ligne : quels champs utiles existent déjà dans raw/activities.json ?
2. Avec --online : structure du détail (tours, zones cardiaques) d'UNE sortie, via Garmin Connect.

N'affiche que des noms de champs et des statistiques, jamais de trace GPS.
Usage : python scripts/inspect_run_details.py [--online]
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # pour réutiliser la connexion du script d'ingestion
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
RUN_TYPES = {"running", "treadmill_running", "trail_running", "track_running"}


def describe(obj, depth: int = 0, max_depth: int = 2) -> None:
    """Affiche la structure d'un objet JSON (clés et types), sans les valeurs."""
    indent = "  " * depth
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:40]:
            kind = type(value).__name__
            size = f" ({len(value)} éléments)" if isinstance(value, list | dict) else ""
            print(f"{indent}- {key}: {kind}{size}")
            if depth < max_depth and isinstance(value, dict | list):
                describe(value, depth + 1, max_depth)
    elif isinstance(obj, list) and obj:
        print(f"{indent}  [premier élément]")
        describe(obj[0], depth + 1, max_depth)


def offline_check() -> list[dict]:
    activities = json.loads((DATA_DIR / "raw" / "activities.json").read_text(encoding="utf-8"))
    runs = [a for a in activities if a.get("activityType", {}).get("typeKey") in RUN_TYPES]
    print(f"{len(runs)} sorties de course à pied sur {len(activities)} activités\n")

    keys = Counter(k for r in runs for k in r)
    interesting = [k for k in keys if any(w in k.lower() for w in ("zone", "event", "lap", "split", "race", "interval", "pace"))]
    print("Champs potentiellement utiles dans le résumé (présents dans X sorties) :")
    for k in sorted(interesting):
        print(f"  {k:<40} {keys[k]}")

    events = Counter(str((r.get("eventType") or {}).get("typeKey")) for r in runs)
    print(f"\nType d'événement Garmin (eventType) : {dict(events)}")
    distances = sorted(round((r.get("distance") or 0) / 1000, 1) for r in runs)
    print(f"Distances (km) : {distances}")
    return runs


def online_check(runs: list[dict]) -> None:
    from ingestion.garmin_export import connect

    run = max(runs, key=lambda r: r.get("startTimeLocal", ""))
    activity_id = run["activityId"]
    print(f"\nDétail de la sortie la plus récente ({run.get('startTimeLocal', '')[:10]}, {run.get('distance', 0) / 1000:.1f} km)")
    client = connect()
    for name in ("get_activity_splits", "get_activity_hr_in_timezones"):
        func = getattr(client, name, None)
        if func is None:
            print(f"\n{name} : méthode absente de cette version de garminconnect")
            continue
        try:
            data = func(activity_id)
        except Exception as exc:
            print(f"\n{name} : erreur {exc}")
            continue
        print(f"\n{name} : structure")
        describe(data)


if __name__ == "__main__":
    runs = offline_check()
    if "--online" in sys.argv and runs:
        online_check(runs)
