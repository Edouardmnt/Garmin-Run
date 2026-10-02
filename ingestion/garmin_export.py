"""Extraction des données Garmin Connect vers data/raw (couche bronze).

- Activités (course, tennis, etc.) sur les N derniers jours -> data/raw/activities.json
- Données quotidiennes (sommeil, VFC, stats du jour) -> data/raw/daily/AAAA-MM-JJ.json

Les jours déjà téléchargés sont ignorés : le script peut être relancé sans risque.
Aucun identifiant n'est stocké dans le dépôt : les jetons Garmin restent dans ~/.garminconnect.
"""

import json
import os
import time
from collections import Counter
from datetime import date, timedelta
from getpass import getpass
from pathlib import Path

from garminconnect import Garmin

TOKENSTORE = os.path.expanduser("~/.garminconnect")
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DAYS = int(os.getenv("GARMIN_DAYS", "200"))  # ~6 mois + marge
PAUSE = 1.0  # secondes entre deux appels, pour ne pas surcharger Garmin


def connect() -> Garmin:
    """Réutilise les jetons enregistrés, sinon demande les identifiants."""
    try:
        client = Garmin()
        client.login(TOKENSTORE)
        print("Connecté avec les jetons enregistrés.")
        return client
    except Exception:
        print("Pas de jetons valides : connexion avec identifiants.")
        email = os.getenv("GARMIN_EMAIL") or input("E-mail Garmin : ")
        client = Garmin(
            email,
            getpass("Mot de passe Garmin (rien ne s'affiche, c'est normal) : "),
            prompt_mfa=lambda: input("Code MFA : "),
        )
        client.login(TOKENSTORE)
        return client


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_call(func, *args):
    """Un appel qui échoue ne doit pas arrêter tout l'export."""
    try:
        return func(*args)
    except Exception as exc:
        print(f"  ! {func.__name__}{args} : {exc}")
        return None


def main() -> None:
    client = connect()
    end = date.today()
    start = end - timedelta(days=DAYS)

    # 1. Activités
    print(f"Activités du {start} au {end}...")
    activities = client.get_activities_by_date(start.isoformat(), end.isoformat())
    save(RAW_DIR / "activities.json", activities)
    types = Counter(a.get("activityType", {}).get("typeKey", "?") for a in activities)
    print(f"{len(activities)} activités : {dict(types)}")

    # 2. Données quotidiennes
    daily_dir = RAW_DIR / "daily"
    for i in range(DAYS + 1):
        day = (start + timedelta(days=i)).isoformat()
        out = daily_dir / f"{day}.json"
        if out.exists():
            continue
        print(f"Jour {day}")
        save(out, {
            "date": day,
            "sleep": safe_call(client.get_sleep_data, day),
            "hrv": safe_call(client.get_hrv_data, day),
            "stats": safe_call(client.get_stats, day),
        })
        time.sleep(PAUSE)

    print(f"Terminé. Données dans {RAW_DIR}")


if __name__ == "__main__":
    main()
