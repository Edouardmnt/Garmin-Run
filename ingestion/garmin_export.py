"""Extraction des données Garmin Connect vers $RUNLAB_DATA_DIR/raw (couche bronze).

- Activités (course, tennis, etc.) -> raw/activities.json, fusionnées avec l'historique existant
- Données quotidiennes (sommeil, VFC, stats du jour) -> raw/daily/AAAA-MM-JJ.json

Deux usages :
- chargement initial : GARMIN_DAYS=200 (valeur par défaut) ;
- synchronisation quotidienne (CronJob) : GARMIN_DAYS=3.

Configuration par variables d'environnement (aucun identifiant dans le code) :
- RUNLAB_DATA_DIR : dossier de données (data/ par défaut)
- GARMIN_DAYS     : nombre de jours à récupérer
- GARMINTOKENS    : dossier des jetons Garmin (~/.garminconnect par défaut)
- GARMIN_EMAIL, GARMIN_PASSWORD : identifiants, seulement si aucun jeton valide n'existe
"""

import json
import os
import time
from collections import Counter
from datetime import date, timedelta
from getpass import getpass
from pathlib import Path

from garminconnect import Garmin

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
TOKENSTORE = os.path.expanduser(os.getenv("GARMINTOKENS", "~/.garminconnect"))
DAYS = int(os.getenv("GARMIN_DAYS", "200"))
REFRESH_DAYS = 2  # les derniers jours sont retéléchargés : la journée en cours était incomplète
PAUSE = 1.0  # secondes entre deux appels, pour ne pas surcharger Garmin


def connect() -> Garmin:
    """Réutilise les jetons enregistrés, sinon se connecte avec les identifiants."""
    try:
        client = Garmin()
        client.login(TOKENSTORE)
        print("Connecté avec les jetons enregistrés.")
        return client
    except Exception:
        print("Pas de jetons valides : connexion avec identifiants.")
        email = os.getenv("GARMIN_EMAIL") or input("E-mail Garmin : ")
        password = os.getenv("GARMIN_PASSWORD") or getpass("Mot de passe Garmin (rien ne s'affiche, c'est normal) : ")
        client = Garmin(email, password, prompt_mfa=lambda: input("Code MFA : "))
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


def merge_activities(new: list[dict]) -> list[dict]:
    """Fusionne les nouvelles activités avec l'historique, sans doublon (clé : activityId)."""
    path = RAW_DIR / "activities.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    by_id = {a["activityId"]: a for a in existing}
    by_id.update({a["activityId"]: a for a in new})  # une activité modifiée remplace l'ancienne
    return sorted(by_id.values(), key=lambda a: a.get("startTimeLocal", ""))


def main() -> None:
    client = connect()
    end = date.today()
    start = end - timedelta(days=DAYS)

    # 1. Activités, fusionnées avec l'historique
    print(f"Activités du {start} au {end}...")
    new = client.get_activities_by_date(start.isoformat(), end.isoformat())
    activities = merge_activities(new)
    save(RAW_DIR / "activities.json", activities)
    types = Counter(a.get("activityType", {}).get("typeKey", "?") for a in activities)
    print(f"{len(new)} activités récupérées, {len(activities)} au total : {dict(types)}")

    # 2. Données quotidiennes
    daily_dir = RAW_DIR / "daily"
    refresh_from = end - timedelta(days=REFRESH_DAYS)
    for i in range(DAYS + 1):
        day = start + timedelta(days=i)
        out = daily_dir / f"{day.isoformat()}.json"
        if out.exists() and day < refresh_from:
            continue
        print(f"Jour {day}")
        save(out, {
            "date": day.isoformat(),
            "sleep": safe_call(client.get_sleep_data, day.isoformat()),
            "hrv": safe_call(client.get_hrv_data, day.isoformat()),
            "stats": safe_call(client.get_stats, day.isoformat()),
        })
        time.sleep(PAUSE)

    print(f"Terminé. Données dans {RAW_DIR}")


if __name__ == "__main__":
    main()
