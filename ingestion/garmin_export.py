"""Extraction des données Garmin Connect vers $RUNLAB_DATA_DIR/raw (couche bronze).

- Activités (course, tennis, etc.) -> raw/activities.json, fusionnées avec l'historique existant
- Données quotidiennes (sommeil, VFC, stats du jour) -> raw/daily/AAAA-MM-JJ.json

Chargement incrémental avec "watermark" : le script repart du dernier jour déjà téléchargé.
Que la dernière synchronisation date d'hier ou de trois semaines, il récupère tous les jours
manquants, sans trou. Si le dossier est vide, il fait le chargement initial (GARMIN_DAYS jours).

Configuration par variables d'environnement (aucun identifiant dans le code) :
- RUNLAB_DATA_DIR : dossier de données (data/ par défaut)
- GARMIN_DAYS     : profondeur du chargement initial, quand aucune donnée n'existe (200 par défaut)
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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
TOKENSTORE = os.path.expanduser(os.getenv("GARMINTOKENS", "~/.garminconnect"))
INITIAL_DAYS = int(os.getenv("GARMIN_DAYS", "200"))
MAX_DAYS = 365  # garde-fou : jamais plus d'un an d'un coup
REFRESH_DAYS = 2  # jours retéléchargés avant le dernier jour connu : ils étaient peut-être incomplets
PAUSE = 1.0  # secondes entre deux appels, pour ne pas surcharger Garmin


def compute_start(end: date, raw_dir: Path) -> date:
    """Premier jour à télécharger : dernier jour connu - REFRESH_DAYS, ou chargement initial."""
    known_days = sorted(p.stem for p in (raw_dir / "daily").glob("*.json"))
    if known_days:
        start = date.fromisoformat(known_days[-1]) - timedelta(days=REFRESH_DAYS)
    else:
        start = end - timedelta(days=INITIAL_DAYS)
    return max(start, end - timedelta(days=MAX_DAYS))


def connect():
    """Réutilise les jetons enregistrés, sinon se connecte avec les identifiants."""
    from garminconnect import Garmin  # import ici : le reste du module reste testable sans la bibliothèque

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


def merge_activities(new: list[dict], raw_dir: Path) -> list[dict]:
    """Fusionne les nouvelles activités avec l'historique, sans doublon (clé : activityId)."""
    path = raw_dir / "activities.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    by_id = {a["activityId"]: a for a in existing}
    by_id.update({a["activityId"]: a for a in new})  # une activité modifiée remplace l'ancienne
    return sorted(by_id.values(), key=lambda a: a.get("startTimeLocal", ""))


def main() -> None:
    end = date.today()
    start = compute_start(end, RAW_DIR)
    print(f"Synchronisation du {start} au {end} ({(end - start).days + 1} jours)")
    client = connect()

    # 1. Activités de la période, fusionnées avec l'historique
    new = client.get_activities_by_date(start.isoformat(), end.isoformat())
    activities = merge_activities(new, RAW_DIR)
    save(RAW_DIR / "activities.json", activities)
    types = Counter(a.get("activityType", {}).get("typeKey", "?") for a in activities)
    print(f"{len(new)} activités récupérées, {len(activities)} au total : {dict(types)}")

    # 2. Prédictions de course calculées par la montre : référence de comparaison pour l'API
    predictions = safe_call(client.get_race_predictions) if hasattr(client, "get_race_predictions") else None
    if predictions:
        save(RAW_DIR / "race_predictions.json", predictions)
        print("Prédictions de course de la montre enregistrées.")

    # 3. Données quotidiennes : chaque jour de la période, dans l'ordre chronologique.
    # Si le script est interrompu, la prochaine exécution reprend au dernier jour écrit.
    for i in range((end - start).days + 1):
        day = (start + timedelta(days=i)).isoformat()
        print(f"Jour {day}")
        save(RAW_DIR / "daily" / f"{day}.json", {
            "date": day,
            "sleep": safe_call(client.get_sleep_data, day),
            "hrv": safe_call(client.get_hrv_data, day),
            "stats": safe_call(client.get_stats, day),
        })
        time.sleep(PAUSE)

    print(f"Terminé. Données dans {RAW_DIR}")


if __name__ == "__main__":
    main()
