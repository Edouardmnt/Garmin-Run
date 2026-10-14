"""Météo de chaque sortie en extérieur (Open-Meteo) -> $RUNLAB_DATA_DIR/raw/weather/<activity_id>.json (bronze).

Pour chaque sortie avec une position de départ et sans météo enregistrée : les relevés heure par heure couvrant
la sortie (température, humidité, point de rosée, température ressentie, vent, pluie).

- Sorties de plus de ARCHIVE_DELAY_DAYS jours : réanalyse historique (archive-api, ERA5, 1940 à aujourd'hui) ;
- sorties récentes : modèle de prévision avec ses jours passés (la réanalyse a quelques jours de retard).
Une seule requête par lieu (arrondi à 0,1°, ~10 km) et par source : quelques appels suffisent pour des mois.

Incrémental (une sortie déjà traitée n'est jamais redemandée), et sans effet bloquant : si Open-Meteo ne répond
pas, la synchronisation continue et la météo sera récupérée la fois suivante.

Données : Open-Meteo.com, licence CC BY 4.0 (https://open-meteo.com/en/licence). Gratuit pour un usage non
commercial, sans clé.
"""

import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "wind_speed_10m",
          "precipitation"]
ARCHIVE_DELAY_DAYS = 6    # la réanalyse est publiée avec environ 5 jours de retard
FORECAST_PAST_DAYS = 92   # au-delà, le modèle de prévision ne garde pas l'historique
TIMEOUT_S = 30


def weather_dir(raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "weather"


def start_of(activity: dict) -> datetime:
    return datetime.fromisoformat(activity["startTimeLocal"])


def pending(activities: list[dict], raw_dir: Path, now: datetime) -> list[dict]:
    """Sorties en extérieur, terminées, avec une position de départ, et sans météo enregistrée."""
    out = []
    for a in activities:
        if a.get("startLatitude") is None or a.get("startLongitude") is None or not a.get("startTimeLocal"):
            continue
        end = start_of(a) + timedelta(seconds=a.get("duration") or 0)
        if end > now or (weather_dir(raw_dir) / f"{a['activityId']}.json").exists():
            continue
        out.append(a)
    return out


def plan_requests(activities: list[dict], today: date) -> list[dict]:
    """Regroupe les sorties par lieu (0,1°) et par source : une requête par groupe."""
    groups = defaultdict(list)
    for a in activities:
        day = start_of(a).date()
        age = (today - day).days
        if age > ARCHIVE_DELAY_DAYS:
            source = "archive"
        elif age <= FORECAST_PAST_DAYS:
            source = "prevision"
        else:  # impossible en pratique (6 < 92), gardé pour la lisibilité
            continue
        groups[(round(a["startLatitude"], 1), round(a["startLongitude"], 1), source)].append(a)
    requests_ = []
    for (lat, lon, source), items in groups.items():
        days = sorted(start_of(a).date() for a in items)
        params = {"latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY), "timezone": "auto",
                  "wind_speed_unit": "kmh"}
        if source == "archive":
            params |= {"start_date": days[0].isoformat(), "end_date": days[-1].isoformat()}
            url = ARCHIVE_URL
        else:
            params |= {"past_days": min(FORECAST_PAST_DAYS, (today - days[0]).days + 1), "forecast_days": 1}
            url = FORECAST_URL
        requests_.append({"url": url, "params": params, "source": source, "activities": items})
    return requests_


def hours_of(activity: dict, hourly: dict) -> list[dict]:
    """Relevés horaires qui couvrent la sortie (de l'heure de départ à l'heure de fin)."""
    start = start_of(activity).replace(minute=0, second=0, microsecond=0)
    end = start_of(activity) + timedelta(seconds=activity.get("duration") or 0)
    out = []
    for i, stamp in enumerate(hourly.get("time", [])):
        t = datetime.fromisoformat(stamp)
        if start <= t <= end:
            record = {"time": stamp, **{k: hourly[k][i] for k in HOURLY if k in hourly}}
            if record.get("temperature_2m") is not None:
                out.append(record)
    return out


def fetch_all(activities: list[dict], raw_dir: Path, today: date, get=None) -> dict:
    """Récupère et enregistre la météo des sorties en attente. `get` : requests.get (remplaçable dans les tests)."""
    if get is None:
        import requests

        get = requests.get
    saved, failed = 0, 0
    weather_dir(raw_dir).mkdir(parents=True, exist_ok=True)
    for request in plan_requests(activities, today):
        try:
            response = get(request["url"], params=request["params"], timeout=TIMEOUT_S)
            response.raise_for_status()
            hourly = response.json()["hourly"]
        except Exception as exc:  # réseau, quota, réponse inattendue : on réessaiera à la prochaine synchro
            print(f"Météo indisponible pour {len(request['activities'])} sortie(s) ({request['source']}) : {exc}")
            failed += len(request["activities"])
            continue
        for a in request["activities"]:
            hours = hours_of(a, hourly)
            if not hours:
                continue  # relevés absents (réanalyse pas encore publiée) : nouvelle tentative plus tard
            record = {"activityId": a["activityId"], "source": f"open-meteo:{request['source']}",
                      "latitude": request["params"]["latitude"], "longitude": request["params"]["longitude"],
                      "hourly": hours}
            path = weather_dir(raw_dir) / f"{a['activityId']}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
            saved += 1
    return {"enregistrees": saved, "echecs": failed}


def forecast(latitude: float, longitude: float, day: date, hour: int, get=None) -> dict:
    """Prévision à une heure donnée (jusqu'à 16 jours) : température, humidité, vent, pluie."""
    if get is None:
        import requests

        get = requests.get
    params = {"latitude": round(latitude, 2), "longitude": round(longitude, 2), "hourly": ",".join(HOURLY),
              "timezone": "auto", "wind_speed_unit": "kmh", "start_date": day.isoformat(), "end_date": day.isoformat()}
    response = get(FORECAST_URL, params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    stamp = f"{day.isoformat()}T{hour:02d}:00"
    i = hourly["time"].index(stamp)
    return {"temperature_c": hourly["temperature_2m"][i], "humidite_pct": hourly["relative_humidity_2m"][i],
            "vent_kmh": hourly["wind_speed_10m"][i], "pluie_mm": hourly["precipitation"][i], "heure": stamp}


def main() -> None:
    path = RAW_DIR / "activities.json"
    if not path.exists():
        print("Aucune activité : rien à faire.")
        return
    activities = json.loads(path.read_text(encoding="utf-8"))
    todo = pending(activities, RAW_DIR, datetime.now())
    if not todo:
        print("Météo : toutes les sorties en extérieur sont déjà renseignées.")
        return
    result = fetch_all(todo, RAW_DIR, date.today())
    print(f"Météo : {result['enregistrees']} sortie(s) renseignée(s), {result['echecs']} en attente "
          "(données Open-Meteo.com, CC BY 4.0).")


if __name__ == "__main__":
    main()
