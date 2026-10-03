"""Génère des données Garmin SYNTHÉTIQUES, au même format que l'export réel (couche bronze).

Permet de lancer tout le pipeline sans compte Garmin (démo, tests, CI).
Sortie : data/sample/raw/ par défaut, ou $RUNLAB_DATA_DIR/raw/ si la variable est définie

Les données sont fictives mais réalistes : tennis le soir, course, musculation,
nuits non suivies, VFC qui baisse légèrement quand la charge s'accumule.
"""

import json
import math
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Écrit dans $RUNLAB_DATA_DIR/raw (data/sample/raw par défaut)
OUT_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data" / "sample")) / "raw"
DAYS = 180
SEED = 42  # même graine = mêmes données à chaque exécution (reproductible)

HR_REST, HR_MAX = 50, 190


def make_activity(activity_id: int, day: date, sport: str) -> dict:
    """Une activité au format de la liste d'activités Garmin."""
    if sport == "tennis_v2":
        start_h, duration_min, avg_hr = random.choice([18, 19, 20, 21]), random.uniform(60, 100), random.uniform(130, 155)
        distance, elevation, speed, vo2 = None, None, None, None
    elif sport in ("running", "treadmill_running"):
        start_h, duration_min, avg_hr = random.choice([7, 8, 12, 18]), random.uniform(25, 70), random.uniform(140, 168)
        pace_min_km = random.uniform(4.8, 6.0)
        distance = duration_min / pace_min_km * 1000
        speed = distance / (duration_min * 60)
        outdoor = sport == "running"
        elevation = random.uniform(10, 120) if outdoor else None
        vo2 = random.choice([49.0, 50.0, 51.0]) if outdoor else None
    else:  # strength_training
        start_h, duration_min, avg_hr = random.choice([12, 18]), random.uniform(40, 70), random.uniform(95, 115)
        distance, elevation, speed, vo2 = None, None, None, None

    start = datetime.combine(day, datetime.min.time()) + timedelta(hours=start_h, minutes=random.randint(0, 45))
    return {
        "activityId": activity_id,
        "activityName": "Séance synthétique",
        "startTimeLocal": start.strftime("%Y-%m-%d %H:%M:%S"),
        "activityType": {"typeKey": sport},
        "duration": duration_min * 60,
        "movingDuration": duration_min * 60 * 0.95,
        "distance": distance,
        "elevationGain": elevation,
        "averageSpeed": speed,
        "averageHR": round(avg_hr),
        "maxHR": round(min(HR_MAX, avg_hr + random.uniform(15, 30))),
        "calories": round(duration_min * random.uniform(8, 12)),
        "aerobicTrainingEffect": round(random.uniform(2.0, 4.0), 1),
        "anaerobicTrainingEffect": round(random.uniform(0.5, 2.5), 1),
        "vO2MaxValue": vo2,
        **fastest_splits(distance, speed),
    }


def fastest_splits(distance, speed) -> dict:
    """Meilleurs temps sur 1, 5 et 10 km dans la sortie (un peu plus rapides que l'allure moyenne)."""
    if not distance or not speed:
        return {}
    out = {}
    for split in (1000, 5000, 10000):
        if distance >= split:
            out[f"fastestSplit_{split}"] = round(split / (speed * (1.06 if split == 1000 else 1.02)), 1)
    return out


def trimp(duration_s: float, avg_hr: float) -> float:
    hrr = (avg_hr - HR_REST) / (HR_MAX - HR_REST)
    return duration_s / 60 * hrr * 0.64 * math.exp(1.92 * hrr)


def main() -> None:
    random.seed(SEED)
    start = date.today() - timedelta(days=DAYS)
    activities, atl, hrv_base = [], 0.0, 65.0
    (OUT_DIR / "daily").mkdir(parents=True, exist_ok=True)

    for i in range(DAYS + 1):
        day = start + timedelta(days=i)
        weekday = day.weekday()  # 0 = lundi

        # Planning type : tennis mardi/jeudi soir, course lundi/samedi, muscu mercredi
        planned = {0: ["running"], 1: ["tennis_v2"], 2: ["strength_training"], 3: ["tennis_v2"], 5: ["running"]}
        today = [s for s in planned.get(weekday, []) if random.random() < 0.6]
        if "running" in today and random.random() < 0.15:
            today = ["treadmill_running"]

        load = 0.0
        for sport in today:
            act = make_activity(len(activities) + 1, day, sport)
            activities.append(act)
            load += trimp(act["duration"], act["averageHR"])

        # Récupération : la charge accumulée fait légèrement baisser la VFC
        atl += (load - atl) / 7
        hrv_base += random.gauss(0, 0.5)
        hrv = hrv_base - 0.08 * atl + random.gauss(0, 7)
        resting_hr = HR_REST + 0.04 * atl - 0.1 * (hrv - 65) + random.gauss(0, 1.5)
        weekend = weekday >= 5
        sleep_h = random.gauss(6.6 if weekend else 7.3, 0.8)

        # ~8 % de nuits sans montre : le sommeil et la VFC sont absents (None), pas à zéro
        night_tracked = random.random() > 0.08
        sleep = None
        if night_tracked:
            sleep_s = max(3.0, sleep_h) * 3600
            sleep = {"dailySleepDTO": {
                "calendarDate": day.isoformat(),
                "sleepTimeSeconds": round(sleep_s),
                "deepSleepSeconds": round(sleep_s * 0.18),
                "lightSleepSeconds": round(sleep_s * 0.55),
                "remSleepSeconds": round(sleep_s * 0.22),
                "awakeSleepSeconds": round(sleep_s * 0.05),
                "sleepScores": {"overall": {"value": int(min(100, max(30, 50 + 6 * (sleep_h - 6) + random.gauss(0, 6))))}},
            }}
        record = {
            "date": day.isoformat(),
            "sleep": sleep,
            "hrv": {"hrvSummary": {
                "lastNightAvg": round(hrv),
                "weeklyAvg": round(hrv_base),
                "status": "BALANCED" if hrv > 55 else "UNBALANCED",
            }} if night_tracked else None,
            "stats": {
                "restingHeartRate": round(resting_hr),
                "totalSteps": random.randint(5000, 14000),
                "averageStressLevel": random.randint(20, 45),
                "bodyBatteryHighestValue": random.randint(60, 100),
                "bodyBatteryLowestValue": random.randint(5, 30),
            },
        }
        (OUT_DIR / "daily" / f"{day.isoformat()}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    (OUT_DIR / "activities.json").write_text(json.dumps(activities, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(activities)} activités et {DAYS + 1} jours synthétiques écrits dans {OUT_DIR}")


if __name__ == "__main__":
    main()
