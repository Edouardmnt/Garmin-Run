"""Couche silver : transforme les JSON bruts Garmin (data/raw) en tables propres (data/silver).

Sorties :
- data/silver/activities.parquet : une ligne par activité
- data/silver/daily.parquet      : une ligne par jour (sommeil, VFC, repos, stress)
- data/silver/splits.parquet     : une ligne par tour (souvent 1 km) des sorties de course récentes

Le script ne modifie jamais data/raw : on peut le relancer à volonté.
"""

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Dossier de données : data/ par défaut, data/sample/ pour la démo
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SILVER_DIR = DATA_DIR / "silver"

# Harmonisation des types Garmin -> (sport, intérieur ?)
SPORT_MAP = {
    "running": ("running", False),
    "treadmill_running": ("running", True),
    "trail_running": ("running", False),
    "tennis": ("tennis", False),
    "tennis_v2": ("tennis", False),
    "strength_training": ("strength", True),
    "walking": ("walking", False),
    "hiking": ("hiking", False),
}


def dig(data, *keys):
    """Lit une valeur imbriquée sans planter si une clé manque : dig(d, 'a', 'b')."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def build_activities() -> pd.DataFrame:
    raw = json.loads((RAW_DIR / "activities.json").read_text(encoding="utf-8"))
    rows = []
    for a in raw:
        type_key = dig(a, "activityType", "typeKey") or "other"
        sport, indoor = SPORT_MAP.get(type_key, ("other", None))
        rows.append({
            "activity_id": a.get("activityId"),
            "start_time": a.get("startTimeLocal"),
            "garmin_type": type_key,
            "sport": sport,
            "indoor": indoor,
            "duration_s": a.get("duration"),
            "moving_duration_s": a.get("movingDuration"),
            "distance_m": a.get("distance"),
            "elevation_gain_m": a.get("elevationGain"),
            "elevation_loss_m": a.get("elevationLoss"),
            # Vitesse ajustée à la pente (GAP) calculée par Garmin, quand la montre la fournit
            "avg_gap_speed_ms": a.get("avgGradeAdjustedSpeed"),
            "avg_speed_ms": a.get("averageSpeed"),
            "avg_hr": a.get("averageHR"),
            "max_hr": a.get("maxHR"),
            "calories": a.get("calories"),
            "aerobic_te": a.get("aerobicTrainingEffect"),
            "anaerobic_te": a.get("anaerobicTrainingEffect"),
            "vo2max": a.get("vO2MaxValue"),
            # Meilleurs temps (s) sur 1 km, 5 km et 10 km à l'intérieur de la sortie, calculés par Garmin
            "fastest_1k_s": a.get("fastestSplit_1000"),
            "fastest_5k_s": a.get("fastestSplit_5000"),
            "fastest_10k_s": a.get("fastestSplit_10000"),
            # Temps (s) passé dans chaque zone cardiaque de la montre
            **{f"hr_z{z}_s": a.get(f"hrTimeInZone_{z}") for z in range(1, 6)},
        })

    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["date"] = df["start_time"].dt.date
    df["duration_min"] = df["duration_s"] / 60
    df["distance_km"] = df["distance_m"] / 1000

    # Allure (min/km) seulement pour la course avec une distance exploitable
    is_run = (df["sport"] == "running") & (df["distance_km"] > 0.5)
    df["pace_min_km"] = (df["duration_min"] / df["distance_km"]).where(is_run)

    return df.sort_values("start_time").reset_index(drop=True)


def build_splits() -> pd.DataFrame:
    """Une ligne par tour (souvent 1 km) des sorties dont les tours ont été récupérés."""
    rows = []
    for path in sorted((RAW_DIR / "splits").glob("*.json")) if (RAW_DIR / "splits").exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        for lap in (data or {}).get("lapDTOs") or []:
            rows.append({
                "activity_id": int(data.get("activityId") or path.stem),
                "lap": lap.get("lapIndex"),
                "distance_m": lap.get("distance"),
                "duration_s": lap.get("movingDuration") or lap.get("duration"),
                "avg_hr": lap.get("averageHR"),
                "max_hr": lap.get("maxHR"),
                "elevation_gain_m": lap.get("elevationGain"),
                "cadence": lap.get("averageRunCadence"),
            })
    cols = ["activity_id", "lap", "distance_m", "duration_s", "avg_hr", "max_hr", "elevation_gain_m", "cadence"]
    return pd.DataFrame(rows, columns=cols)


def build_daily() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW_DIR / "daily").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        sleep, hrv, stats = d.get("sleep"), d.get("hrv"), d.get("stats")
        rows.append({
            "date": d.get("date"),
            # Sommeil
            "sleep_s": dig(sleep, "dailySleepDTO", "sleepTimeSeconds"),
            "deep_sleep_s": dig(sleep, "dailySleepDTO", "deepSleepSeconds"),
            "light_sleep_s": dig(sleep, "dailySleepDTO", "lightSleepSeconds"),
            "rem_sleep_s": dig(sleep, "dailySleepDTO", "remSleepSeconds"),
            "awake_s": dig(sleep, "dailySleepDTO", "awakeSleepSeconds"),
            "sleep_score": dig(sleep, "dailySleepDTO", "sleepScores", "overall", "value"),
            # Heures de coucher et de lever, en heure locale (horodatage Garmin en millisecondes)
            "sleep_start_ms": dig(sleep, "dailySleepDTO", "sleepStartTimestampLocal"),
            "sleep_end_ms": dig(sleep, "dailySleepDTO", "sleepEndTimestampLocal"),
            # Variabilité de la fréquence cardiaque (VFC / HRV)
            "hrv_last_night": dig(hrv, "hrvSummary", "lastNightAvg"),
            "hrv_weekly_avg": dig(hrv, "hrvSummary", "weeklyAvg"),
            "hrv_status": dig(hrv, "hrvSummary", "status"),
            # Journée
            "resting_hr": dig(stats, "restingHeartRate"),
            "steps": dig(stats, "totalSteps"),
            "avg_stress": dig(stats, "averageStressLevel"),
            "max_stress": dig(stats, "maxStressLevel"),
            "high_stress_s": dig(stats, "highStressDuration"),
            "body_battery_charged": dig(stats, "bodyBatteryChargedValue"),
            "body_battery_drained": dig(stats, "bodyBatteryDrainedValue"),
            "body_battery_max": dig(stats, "bodyBatteryHighestValue"),
            "body_battery_min": dig(stats, "bodyBatteryLowestValue"),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sleep_h"] = df["sleep_s"] / 3600

    def local_hour(ms):
        t = pd.to_datetime(ms, unit="ms", errors="coerce")
        return t.dt.hour + t.dt.minute / 60

    # Heure de coucher sur une échelle continue : 23 h = 23, 1 h du matin = 25
    start = local_hour(df["sleep_start_ms"])
    df["bedtime_h"] = start.where(start >= 12, start + 24)
    df["wake_h"] = local_hour(df["sleep_end_ms"])
    df["high_stress_min"] = df["high_stress_s"] / 60
    df = df.drop(columns=["sleep_start_ms", "sleep_end_ms", "high_stress_s"])
    return df


def quality_report(name: str, df: pd.DataFrame) -> None:
    """Contrôle qualité : taux de valeurs manquantes par colonne."""
    print(f"\n=== {name} : {len(df)} lignes, {df.shape[1]} colonnes")
    missing = (df.isna().mean() * 100).round(0)
    print("Valeurs manquantes (%) :")
    print(missing[missing > 0].sort_values(ascending=False).to_string())


def main() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    activities = build_activities()
    activities.to_parquet(SILVER_DIR / "activities.parquet", index=False)
    quality_report("activities", activities)
    print("\nActivités par sport :")
    print(activities["sport"].value_counts().to_string())

    splits = build_splits()
    splits.to_parquet(SILVER_DIR / "splits.parquet", index=False)
    print(f"\nTours au km : {len(splits)} tours, {splits['activity_id'].nunique()} sorties")

    daily = build_daily()
    daily.to_parquet(SILVER_DIR / "daily.parquet", index=False)
    quality_report("daily", daily)

    print(f"\nTables écrites dans {SILVER_DIR}")


if __name__ == "__main__":
    main()
