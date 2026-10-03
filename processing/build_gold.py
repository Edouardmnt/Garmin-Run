"""Couche gold : indicateurs de charge et de récupération, une ligne par jour.

Entrées : data/silver/activities.parquet, data/silver/daily.parquet
Sortie  : data/gold/daily_features.parquet
"""

import math
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Dossier de données : data/ par défaut, data/sample/ pour la démo
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Coefficients du TRIMP de Banister : homme 0.64 / 1.92, femme 0.86 / 1.67
TRIMP_A, TRIMP_B = 0.64, 1.92


def trimp(duration_min: float, avg_hr: float, hr_rest: float, hr_max: float) -> float:
    """Charge d'une séance : durée x intensité, l'intensité pesant de façon exponentielle."""
    hrr = (avg_hr - hr_rest) / (hr_max - hr_rest)  # fraction de la réserve cardiaque
    hrr = max(0.0, min(1.0, hrr))
    return duration_min * hrr * TRIMP_A * math.exp(TRIMP_B * hrr)


def main() -> None:
    act = pd.read_parquet(SILVER_DIR / "activities.parquet")
    daily = pd.read_parquet(SILVER_DIR / "daily.parquet")
    act["date"] = pd.to_datetime(act["date"])
    daily["date"] = pd.to_datetime(daily["date"])

    # Paramètres personnels estimés à partir des données
    hr_max = act["max_hr"].max()
    hr_rest = daily["resting_hr"].median()
    print(f"FC max observée : {hr_max:.0f} bpm | FC repos médiane : {hr_rest:.0f} bpm")

    # 1. Charge de chaque séance (uniquement celles avec une FC moyenne)
    act = act[act["avg_hr"].notna()].copy()
    act["trimp"] = [
        trimp(d, hr, hr_rest, hr_max) for d, hr in zip(act["duration_min"], act["avg_hr"])
    ]

    # 2. Charge par jour et par sport
    load = act.pivot_table(
        index="date", columns="sport", values="trimp", aggfunc="sum", fill_value=0
    ).add_prefix("load_")
    load["load_total"] = load.sum(axis=1)
    # Heure de début de la dernière séance de la journée (piste ouverte par l'exploration)
    act["start_hour"] = pd.to_datetime(act["start_time"]).dt.hour + pd.to_datetime(act["start_time"]).dt.minute / 60
    load["last_session_hour"] = act.groupby("date")["start_hour"].max()

    # 3. Calendrier complet : un jour sans activité = charge 0
    df = daily.merge(load, left_on="date", right_index=True, how="left").sort_values("date")
    load_cols = [c for c in df.columns if c.startswith("load_")]
    df[load_cols] = df[load_cols].fillna(0)

    # 4. Charge aiguë (fatigue, ~7 j), chronique (forme de fond, ~42 j), équilibre
    df["atl"] = df["load_total"].ewm(alpha=1 / 7, adjust=False).mean()
    df["ctl"] = df["load_total"].ewm(alpha=1 / 42, adjust=False).mean()
    df["tsb"] = df["ctl"] - df["atl"]  # négatif = plus fatigué que d'habitude
    df["acwr"] = df["atl"] / df["ctl"].where(df["ctl"] > 0)

    # 5. Contexte du jour
    df["weekday"] = df["date"].dt.weekday  # 0 = lundi
    df["is_rest_day"] = df["load_total"] == 0

    # 6. Récupération : une nuit absente n'est PAS une nuit à zéro
    df["night_tracked"] = df["sleep_s"].notna()
    # Nuit de moins de 4 h : probablement enregistrée en partie seulement (montre mise en cours de nuit)
    df["night_suspect"] = df["sleep_h"] < 4
    # Cibles du futur modèle : la récupération de la nuit SUIVANTE
    df["hrv_next"] = df["hrv_last_night"].shift(-1)
    df["resting_hr_next"] = df["resting_hr"].shift(-1)
    df["next_night_suspect"] = df["night_suspect"].shift(-1, fill_value=False)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GOLD_DIR / "daily_features.parquet", index=False)

    # Résumé
    print(f"\n{len(act)} séances avec FC, {len(df)} jours")
    print("\nCharge totale par sport (TRIMP) :")
    print(act.groupby("sport")["trimp"].agg(["count", "sum", "mean"]).round(0).to_string())
    tracked = df[df["night_tracked"] & df["hrv_next"].notna()]
    print(f"\nPremier aperçu sur {len(tracked)} jours suivis :")
    print(tracked[["load_total", "hrv_next", "resting_hr_next"]].corr().round(2).to_string())
    print(f"\nTable écrite dans {GOLD_DIR}")


if __name__ == "__main__":
    main()
