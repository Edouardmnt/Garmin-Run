"""Profil du jeu public LifeSnaps : vérifie qu'il est exploitable AVANT toute harmonisation.

Source : Yfantidou et al. (2022), LifeSnaps, Zenodo, doi:10.5281/zenodo.7229547 (CC BY 4.0).
Fichier attendu : data/public/lifesnaps/daily_fitbit_sema_df_unprocessed.csv (jamais commité).

Question décisive : combien de paires de nuits CONSÉCUTIVES avec une VFC ?
Le modèle de récupération prédit la VFC du lendemain : sans paires consécutives, pas de cible.
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
CSV = DATA_DIR / "public" / "lifesnaps" / "daily_fitbit_sema_df_unprocessed.csv"

USEFUL_COLUMNS = {
    "rmssd": "VFC (RMSSD) : la cible",
    "resting_hr": "FC de repos",
    "nremhr": "FC en sommeil profond",
    "sleep_duration": "durée de sommeil",
    "minutesAsleep": "minutes endormi",
    "sleep_efficiency": "efficacité du sommeil",
    "stress_score": "score de stress",
    "minutes_below_default_zone_1": "minutes sous la zone 1",
    "minutes_in_default_zone_1": "minutes en zone 1",
    "minutes_in_default_zone_2": "minutes en zone 2",
    "minutes_in_default_zone_3": "minutes en zone 3",
    "moderately_active_minutes": "minutes d'activité modérée",
    "very_active_minutes": "minutes d'activité intense",
}


def main() -> None:
    if not CSV.exists():
        print(f"Fichier introuvable : {CSV}")
        print("Télécharge rais_anonymized.zip sur https://zenodo.org/records/7229547 et extrais-y le CSV journalier.")
        sys.exit(1)

    df = pd.read_csv(CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"{len(df)} lignes, {df.shape[1]} colonnes, {df['id'].nunique()} participants")
    print(f"Période : du {df['date'].min():%Y-%m-%d} au {df['date'].max():%Y-%m-%d}")

    print("\nColonnes utiles (% de lignes renseignées) :")
    for col, label in USEFUL_COLUMNS.items():
        if col in df.columns:
            print(f"  {col:<30} {df[col].notna().mean() * 100:5.1f} %   {label}")
        else:
            print(f"  {col:<30}  ABSENTE   {label}")

    # Paires de nuits consécutives avec une VFC, par participant
    hrv = df.loc[df["rmssd"].notna(), ["id", "date"]].drop_duplicates().sort_values(["id", "date"])
    gap = hrv.groupby("id")["date"].shift(-1) - hrv["date"]
    hrv["has_next_night"] = gap.dt.days == 1
    per_user = hrv.groupby("id").agg(nights=("date", "size"), pairs=("has_next_night", "sum"))

    print(f"\nNuits avec VFC : {len(hrv)}, dont {int(hrv['has_next_night'].sum())} suivies d'une nuit avec VFC")
    print(f"Participants avec au moins 30 paires consécutives : {(per_user['pairs'] >= 30).sum()} sur {len(per_user)}")
    print("\nRépartition des paires par participant :")
    print(per_user["pairs"].describe().round(1).to_string())

    pairs = int(hrv["has_next_night"].sum())
    print("\nVerdict :", end=" ")
    if pairs >= 1000:
        print(f"exploitable ({pairs} paires, environ {pairs / 129:.0f} fois tes données).")
    elif pairs >= 300:
        print(f"utilisable avec prudence ({pairs} paires).")
    else:
        print(f"peu exploitable pour ce modèle ({pairs} paires seulement).")

    print("\nToutes les colonnes du fichier :")
    print(", ".join(df.columns))


if __name__ == "__main__":
    main()
