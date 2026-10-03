"""Variables communes à plusieurs sources (Garmin, LifeSnaps), exprimées par rapport à chaque personne.

Principe : deux montres différentes (Garmin, Fitbit) et des personnes très différentes ne sont
comparables qu'en RELATIF. Toutes les variables sont donc rapportées à la référence personnelle,
calculée uniquement sur le passé (fenêtres glissantes vers l'arrière : aucune fuite de l'avenir).

Table commune : person_id, source, date, hrv, resting_hr, sleep_h, load
"""

from pathlib import Path

import numpy as np
import pandas as pd

LIFESNAPS_CSV = Path("public") / "lifesnaps" / "daily_fitbit_sema_df_unprocessed.csv"

# Poids du TRIMP d'Edwards pour les trois zones Fitbit par défaut
# (zone 1 ≈ 50-69 % FCmax, zone 2 ≈ 70-84 %, zone 3 ≥ 85 %) : milieu des poids d'Edwards correspondants
EDWARDS_WEIGHTS = {
    "minutes_in_default_zone_1": 1.5,
    "minutes_in_default_zone_2": 3.5,
    "minutes_in_default_zone_3": 4.75,
}

FEATURES = [
    "hrv_today_rel",   # VFC de la nuit, en écart relatif à la moyenne des 7 jours
    "hrv_trend_rel",   # tendance : moyenne 7 jours vs référence 28 jours
    "hrv_cv28",        # variabilité habituelle de la VFC de la personne
    "rhr_dev",         # FC de repos - sa référence 28 jours (bpm)
    "sleep_h",         # durée de sommeil (h)
    "sleep_dev",       # sommeil - sa référence 28 jours (h)
    "load_rel",        # charge du jour / charge chronique de la personne
    "acwr",            # charge aiguë / charge chronique : sans unité, comparable entre montres
    "is_weekend",
]
TARGET_REL = "target_rel"  # variation de VFC du lendemain, relative à la référence personnelle


# --- Harmonisation vers la table commune ---------------------------------------------------------

def garmin_to_common(gold: pd.DataFrame, person_id: str = "moi") -> pd.DataFrame:
    df = pd.DataFrame({
        "person_id": person_id,
        "source": "garmin",
        "date": pd.to_datetime(gold["date"]),
        # une nuit suspecte (< 4 h) n'est pas une mesure fiable : on l'écarte
        "hrv": gold["hrv_last_night"].where(~gold["night_suspect"].fillna(False).astype(bool)),
        "resting_hr": gold["resting_hr"],
        "sleep_h": gold["sleep_h"],
        "load": gold["load_total"],
    })
    return df


def lifesnaps_to_common(raw: pd.DataFrame) -> pd.DataFrame:
    zones = raw[list(EDWARDS_WEIGHTS)].apply(pd.to_numeric, errors="coerce")
    edwards = sum(zones[col] * weight for col, weight in EDWARDS_WEIGHTS.items())
    df = pd.DataFrame({
        "person_id": raw["id"].astype(str),
        "source": "lifesnaps",
        "date": pd.to_datetime(raw["date"], errors="coerce"),
        "hrv": pd.to_numeric(raw["rmssd"], errors="coerce"),
        "resting_hr": pd.to_numeric(raw["resting_hr"], errors="coerce"),
        "sleep_h": pd.to_numeric(raw["minutesAsleep"], errors="coerce") / 60,
        "load": edwards,  # NaN si la montre n'était pas portée : charge inconnue, pas nulle
    })
    df = df.dropna(subset=["date"])
    # Une ligne par personne et par jour
    return df.groupby(["person_id", "source", "date"], as_index=False).mean(numeric_only=True)


def load_lifesnaps(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / LIFESNAPS_CSV
    if not path.exists():
        return None
    return lifesnaps_to_common(pd.read_csv(path, low_memory=False))


# --- Variables relatives ---------------------------------------------------------------------------

def _complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Un jour par ligne, sans trou, pour chaque personne : décaler d'une ligne = passer au lendemain."""
    parts = []
    for (pid, source), g in df.groupby(["person_id", "source"]):
        g = g.set_index("date").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        g["person_id"], g["source"] = pid, source
        parts.append(g.rename_axis("date").reset_index())
    return pd.concat(parts, ignore_index=True)


def build_features(common: pd.DataFrame) -> pd.DataFrame:
    """Calcule les variables relatives et la cible. Toutes les fenêtres regardent vers le passé."""
    df = _complete_calendar(common)
    by = df.groupby("person_id")

    def past(col, window, how, min_periods):
        return by[col].transform(lambda s: getattr(s.rolling(window, min_periods=min_periods), how)())

    df["hrv_7d"] = past("hrv", 7, "mean", 3)
    df["hrv_base28"] = past("hrv", 28, "mean", 7)
    hrv_sd28 = past("hrv", 28, "std", 7)
    df["hrv_today_rel"] = (df["hrv"] - df["hrv_7d"]) / df["hrv_base28"]
    df["hrv_trend_rel"] = df["hrv_7d"] / df["hrv_base28"] - 1
    df["hrv_cv28"] = hrv_sd28 / df["hrv_base28"]
    df["rhr_dev"] = df["resting_hr"] - past("resting_hr", 28, "mean", 7)
    df["sleep_dev"] = df["sleep_h"] - past("sleep_h", 28, "mean", 7)

    atl = by["load"].transform(lambda s: s.ewm(alpha=1 / 7, adjust=False, ignore_na=True).mean())
    ctl = by["load"].transform(lambda s: s.ewm(alpha=1 / 42, adjust=False, ignore_na=True).mean())
    ctl = ctl.where(ctl > 0)
    df["load_rel"] = df["load"] / ctl
    df["acwr"] = atl / ctl
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(float)

    # Cible : VFC de la nuit suivante, en écart relatif à la moyenne des 7 jours
    df["hrv_next"] = by["hrv"].shift(-1)
    df[TARGET_REL] = (df["hrv_next"] - df["hrv_7d"]) / df["hrv_base28"]
    return df


def usable_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Jours utilisables pour l'apprentissage : VFC du jour, référence et cible connues."""
    ok = features[TARGET_REL].notna() & features["hrv"].notna() & features["hrv_base28"].notna()
    ok &= np.isfinite(features[TARGET_REL])
    return features.loc[ok].sort_values(["person_id", "date"]).reset_index(drop=True)
