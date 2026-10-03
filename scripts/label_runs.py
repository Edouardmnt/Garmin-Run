"""Étiquetage interactif de mes sorties de course, directement dans le terminal.

Pour chaque sortie sans étiquette : Entrée = accepter la suggestion,
e = ef, t = tempo, f = fractionne, c = course, s = passer, q = quitter.
Chaque réponse est enregistrée immédiatement dans data/labels/run_labels.csv.
Prérequis : avoir lancé scripts/make_run_labels.py.
"""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data"))
FILE = DATA_DIR / "labels" / "run_labels.csv"
KEYS = {"e": "ef", "t": "tempo", "f": "fractionne", "c": "course"}


def load() -> pd.DataFrame:
    return pd.read_csv(FILE, sep=";", decimal=",", dtype={"label": str})


def save(df: pd.DataFrame) -> None:
    df.to_csv(FILE, index=False, sep=";", decimal=",", encoding="utf-8-sig")


def show(row: pd.Series, position: str) -> None:
    pace = row["pace_min_km"]
    pace_txt = f"{int(pace)}'{round((pace % 1) * 60):02d}\"/km" if pd.notna(pace) else "-"
    print(f"\n[{position}] {row['date']}  {row['type']}")
    print(f"  {row['distance_km']:.2f} km en {row['duration_min']:.0f} min, allure {pace_txt}")
    print(f"  FC moyenne {row['avg_hr']:.0f}, max {row['max_hr']:.0f}")
    print(f"  Zones : 1-2 = {row['z1_z2']:.0%}   3 = {row['z3']:.0%}   4-5 = {row['z4_z5']:.0%}")
    intervals = "oui" if str(row["has_intervals"]) == "True" else "non"
    print(f"  Intervalles détectés par Garmin : {intervals} ({row['interval_reps']} portions), {row['lap_count']} tours")
    print(f"  Suggestion : {row['suggestion']}")


def main() -> None:
    df = load()
    todo = df.index[df["label"].isna() | (df["label"].str.strip() == "")]
    print(f"{len(todo)} sorties à étiqueter.")
    print("Entrée = suggestion | e = ef | t = tempo | f = fractionne | c = course | s = passer | q = quitter")

    for n, i in enumerate(todo, start=1):
        show(df.loc[i], f"{n}/{len(todo)}")
        while True:
            answer = input("  Ta classe : ").strip().lower()
            if answer == "q":
                print(f"\nArrêt. Étiquettes enregistrées dans {FILE}")
                return
            if answer == "s":
                break
            if answer == "" and df.loc[i, "suggestion"] in KEYS.values():
                df.loc[i, "label"] = df.loc[i, "suggestion"]
            elif answer in KEYS:
                df.loc[i, "label"] = KEYS[answer]
            else:
                print("  Réponse non reconnue : Entrée, e, t, f, c, s ou q.")
                continue
            save(df)  # enregistrement immédiat : rien n'est perdu si on s'arrête
            print(f"  -> {df.loc[i, 'label']}")
            break

    labelled = df["label"].notna() & (df["label"].str.strip() != "")
    print(f"\nTerminé : {labelled.sum()} sorties étiquetées sur {len(df)}.")


if __name__ == "__main__":
    main()
