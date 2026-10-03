"""Génère un CSV SYNTHÉTIQUE au format du fichier journalier LifeSnaps (Fitbit), pour les tests et la CI.

Sortie : $RUNLAB_DATA_DIR/public/lifesnaps/daily_fitbit_sema_df_unprocessed.csv
(data/sample/... par défaut : ne remplace jamais le vrai fichier)

Chaque participant fictif a son propre niveau de VFC (de 40 à 130 ms, comme sur Fitbit),
porte sa montre de façon irrégulière, et sa VFC baisse un peu quand sa charge dépasse son habitude.
"""

import os
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RUNLAB_DATA_DIR", ROOT / "data" / "sample"))
OUT = DATA_DIR / "public" / "lifesnaps" / "daily_fitbit_sema_df_unprocessed.csv"
N_PEOPLE = 12
SEED = 7


def main() -> None:
    random.seed(SEED)
    rows = []
    for p in range(N_PEOPLE):
        pid = f"synthetique{p:02d}"
        hrv_base, rhr_base = random.uniform(40, 130), random.uniform(50, 70)
        start = date(2021, 5, 1) + timedelta(days=random.randint(0, 30))
        atl, ctl = 0.0, 1.0
        for i in range(random.randint(60, 120)):
            day = start + timedelta(days=i)
            worn = random.random() > 0.15
            z1, z2, z3 = random.uniform(10, 60), random.uniform(0, 30), random.uniform(0, 10)
            load = 1.5 * z1 + 3.5 * z2 + 4.75 * z3
            atl += (load - atl) / 7
            ctl += (load - ctl) / 42
            hrv = hrv_base * (1 - 0.15 * (atl / ctl - 1)) * random.gauss(1, 0.08)
            row = {"Unnamed: 0": len(rows), "id": pid, "date": day.isoformat()}
            if worn:
                row.update({
                    "rmssd": round(hrv, 3),
                    "nremhr": round(rhr_base - 3 + random.gauss(0, 1.5), 3),
                    "resting_hr": round(rhr_base + random.gauss(0, 1.5), 3),
                    "minutesAsleep": round(random.gauss(420, 50)),
                    "minutes_below_default_zone_1": round(random.uniform(1200, 1300)),
                    "minutes_in_default_zone_1": round(z1),
                    "minutes_in_default_zone_2": round(z2),
                    "minutes_in_default_zone_3": round(z3),
                    "very_active_minutes": round(z3 + z2 / 2),
                })
            rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)} jours synthétiques pour {N_PEOPLE} participants écrits dans {OUT}")


if __name__ == "__main__":
    main()
