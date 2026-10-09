"""Point d'entrée unique du pipeline, utilisé par l'image Docker.

Modes :
- demo    : données synthétiques -> silver -> gold (aucun compte nécessaire)
- sync    : export Garmin -> silver -> gold (synchronisation réelle)
- process : silver -> gold, à partir des données brutes déjà présentes
- train   : entraînement et évaluation du modèle de récupération sur la couche gold
- transfer: expérience personnel / global (LifeSnaps) / global + personnel
- matin   : synchronisation complète, puis envoi de la séance du jour sur la montre
- evaluation : backtest des prédictions de temps (erreur mesurée, suivie dans MLflow)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# make_run_labels ajoute les nouvelles sorties au fichier d'étiquetage, sans écraser les étiquettes existantes
TRANSFORM = [["processing/build_silver.py"], ["processing/build_gold.py"], ["scripts/make_run_labels.py"]]
MODES = {
    "demo": [["scripts/generate_sample_data.py"], *TRANSFORM],
    "sync": [["ingestion/garmin_export.py"], *TRANSFORM],
    "process": TRANSFORM,
    "train": [["-m", "ml.train_recovery"]],  # lancé comme module : les modèles maison restent importables
    "transfer": [["-m", "ml.train_transfer"]],
    "evaluation": [["-m", "ml.backtest_predictions"]],
    "matin": [["ingestion/garmin_export.py"], *TRANSFORM, ["-m", "ingestion.garmin_push"]],
}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode not in MODES:
        print(f"Mode inconnu : {mode}. Modes possibles : {', '.join(MODES)}")
        sys.exit(2)

    for step in MODES[mode]:
        print(f"\n==> {' '.join(step)}", flush=True)
        # check=True : si une étape échoue, le pipeline s'arrête avec un code d'erreur
        subprocess.run([sys.executable, *step], cwd=ROOT, check=True)

    print(f"\nPipeline '{mode}' terminé.")


if __name__ == "__main__":
    main()
