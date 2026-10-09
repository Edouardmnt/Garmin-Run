"""Démo complète en une commande : données synthétiques -> pipeline -> tableau de bord.

Sans compte Garmin ni Ollama : les données sont générées, l'API tourne dans le même processus que le tableau
de bord, et le coach répond par des règles (clairement indiqué dans l'interface).

    python scripts/demo.py                 # http://localhost:8502
    python scripts/demo.py --regenerer     # repart de données neuves
    python scripts/demo.py --port 7860 --public   # Hugging Face Spaces (voir deploy/huggingface/)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_ENV = {
    "RUNLAB_DEMO": "1",
    "RUNLAB_DATA_DIR": str(ROOT / "data" / "demo"),
    "RUNLAB_API_URL": "inprocess",   # l'API FastAPI est appelée dans le même processus
    "RUNLAB_LLM": "demo",            # coach par règles, sans modèle de langage
    "RUNLAB_AUTO_SYNC": "0",
    "RUNLAB_SYNC_STEPS": "demo",
    "RUNLAB_WATCH_AUTO": "0",        # jamais d'appel à Garmin
    "RUNLAB_TODAY": "2026-09-30",    # dernier jour des données synthétiques : « aujourd'hui » reste cohérent
    "RUNLAB_MLFLOW": "0",
    "PYTHONUTF8": "1",
}
PIPELINE = ["scripts/generate_sample_data.py", "processing/build_silver.py", "processing/build_gold.py",
            "scripts/make_run_labels.py"]


def seed_goal(data: Path) -> None:
    """Un objectif d'exemple : le planning montre alors ses phases jusqu'à la course et les jours de tennis."""
    sys.path.insert(0, str(ROOT))
    from processing.goals import create_goal, load_goals

    if not load_goals(data):
        create_goal(data, {"nom": "Semi d'automne (démo)", "distance": "semi", "date_course": "2026-12-06",
                           "temps_vise": "1:52:00", "denivele_m": 60, "seances_par_semaine": 3,
                           "jours_tennis": [1, 3], "jour_sortie_longue": 6})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="8502", help="8502 par défaut (8501 sert souvent déjà au cluster)")
    parser.add_argument("--public", action="store_true", help="écouter sur toutes les interfaces (conteneur)")
    parser.add_argument("--regenerer", action="store_true", help="regénérer les données synthétiques")
    args = parser.parse_args()

    env = {**os.environ, **{k: v for k, v in DEMO_ENV.items() if k not in os.environ}}
    data = Path(env["RUNLAB_DATA_DIR"])
    if args.regenerer or not (data / "gold" / "daily_features.parquet").exists():
        print("Préparation des données synthétiques (bronze -> silver -> gold)...")
        for script in PIPELINE:
            subprocess.run([sys.executable, script], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        seed_goal(data)
    print(f"Foulée (démo) : http://localhost:{args.port}")
    command = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", args.port,
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    if args.public:
        command += ["--server.address", "0.0.0.0"]
    sys.exit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
