"""Point d'entrée de la démo en ligne sur Streamlit Community Cloud (fichier principal : dashboard/streamlit_app.py).

Même démo que « python scripts/demo.py », sans sous-processus Streamlit : on fixe l'environnement de démo,
on génère les données synthétiques une seule fois par conteneur, puis on exécute le tableau de bord.
Les dépendances viennent de dashboard/requirements.txt (Community Cloud cherche d'abord à côté de ce fichier).
"""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.demo import DEMO_ENV, PIPELINE, seed_goal  # noqa: E402

for key, value in DEMO_ENV.items():
    os.environ.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def prepare_demo_data() -> bool:
    """Bronze -> silver -> gold sur les données synthétiques, une fois par conteneur (quelques secondes)."""
    data = Path(os.environ["RUNLAB_DATA_DIR"])
    if not (data / "gold" / "daily_features.parquet").exists():
        for script in PIPELINE:
            subprocess.run([sys.executable, script], cwd=ROOT, env=dict(os.environ), check=True,
                           stdout=subprocess.DEVNULL)
        seed_goal(data)
    return True


prepare_demo_data()
runpy.run_path(str(Path(__file__).parent / "app.py"), run_name="__main__")
