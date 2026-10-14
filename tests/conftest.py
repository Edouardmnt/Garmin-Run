"""Fixtures partagées : le pipeline complet est exécuté une seule fois pour toute la session de tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Les données synthétiques s'arrêtent au 30/09/2026 : « aujourd'hui » est fixé au même jour, pour que les tests
# (séances à venir, propositions du coach) donnent le même résultat quelle que soit la date réelle.
os.environ.setdefault("RUNLAB_TODAY", "2026-09-30")
SCRIPTS = [
    "scripts/generate_sample_data.py",
    "scripts/generate_sample_lifesnaps.py",
    "processing/build_silver.py",
    "processing/build_gold.py",
    # Étiquettes créées d'emblée : avec les tests en parallèle (-n auto), aucun test ne doit compter sur un autre
    "scripts/make_run_labels.py",
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Données synthétiques -> bronze -> silver -> gold, dans un dossier temporaire."""
    out = tmp_path_factory.mktemp("data")
    env = {**os.environ, "RUNLAB_DATA_DIR": str(out)}
    for script in SCRIPTS:
        subprocess.run([sys.executable, script], cwd=ROOT, env=env, check=True)
    return out


# --- Tests rapides / tests complets -----------------------------------------------------------------------
# Les tests marqués « slow » simulent l'interface Streamlit en entier, attendent une synchronisation en
# arrière-plan ou entraînent des modèles. Ils tournent toujours dans la CI (qui seule publie l'image) ;
# en local, `pytest -m "not slow"` donne un retour en quelques dizaines de secondes avant chaque commit.
SLOW_MODULES = {"test_dashboard.py", "test_demo.py", "test_sync.py", "test_transfer.py", "test_ml.py"}
SLOW_TESTS = {"test_evaluation_complete_avec_le_faux_modele", "test_le_pipeline_met_a_jour_l_etiquetage"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name in SLOW_MODULES or item.originalname in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
