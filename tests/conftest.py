"""Fixtures partagées : le pipeline complet est exécuté une seule fois pour toute la session de tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/generate_sample_data.py",
    "scripts/generate_sample_lifesnaps.py",
    "processing/build_silver.py",
    "processing/build_gold.py",
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Données synthétiques -> bronze -> silver -> gold, dans un dossier temporaire."""
    out = tmp_path_factory.mktemp("data")
    env = {**os.environ, "RUNLAB_DATA_DIR": str(out)}
    for script in SCRIPTS:
        subprocess.run([sys.executable, script], cwd=ROOT, env=env, check=True)
    return out
