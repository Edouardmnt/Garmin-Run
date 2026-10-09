"""Démo en ligne (Hugging Face Spaces) : coach par règles, tableau de bord en mode démo, fichiers publiés."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from processing.coach import DemoLLM, get_llm

ROOT = Path(__file__).resolve().parents[1]


def system(context: dict, extra: dict | None = None) -> list[dict]:
    data = json.dumps(context, ensure_ascii=False)
    if extra:
        data += "\n\nDONNÉES UTILES : " + json.dumps(extra, ensure_ascii=False)
    return [{"role": "system", "content": f"CONTEXTE (données de l'utilisateur, à jour) :\n{data}"}]


CONTEXT = {"verdict_du_jour": {"titre": "Feu vert", "score": 90, "explication": "Ta VFC est haute."},
           "analyses": {"sommeil": "Tu as dormi 7,2 h en moyenne."},
           "prochaines_seances": [{"date": "2026-10-02", "jour": "Vendredi", "type": "fractionne",
                                   "titre": "Fractionné", "allure": "4'35\"/km"}]}


def test_coach_demo_repond_avec_les_seules_donnees():
    llm = DemoLLM()
    paces = {"allures": {"zones": {"ef": {"allure": "4'37\"/km à 5'15\"/km", "fc_cible": [144, 150]}}}}
    answer = llm.answer(system(CONTEXT, paces) + [{"role": "user", "content": "Quelle allure en footing ?"}])
    assert "4'37\"/km à 5'15\"/km" in answer and "144" in answer
    default = llm.answer(system(CONTEXT) + [{"role": "user", "content": "Ça va ?"}])
    assert "Feu vert" in default and "7,2 h" in default


def test_coach_demo_douleur_exercices_et_proposition():
    pain = {"douleur": {"zones": [{"nom": "Tendon d'Achille", "specialiste": "kinésithérapeute",
                                   "exercices": [{"nom": "Montées sur pointes", "dosage": "3 × 15"}],
                                   "course": "Évite les côtes."}], "signaux_alerte": ["gonflement", "douleur la nuit"]}}
    answer = DemoLLM().answer(system(CONTEXT, pain) + [{"role": "user", "content": "Mal au tendon d'Achille"}])
    assert "diagnostic" in answer and "kinésithérapeute" in answer and "[[PROPOSITION]]" in answer


def test_choix_du_coach(monkeypatch):
    monkeypatch.setenv("RUNLAB_LLM", "demo")
    assert isinstance(get_llm(), DemoLLM) and get_llm().status()["disponible"]


@pytest.fixture
def demo_env(data_dir, tmp_path, monkeypatch):
    import shutil

    import streamlit as st

    import api.main

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    subprocess.run([sys.executable, "scripts/make_run_labels.py"], env={**os.environ, "RUNLAB_DATA_DIR": str(data)},
                   check=True, capture_output=True)
    for key, value in {"RUNLAB_DATA_DIR": str(data), "RUNLAB_DEMO": "1", "RUNLAB_LLM": "demo",
                       "RUNLAB_API_URL": "inprocess", "RUNLAB_AUTO_SYNC": "0", "RUNLAB_WATCH_AUTO": "0"}.items():
        monkeypatch.setenv(key, value)
    importlib.reload(api.main)
    st.cache_data.clear()
    yield data
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)
    st.cache_data.clear()


@pytest.mark.parametrize("page", ["Accueil", "Coach", "Planning", "Prédictions", "Séances"])
def test_tableau_de_bord_en_mode_demo(demo_env, page):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    at.radio(key="page").set_value(page).run()
    assert not at.exception, at.exception
    page_text = " ".join(m.value for m in at.markdown)
    assert "Démo en ligne" in page_text and "synthétiques" in page_text  # le visiteur sait que rien n'est réel
    assert not [b for b in at.button if b.label == "Mettre à jour"]  # pas de synchronisation Garmin en démo


def test_coach_demo_bout_en_bout(demo_env):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    at.radio(key="page").set_value("Coach").run()
    at.chat_input[0].set_value("Je suis fatigué, allège ma prochaine séance").run()
    assert not at.exception, at.exception
    page_text = " ".join(m.value for m in at.markdown)
    assert "règles" in page_text and ("Proposition du coach" in page_text or "D'après ta demande" in page_text)


def test_fichiers_publies_sur_le_space(tmp_path):
    """Le Space reçoit le code utile à la démo, sa fiche et son Dockerfile, jamais de données."""
    sys.path.insert(0, str(ROOT / "deploy" / "huggingface"))
    push_space = importlib.import_module("push_space")
    push_space.stage(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert {"Dockerfile", "README.md", "scripts/demo.py", "dashboard/app.py", "api/main.py"} <= names
    assert not any(n.startswith("data/") or "__pycache__" in n for n in names)
    assert (tmp_path / "README.md").read_text(encoding="utf-8").startswith("---\ntitle: Foulée")
