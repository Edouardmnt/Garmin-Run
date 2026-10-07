"""Le tableau de bord s'affiche sans erreur sur chaque page (API appelée dans le même processus)."""

import os
import subprocess
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

TITLES = {"Accueil": "Ta journée", "Coach": "Ton coach", "Objectifs": "Mes objectifs", "Ma forme": "Ma forme",
          "Nuits & journées": "Nuits & journées", "Planning": "Mon planning",
          "Prédictions": "Mes prédictions", "Allures": "Mes allures", "Séances": "Mes séances"}


@pytest.fixture(scope="module")
def app_env(data_dir):
    env = {**os.environ, "RUNLAB_DATA_DIR": str(data_dir)}
    subprocess.run([sys.executable, "scripts/make_run_labels.py"], env=env, check=True, capture_output=True)
    mp = pytest.MonkeyPatch()
    mp.setenv("RUNLAB_DATA_DIR", str(data_dir))
    mp.setenv("RUNLAB_API_URL", "inprocess")
    mp.setenv("RUNLAB_LLM", "fake")  # faux modèle : pas d'Ollama dans les tests
    mp.setenv("RUNLAB_AUTO_SYNC", "0")  # pas de synchronisation à l'ouverture, sauf dans le test dédié
    import importlib

    import api.main

    importlib.reload(api.main)
    yield data_dir
    mp.undo()


def open_page(page: str, objective: dict | None = None) -> AppTest:
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)  # chemin relatif à ce fichier de test
    if objective:
        at.session_state["objectif"] = objective
    at.run()
    at.radio(key="page").set_value(page).run()
    return at


@pytest.mark.parametrize("page", TITLES)
def test_chaque_page_s_affiche(app_env, page):
    at = open_page(page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == TITLES[page]


@pytest.fixture
def private_data(app_env, tmp_path, monkeypatch):
    """Copie des données pour les tests qui écrivent (objectifs, questionnaires), puis retour à l'original."""
    import importlib
    import shutil

    import streamlit as st

    import api.main

    data = tmp_path / "data"
    shutil.copytree(app_env, data)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    importlib.reload(api.main)
    st.cache_data.clear()
    yield data
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(app_env))
    importlib.reload(api.main)
    st.cache_data.clear()


def test_creer_un_objectif_puis_suivre_le_planning(private_data):
    """Le vrai parcours : remplir le formulaire d'objectif, cliquer, puis consulter le planning."""
    at = open_page("Objectifs")
    at.text_input[0].set_value("Semi de test")
    at.selectbox[0].set_value("Semi")
    next(b for b in at.button if "Créer" in b.label).click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    goals = __import__("json").loads((private_data / "objectifs.json").read_text(encoding="utf-8"))
    assert goals[0]["nom"] == "Semi de test" and goals[0]["actif"]

    planning = open_page("Planning")
    assert not planning.exception, planning.exception
    weeks = [h for h in planning.header if h.value.startswith("Semaine")]
    assert len(weeks) >= 8  # une section par semaine jusqu'à la course


def test_questionnaire_affiche_puis_enregistre(private_data):
    """Après une sortie récente, l'accueil propose le questionnaire et l'envoi enregistre les réponses."""
    import importlib
    import json

    import streamlit as st

    import api.main

    acts = pd.read_parquet(private_data / "silver" / "activities.parquet")
    last = pd.to_datetime(pd.read_parquet(private_data / "gold" / "daily_features.parquet")["date"]).max()
    run = acts[acts["sport"] == "running"].iloc[-1].copy()
    run["activity_id"], run["start_time"] = 999999, (last - pd.Timedelta(days=1)).replace(hour=8)
    pd.concat([acts, run.to_frame().T.astype(acts.dtypes)], ignore_index=True).to_parquet(
        private_data / "silver" / "activities.parquet", index=False)
    importlib.reload(api.main)
    st.cache_data.clear()

    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    questions = [r for r in at.radio if r.key and r.key.startswith("q_")]
    assert 3 <= len(questions) <= 5
    for radio in questions:
        radio.set_value(radio.options[0])
    next(b for b in at.button if "Envoyer mes" in b.label).click().run()
    assert not at.exception, at.exception
    saved = json.loads((private_data / "feedback" / "feedback.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert saved["activity_id"] == 999999
    # Rafraîchissement automatique : le questionnaire a disparu et le message de remerciement s'affiche
    assert not [r for r in at.radio if r.key and r.key.startswith("q_")]
    assert any("Merci" in m.value for m in at.success)


def test_coach_repond_a_une_question(app_env):
    at = open_page("Coach")
    assert not at.exception, at.exception
    at.chat_input[0].set_value("Quelle allure pour mon footing ?").run()
    assert not at.exception, at.exception
    assert any("Réponse de test" in m.value for m in at.markdown)  # réponse affichée mot à mot
    note = " ".join(m.value for m in at.markdown)
    assert "Allures" in note and "jetons/s" in note  # données consultées et vitesse mesurée
    assert len(at.session_state["coach_messages"]) == 2
    next(b for b in at.button if "bilan" in b.label).click().run()
    assert not at.exception and "bilan" in at.session_state


def test_synchronisation_a_l_ouverture(private_data, monkeypatch):
    """Données anciennes : la synchronisation démarre, la page reste utilisable, puis se recharge."""
    import html
    import os
    import time

    monkeypatch.setenv("RUNLAB_AUTO_SYNC", "1")
    monkeypatch.setenv("RUNLAB_SYNC_STEPS", "demo")
    gold = private_data / "gold" / "daily_features.parquet"
    old = gold.stat().st_mtime - 3 * 3600
    os.utime(gold, (old, old))  # données vieilles de 3 heures
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.radio(key="page").value == "Accueil" and at.title[0].value == "Ta journée"  # rien n'est bloqué
    assert at.session_state["synchro_suivie"] is True
    for _ in range(90):  # le rafraîchissement automatique est simulé par des relances
        time.sleep(1)
        at.run()
        if any("synchronisées à l'instant" in html.unescape(m.value) for m in at.markdown):
            break
    assert not at.exception, at.exception
    assert gold.stat().st_mtime > old  # la synchronisation a bien tourné
    assert any("synchronisées à l'instant" in html.unescape(m.value) for m in at.markdown)  # texte tel qu'affiché
