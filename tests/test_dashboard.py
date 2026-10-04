"""Le tableau de bord s'affiche sans erreur sur chaque page (API appelée dans le même processus)."""

import os
import subprocess
import sys
from datetime import timedelta

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

TITLES = {"Accueil": "Ta journée", "Ma forme": "Ma forme", "Nuits & journées": "Nuits & journées",
          "Planning": "Mon planning",
          "Prédictions": "Mes prédictions", "Allures": "Mes allures", "Séances": "Mes séances"}


@pytest.fixture(scope="module")
def app_env(data_dir):
    env = {**os.environ, "RUNLAB_DATA_DIR": str(data_dir)}
    subprocess.run([sys.executable, "scripts/make_run_labels.py"], env=env, check=True, capture_output=True)
    mp = pytest.MonkeyPatch()
    mp.setenv("RUNLAB_DATA_DIR", str(data_dir))
    mp.setenv("RUNLAB_API_URL", "inprocess")
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


def test_planning_avec_objectif(app_env):
    last = pd.to_datetime(pd.read_parquet(app_env / "gold" / "daily_features.parquet")["date"]).max().date()
    objective = {"distance": "semi", "date_course": (last + timedelta(days=60)).isoformat(), "denivele_m": 150,
                 "seances_par_semaine": 4, "jours_tennis": "1,3", "jour_sortie_longue": 6}
    at = open_page("Planning", objective)
    assert not at.exception, at.exception
    weeks = [h for h in at.header if h.value.startswith("Semaine")]
    assert len(weeks) >= 8  # une section par semaine jusqu'à la course


def test_bouton_construire_mon_planning(app_env):
    """Le vrai parcours utilisateur : remplir le formulaire et cliquer sur le bouton."""
    at = open_page("Planning")
    next(b for b in at.button if "Construire" in b.label).click().run()
    assert not at.exception, at.exception
    assert at.session_state["objectif"]["distance"] == "10k"
    assert any(h.value.startswith("Semaine") for h in at.header)


def test_questionnaire_affiche_puis_enregistre(app_env, tmp_path, monkeypatch):
    """Après une sortie récente, l'accueil propose le questionnaire et l'envoi enregistre les réponses."""
    import importlib
    import json
    import shutil

    import api.main

    data = tmp_path / "data"
    shutil.copytree(app_env, data)
    acts = pd.read_parquet(data / "silver" / "activities.parquet")
    last = pd.to_datetime(pd.read_parquet(data / "gold" / "daily_features.parquet")["date"]).max()
    run = acts[acts["sport"] == "running"].iloc[-1].copy()
    run["activity_id"], run["start_time"] = 999999, (last - pd.Timedelta(days=1)).replace(hour=8)
    pd.concat([acts, run.to_frame().T.astype(acts.dtypes)], ignore_index=True).to_parquet(
        data / "silver" / "activities.parquet", index=False)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    importlib.reload(api.main)
    import streamlit as st

    st.cache_data.clear()  # les réponses de l'API des tests précédents sont en cache dans ce processus

    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    questions = [r for r in at.radio if r.key and r.key.startswith("q_")]
    assert 3 <= len(questions) <= 5
    for radio in questions:
        radio.set_value(radio.options[0])
    next(b for b in at.button if "Envoyer" in b.label).click().run()
    assert not at.exception, at.exception
    saved = json.loads((data / "feedback" / "feedback.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert saved["activity_id"] == 999999
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(app_env))
    importlib.reload(api.main)
    st.cache_data.clear()
