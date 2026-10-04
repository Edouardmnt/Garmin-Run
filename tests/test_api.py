"""Tests de l'API sur les données synthétiques du pipeline."""

import importlib
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(data_dir, monkeypatch_module):
    env = {**os.environ, "RUNLAB_DATA_DIR": str(data_dir)}
    subprocess.run([sys.executable, "scripts/make_run_labels.py"], env=env, check=True, capture_output=True)
    monkeypatch_module.setenv("RUNLAB_DATA_DIR", str(data_dir))
    import api.main

    importlib.reload(api.main)  # relit RUNLAB_DATA_DIR
    return TestClient(api.main.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_forme(client):
    body = client.get("/forme").json()
    assert {"hrv_ecart_pct", "fraicheur_tsb", "ajustement_chrono_pct", "explications"} <= body.keys()
    assert -1 <= body["ajustement_chrono_pct"] <= 3


def test_predictions_toutes_distances(client):
    body = client.get("/predictions").json()
    assert set(body["predictions"]) == {"5k", "10k", "semi", "marathon"}
    times = [body["predictions"][d]["temps_base_s"] for d in ("5k", "10k", "semi", "marathon")]
    assert times == sorted(times)  # plus c'est long, plus c'est lent


def test_predictions_sans_ajustement(client):
    p = client.get("/predictions", params={"distance": "10k", "ajuster_au_jour": False}).json()["predictions"]["10k"]
    assert p["temps_base_s"] == p["temps_ajuste_s"]


def test_distance_invalide_refusee(client):
    assert client.get("/predictions", params={"distance": "15k"}).status_code == 422


def test_allures_et_seances(client):
    body = client.get("/allures", params={"fenetre_jours": 180}).json()
    assert set(body["zones"]) == {"ef", "tempo", "fractionne"}
    assert all(z["recommandation"]["allure_rapide"] for z in body["zones"].values())
    assert len(client.get("/seances", params={"limite": 3}).json()["seances"]) == 3
