"""Synchronisation à la demande : progression, verrou, données récentes, erreurs (étapes de démonstration)."""

import importlib
import shutil
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sync_client(data_dir, tmp_path, monkeypatch):
    import api.main

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RUNLAB_SYNC_STEPS", "demo")
    importlib.reload(api.main)
    yield TestClient(api.main.app), api.main
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)


def test_synchronisation_complete(sync_client):
    client, main = sync_client
    before = client.get("/sync/statut").json()
    started = client.post("/sync").json()
    assert started["lancee"] and started["en_cours"]
    assert client.post("/sync").json()["raison"] == "deja_en_cours"  # une seule à la fois
    main.SYNC.wait()
    after = client.get("/sync/statut").json()
    assert not after["en_cours"] and after["erreur"] is None and after["progression"] == 1.0
    assert after["derniere_synchro"] >= before["derniere_synchro"]
    assert not (main.DATA_DIR / "sync" / "verrou").exists()  # le verrou est libéré


def test_pas_de_synchronisation_si_les_donnees_sont_recentes(sync_client):
    client, main = sync_client
    (main.DATA_DIR / "gold" / "daily_features.parquet").touch()  # données de la minute
    answer = client.post("/sync", params={"si_plus_ancienne_que_min": 30}).json()
    assert answer["lancee"] is False and answer["raison"] == "donnees_recentes"


def test_verrou_perime_ignore(sync_client):
    import os

    from api.sync import STALE_LOCK_S

    client, main = sync_client
    lock = main.DATA_DIR / "sync" / "verrou"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("synchronisation interrompue")
    old = time.time() - STALE_LOCK_S - 60
    os.utime(lock, (old, old))
    assert client.get("/sync/statut").json()["en_cours"] is False
    assert client.post("/sync").json()["lancee"] is True  # le verrou périmé est repris
    main.SYNC.wait()


def test_erreur_d_etape_remontee(sync_client, monkeypatch):
    import api.sync

    client, main = sync_client
    monkeypatch.setattr(api.sync, "steps", lambda: [("Étape qui échoue", ["-c", "import sys; sys.exit('boum')"])])
    client.post("/sync")
    main.SYNC.wait()
    status = client.get("/sync/statut").json()
    assert "Étape qui échoue" in status["erreur"] and "boum" in status["erreur"] and not status["en_cours"]


def test_statut_lisible_pendant_les_ecritures(tmp_path):
    """Lectures et écritures simultanées : la lecture ne voit jamais un fichier vide ou partiel."""
    import threading

    from api.sync import SyncManager

    manager = SyncManager(tmp_path)
    stop, errors = threading.Event(), []

    def writer():
        i = 0
        while not stop.is_set():
            manager._save(etape=f"étape {i}", progression=i % 100 / 100, erreur=None)
            i += 1

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(3000):
            saved = manager._read_status()
            if saved and "etape" not in saved:
                errors.append(saved)
    finally:
        stop.set()
        thread.join()
    assert not errors
    assert manager._read_status()["etape"].startswith("étape")
    assert not list(tmp_path.glob("sync/*.tmp"))  # aucun fichier temporaire oublié


def test_statut_illisible_ne_plante_pas(tmp_path):
    from api.sync import SyncManager

    manager = SyncManager(tmp_path)
    manager.dir.mkdir(parents=True)
    manager.status_file.write_text("", encoding="utf-8")  # fichier vide, comme en pleine écriture
    assert manager.status()["en_cours"] is False
