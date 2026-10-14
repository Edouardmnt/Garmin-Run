"""Météo des sorties : collecte Open-Meteo (réseau simulé), couche silver, effet de la chaleur, API, backtest."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ingestion import weather_export as wx
from processing.heat import (
    PRIOR_PCT_PER_C,
    attach_weather,
    heat_penalty_pct,
    learn_sensitivity,
    normalize,
    race_weather_penalty,
    wbgt_simplified,
)

FIXTURE = Path(__file__).parent / "fixtures" / "open_meteo_archive_paris_2026-09-01.json"  # vraie réponse


# --- Indice de chaleur et pénalité -------------------------------------------------------------------

def test_wbgt_et_penalite():
    assert wbgt_simplified(25, 60) == pytest.approx(25.6, abs=0.3)  # table du Bureau of Meteorology : ~25,5
    assert wbgt_simplified(10, 50) < wbgt_simplified(10, 90) < wbgt_simplified(20, 90)  # humidité et chaleur
    assert heat_penalty_pct(8.0) == 0 and heat_penalty_pct(None) == 0 and heat_penalty_pct(float("nan")) == 0
    assert heat_penalty_pct(20.0, 0.3) == pytest.approx(3.0)
    hot, mild = race_weather_penalty(30, 60, 0.3), race_weather_penalty(None, None, 0.3)
    assert hot["penalite_pct"] > mild["penalite_pct"] > 0 and mild["par_defaut"] and not hot["par_defaut"]


def steady_runs(sensitivity_pct: float, noise: float, n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wbgt = rng.uniform(8, 26, n)
    vo2 = 50 * (1 - np.clip(wbgt - 10, 0, None) * sensitivity_pct / 100) * (1 + rng.normal(0, noise, n))
    return pd.DataFrame({"date": pd.date_range("2026-03-01", periods=n, freq="D"), "vo2max": vo2, "wbgt_c": wbgt})


def test_sensibilite_personnelle_retrouvee_quand_le_signal_est_net():
    learned = learn_sensitivity(steady_runs(0.6, 0.005))
    assert learned["origine"] == "personnelle" and learned["pct_par_degre"] == pytest.approx(0.6, abs=0.05)


def test_mesure_bruitee_proche_de_la_reference():
    learned = learn_sensitivity(steady_runs(0.0, 0.10, n=20))
    assert abs(learned["pct_par_degre"] - PRIOR_PCT_PER_C) < abs(learned["mesure_brute"] - PRIOR_PCT_PER_C)
    assert learned["poids_mesure_pct"] < 50


def test_sans_meteo_la_reference_s_applique():
    no_weather = steady_runs(0.6, 0.0).assign(wbgt_c=np.nan)
    assert learn_sensitivity(no_weather)["origine"] == "reference"
    assert learn_sensitivity(no_weather.drop(columns="wbgt_c"))["pct_par_degre"] == PRIOR_PCT_PER_C


def test_une_performance_courue_a_la_chaleur_vaut_mieux_que_son_chrono():
    perf = pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-03-01"]), "distance_m": [10000.0, 10000.0],
                         "time_s": [3000.0, 3000.0], "source": "course", "dplus_m": 0, "activity_id": [1, 2],
                         "vdot": [40.0, 40.0]})
    weather = pd.DataFrame({"activity_id": [1, 2], "wbgt_c": [25.0, 8.0]})
    steady = pd.DataFrame({"date": pd.to_datetime(["2026-07-02"]), "vo2max": [50.0], "activity_id": [1]})
    perf, steady = attach_weather(perf, weather), attach_weather(steady, weather)
    fixed_perf, fixed_steady = normalize(perf, steady, 0.3)
    assert fixed_perf.loc[0, "time_s"] == pytest.approx(3000 / 1.045) and fixed_perf.loc[1, "time_s"] == 3000
    assert fixed_perf.loc[0, "vdot"] > fixed_perf.loc[1, "vdot"]
    assert fixed_steady.loc[0, "vo2max"] == pytest.approx(50 * 1.045)
    assert fixed_perf.loc[0, "time_s_mesure"] == 3000  # le chrono réel est gardé


# --- Collecte Open-Meteo (réseau simulé avec une vraie réponse enregistrée) ---------------------------

def activity(i, start, lat=48.82, duration=3000):
    return {"activityId": i, "startTimeLocal": start, "duration": duration, "startLatitude": lat,
            "startLongitude": 2.29}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_sorties_a_traiter(tmp_path):
    (tmp_path / "weather").mkdir()
    (tmp_path / "weather" / "3.json").write_text("{}")
    acts = [activity(1, "2026-09-01 18:05:00"), {**activity(2, "2026-09-01 12:00:00"), "startLatitude": None},
            activity(3, "2026-09-01 18:05:00"), activity(4, "2026-09-10 23:30:00")]
    todo = wx.pending(acts, tmp_path, datetime(2026, 9, 10, 23, 45))
    assert [a["activityId"] for a in todo] == [1]  # tapis (sans position), déjà fait, pas encore finie : écartées


def test_une_requete_par_lieu_et_par_source():
    acts = [activity(1, "2026-08-01 08:00:00"), activity(2, "2026-08-20 08:00:00"),
            activity(3, "2026-09-08 08:00:00"), activity(4, "2026-08-02 08:00:00", lat=45.76)]
    plan = wx.plan_requests(acts, date(2026, 9, 10))
    archive = [r for r in plan if r["source"] == "archive" and r["params"]["latitude"] == 48.8]
    assert len(plan) == 3 and len(archive[0]["activities"]) == 2
    assert archive[0]["params"]["start_date"] == "2026-08-01" and archive[0]["params"]["end_date"] == "2026-08-20"
    recent = next(r for r in plan if r["source"] == "prevision")
    assert recent["url"] == wx.FORECAST_URL and recent["params"]["past_days"] == 3


def test_collecte_avec_une_vraie_reponse(tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return FakeResponse(payload)

    run = activity(7, "2026-09-01 18:05:00", duration=50 * 60)
    result = wx.fetch_all([run], tmp_path, date(2026, 9, 20), get=fake_get)
    saved = json.loads((tmp_path / "weather" / "7.json").read_text(encoding="utf-8"))
    assert result == {"enregistrees": 1, "echecs": 0} and calls[0][0] == wx.ARCHIVE_URL
    assert [h["time"] for h in saved["hourly"]] == ["2026-09-01T18:00"]  # heure de départ ; 19 h est après la fin
    assert saved["hourly"][0]["temperature_2m"] == 23.2 and saved["source"] == "open-meteo:archive"


def test_open_meteo_injoignable_ne_bloque_rien(tmp_path):
    def broken(url, params, timeout):
        return FakeResponse({}, status=503)

    result = wx.fetch_all([activity(8, "2026-09-01 18:05:00")], tmp_path, date(2026, 9, 20), get=broken)
    assert result == {"enregistrees": 0, "echecs": 1} and not list((tmp_path / "weather").glob("*.json"))


def test_prevision_a_une_heure_donnee():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    got = wx.forecast(48.82, 2.29, date(2026, 9, 1), 9, get=lambda url, params, timeout: FakeResponse(payload))
    assert got["temperature_c"] == 16.0 and got["humidite_pct"] == 82


# --- Couche silver, API et backtest sur les données synthétiques -------------------------------------

def test_table_silver_meteo(data_dir):
    weather = pd.read_parquet(data_dir / "silver" / "activity_weather.parquet")
    activities = pd.read_parquet(data_dir / "silver" / "activities.parquet")
    outdoor = activities[(activities["garmin_type"] == "running")]
    assert len(weather) == len(outdoor) and weather["wbgt_c"].between(0, 35).all()
    assert outdoor["start_lat"].notna().all()


@pytest.fixture
def api(data_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    subprocess.run([sys.executable, "scripts/make_run_labels.py"], env={**os.environ, "RUNLAB_DATA_DIR": str(data)},
                   check=True, capture_output=True)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    importlib.reload(api.main)
    yield api.main, TestClient(api.main.app), data
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)


def test_predictions_selon_la_meteo_du_jour_j(api):
    _, client, _ = api
    params = {"distance": "10k", "ajuster_au_jour": False}
    mild = client.get("/predictions", params=params).json()
    cool = client.get("/predictions", params={**params, "temperature_c": 4, "humidite_pct": 60}).json()
    hot = client.get("/predictions", params={**params, "temperature_c": 29, "humidite_pct": 70}).json()
    times = [x["predictions"]["10k"]["temps_base_s"] for x in (cool, mild, hot)]
    assert times[0] < times[1] < times[2] and cool["meteo"]["penalite_pct"] == 0
    assert mild["meteo"]["par_defaut"] and any("Météo" in w for w in hot["avertissements"])
    assert "pct_par_degre" in hot["sensibilite_chaleur"]


def test_meteo_dans_l_analyse_de_sortie(api):
    _, client, _ = api
    run_id = next(r["activity_id"] for r in client.get("/courses").json()["courses"])
    analysis = client.get("/courses/analyse", params={"activity_id": run_id}).json()
    if analysis["meteo"] is not None:  # la dernière sortie peut être sur tapis (pas de météo)
        assert "°C" in analysis["meteo"]["resume"] and analysis["meteo"]["wbgt_c"] > 0


def test_prevision_du_jour_de_course(api, monkeypatch):
    main, client, data = api
    calls = []

    def fake_forecast(lat, lon, day, hour):
        calls.append((round(lat, 1), round(lon, 1), day, hour))
        return {"temperature_c": 24.0, "humidite_pct": 55, "vent_kmh": 8.0, "pluie_mm": 0.0, "heure": f"{day}T09:00"}

    monkeypatch.setattr("ingestion.weather_export.forecast", fake_forecast)
    answer = client.get("/meteo/prevision", params={"date_course": "2026-10-05"}).json()
    assert answer["temperature_c"] == 24.0 and answer["penalite_pct"] > 0 and calls[0][:2] == (48.8, 2.4)
    client.get("/meteo/prevision", params={"date_course": "2026-10-05"})
    assert len(calls) == 1  # prévision gardée en cache quelques heures
    assert client.get("/meteo/prevision", params={"date_course": "2026-11-30"}).status_code == 422


def test_backtest_compare_avec_et_sans_meteo_sans_fuite(api):
    from processing.backtest import backtest, estimate_before

    main, _, _ = api
    gold, perf, physio = main.estimation_inputs()
    results = backtest(perf, physio, main.chronic_load(gold))
    assert {"sans_meteo", "chaleur_pct", "wbgt_c"} <= set(results.columns) and results["sans_meteo"].notna().all()

    # Aucune fuite : changer la météo des sorties FUTURES ne change pas l'estimation d'une date passée
    day = results["date"].iloc[3]
    before = estimate_before(day, perf, physio, None)
    hotter = {k: v.copy() for k, v in physio.items()}
    future = hotter["relation_fc_vitesse"]["date"] >= day
    hotter["relation_fc_vitesse"].loc[future, "wbgt_c"] = 35.0
    future_perf = perf.copy()
    future_perf.loc[future_perf["date"] >= day, "wbgt_c"] = 35.0
    after = estimate_before(day, future_perf, hotter, None)
    assert after["vdot"] == before["vdot"] and after["sensibilite_chaleur"] == before["sensibilite_chaleur"]
