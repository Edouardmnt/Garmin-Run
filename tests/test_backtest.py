"""Backtest des prédictions : pas de fuite de l'avenir, métriques correctes, points d'accès."""

import pandas as pd
import pytest

from processing.backtest import backtest, by_source, metrics, questionnaire_agreement
from processing.performance import collect_performances, hr_speed_vo2max, vo2max_history


def inputs(data_dir):
    import os
    import subprocess
    import sys

    # Le test crée lui-même les étiquettes : il ne doit pas dépendre de l'ordre d'exécution des autres tests
    subprocess.run([sys.executable, "scripts/make_run_labels.py"], env={**os.environ, "RUNLAB_DATA_DIR": str(data_dir)},
                   check=True, capture_output=True)
    acts = pd.read_parquet(data_dir / "silver" / "activities.parquet")
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    labels = pd.read_csv(data_dir / "labels" / "run_labels.csv", sep=";", decimal=",", dtype={"label": str}) \
        if (data_dir / "labels" / "run_labels.csv").exists() else None
    perf = collect_performances(acts, labels)
    physio = {"vo2max_montre": vo2max_history(acts),
              "relation_fc_vitesse": hr_speed_vo2max(acts, labels, float(gold["resting_hr"].median()),
                                                     float(acts["max_hr"].max()))}
    load = gold.assign(date=pd.to_datetime(gold["date"])).set_index("date")["ctl"]
    return perf, physio, load


def test_aucune_fuite_de_l_avenir(data_dir):
    """Modifier tout ce qui se passe à partir du jour d'une performance ne change pas sa prédiction."""
    perf, physio, load = inputs(data_dir)
    results = backtest(perf, physio, load)
    target = results.iloc[len(results) // 2]
    day = target["date"]
    later = perf["date"] >= day
    perf2 = perf.copy()
    perf2.loc[later & (perf2["date"] > day), "vdot"] = 80.0  # avenir absurde
    physio2 = {k: v.assign(vo2max=v["vo2max"].where(v["date"] < day, 90.0)) for k, v in physio.items()}
    load2 = load.where(load.index < day, 999.0)
    results2 = backtest(perf2, physio2, load2)
    same = results2[(results2["date"] == day) & (results2["distance_m"] == target["distance_m"])].iloc[0]
    assert same["application"] == pytest.approx(target["application"])
    assert same["vdot_estime"] == pytest.approx(target["vdot_estime"])


def test_metriques():
    results = pd.DataFrame({"temps_reel_s": [1000.0, 2000.0], "application": [1100.0, 1900.0],
                            "sans_correction": [1000.0, 2000.0], "source": ["course", "course"]})
    m = metrics(results)
    assert m["application"]["mape_pct"] == 7.5 and m["application"]["biais_pct"] == 2.5  # +10 % et -5 %
    assert m["sans_correction"]["mape_pct"] == 0 and m["sans_correction"]["part_a_3pct"] == 100
    assert by_source(results)["course"]["n"] == 2


def test_la_correction_des_questionnaires_ne_vient_que_du_passe(data_dir):
    perf, physio, load = inputs(data_dir)
    first = backtest(perf, physio, load).iloc[0]["date"]
    late = [{"date_sortie": "2099-01-01", "reponses": {"prediction": "Trop optimiste"}}]  # réponse « future »
    results = backtest(perf, physio, load, late)
    assert (results["questionnaires"] == results["sans_correction"]).all()
    early = [{"date_sortie": (first - pd.Timedelta(days=5)).date().isoformat(),
              "reponses": {"prediction": "Trop optimiste"}}]
    row = backtest(perf, physio, load, early).iloc[0]
    assert row["questionnaires"] > row["sans_correction"]


def test_recalibrage_prudent_et_plafonne():
    from processing.backtest import recalibration

    assert recalibration([], []) == 0
    one = recalibration([1100], [1000])  # une seule erreur de -10 % : on n'en corrige qu'un quart
    assert 0 < one < 3
    many = recalibration([1100] * 30, [1000] * 30)
    assert 8 < many < 10  # beaucoup d'erreurs identiques : on corrige presque tout l'écart
    assert recalibration([2000] * 50, [1000] * 50) == 15  # plafond
    assert recalibration([900] * 10, [1000] * 10) < 0  # prédictions trop lentes : on les raccourcit


def test_le_recalibrage_n_apprend_que_du_passe(data_dir):
    """La correction d'une performance ne dépend que des performances antérieures."""
    perf, physio, load = inputs(data_dir)
    results = backtest(perf, physio, load)
    assert results["erreurs_passees"].iloc[0] == 0 and results["erreurs_passees"].is_monotonic_increasing
    cut = len(results) // 2
    day = results.iloc[cut]["date"]
    truncated = backtest(perf[perf["date"] <= day], physio, load)  # on retire toutes les performances suivantes
    assert truncated.iloc[-1]["correction_pct"] == pytest.approx(results.iloc[cut]["correction_pct"])


def test_questionnaires():
    fb = [{"reponses": {"prediction": v}} for v in ["Juste", "Juste", "Trop optimiste", "Je n'avais pas regardé"]]
    assert questionnaire_agreement(fb) == {"reponses": 3, "juste_pct": 67, "trop_optimiste": 1, "trop_pessimiste": 0}
    assert questionnaire_agreement([]) == {"reponses": 0}


def test_point_d_acces(data_dir, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    import api.main

    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)
    body = TestClient(api.main.app).get("/qualite/predictions").json()
    assert body["n"] > 5 and "application" in body["methodes"] and "riegel_derniere" in body["methodes"]
    assert len(body["points"]) == body["n"]


def test_sans_performance_rien_a_tester():
    empty = pd.DataFrame(columns=["date", "distance_m", "time_s", "source", "dplus_m", "vdot"])
    assert backtest(empty, {"vo2max_montre": pd.DataFrame(columns=["date", "vo2max"])}).empty
