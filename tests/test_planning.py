"""Tests du planning et des analyses rédigées."""

from datetime import date, timedelta

import pandas as pd

from processing.insights import build_analysis, readiness
from processing.planning import HARD, build_plan

PACES = {"ef": {"rapide_s": 330, "lente_s": 360, "fc_cible": [130, 145]},
         "tempo": {"rapide_s": 285, "lente_s": 295, "fc_cible": [162, 172]},
         "fractionne": {"rapide_s": 250, "lente_s": 260, "fc_cible": None}}
TODAY = date(2026, 10, 5)  # un lundi


def plan(**kwargs):
    args = {"today": TODAY, "distance": "10k", "race_date": date(2026, 11, 29), "sessions_per_week": 4,
            "tennis_days": [1, 3], "long_day": 6, "base_weekly_km": 25, "paces": PACES, "race_pace_s": 280}
    return build_plan(**{**args, **kwargs})


def all_sessions(p):
    return [s for w in p["semaines"] for s in w["seances"]]


def test_phases_dans_l_ordre_et_course_le_jour_j():
    p = plan()
    phases = [w["phase"] for w in p["semaines"]]
    assert phases[-1] == "Semaine de course" and phases[-2] == "Affûtage"
    assert phases.index("Spécifique") > 0  # on commence par développer
    race = [s for s in all_sessions(p) if s["type"] == "course"]
    assert len(race) == 1 and race[0]["date"] == "2026-11-29"


def test_aucune_seance_les_jours_de_tennis_ni_deux_seances_dures_de_suite():
    p = plan()
    sessions = all_sessions(p)
    assert all(date.fromisoformat(s["date"]).weekday() not in (1, 3) for s in sessions)
    hard_dates = sorted(date.fromisoformat(s["date"]) for s in sessions if s["type"] in HARD)
    assert all((b - a).days > 1 for a, b in zip(hard_dates, hard_dates[1:]))


def test_affutage_reduit_le_volume():
    weeks = plan()["semaines"]
    peak = max(w["volume_km"] for w in weeks[:-2])
    assert weeks[-2]["volume_km"] < peak


def test_seance_du_jour_allegee_si_la_forme_est_mauvaise():
    # Aujourd'hui lundi, sans tennis et sans autre sortie : la première séance dure tombe aujourd'hui ou plus tard
    p = plan(tennis_days=[], sessions_per_week=3, verdict="rouge", today=date(2026, 10, 6))
    today = [s for s in all_sessions(p) if s["date"] == "2026-10-06"]
    assert not today or today[0]["type"] == "ef"


def test_sans_date_de_course_deux_semaines():
    p = plan(race_date=None)
    assert len(p["semaines"]) == 2 and "Aucune date de course" in p["notes"][0]


def test_verdict_du_jour():
    assert readiness({"hrv_ecart_pct": -25, "sommeil_h": 5.5, "fraicheur_relative": -0.5})["niveau"] == "rouge"
    assert readiness({"hrv_ecart_pct": 6, "sommeil_h": 8, "fraicheur_relative": 0.2})["niveau"] == "vert"


def test_analyses_redigees(data_dir):
    gold = pd.read_parquet(data_dir / "gold" / "daily_features.parquet")
    sports = [c.removeprefix("load_") for c in gold.columns if c.startswith("load_") and c != "load_total"]
    analysis = build_analysis(gold, sports, {"hrv_ecart_pct": 0, "sommeil_h": 7, "fraicheur_relative": 0})
    assert {"verdict", "charge", "forme", "recuperation", "sommeil"} <= analysis.keys()
    assert all(analysis[k]["points"] for k in ("charge", "forme", "recuperation", "sommeil"))
    assert "ACWR" in " ".join(analysis["charge"]["points"])


def test_planning_et_analyse_par_l_api(data_dir, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    import api.main

    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)
    client = TestClient(api.main.app)
    last = pd.to_datetime(pd.read_parquet(data_dir / "gold" / "daily_features.parquet")["date"]).max().date()
    race = (last + timedelta(days=50)).isoformat()
    body = client.get("/planning", params={"distance": "semi", "date_course": race, "jours_tennis": "1,3"}).json()
    assert body["semaines"][-1]["phase"] == "Semaine de course"
    assert client.get("/planning", params={"jours_tennis": "lundi"}).status_code == 422
    assert client.get("/analyse").json()["verdict"]["niveau"] in {"vert", "ambre", "rouge"}
