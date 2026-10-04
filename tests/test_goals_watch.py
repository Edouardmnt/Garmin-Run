"""Tests des objectifs et de l'envoi des séances à la montre (avec un faux client Garmin)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from processing.goals import active_goal, create_goal, delete_goal, load_goals, parse_duration, set_active, validate_goal
from processing.planning import build_plan
from processing.watch import fingerprint, send_session, to_garmin_workout

TODAY = date(2026, 10, 5)
PACES = {"ef": {"rapide_s": 330, "lente_s": 360, "fc_cible": [130, 145]},
         "tempo": {"rapide_s": 285, "lente_s": 295}, "fractionne": {"rapide_s": 250, "lente_s": 260}}


class FakeGarmin:
    """Faux client Garmin : enregistre les appels au lieu de contacter Garmin Connect."""

    def __init__(self):
        self.uploaded, self.scheduled, self.deleted = [], [], []

    def upload_workout(self, workout):
        self.uploaded.append(workout)
        return {"workoutId": 1000 + len(self.uploaded)}

    def schedule_workout(self, workout_id, date_str):
        self.scheduled.append((workout_id, date_str))

    def delete_workout(self, workout_id):
        self.deleted.append(workout_id)


def sessions_of(kind):
    plan = build_plan(TODAY, "10k", date(2026, 11, 29), 4, [1, 3], 6, 25, PACES, 280)
    return [s for w in plan["semaines"] for s in w["seances"] if s["type"] == kind]


@pytest.mark.parametrize("text,seconds", [("47:30", 2850), ("1:45:00", 6300), ("", None), (None, None)])
def test_temps_vise(text, seconds):
    assert parse_duration(text) == seconds


def test_validation_d_un_objectif():
    good = {"nom": "10 km de Paris", "distance": "10k", "date_course": "2026-11-29", "temps_vise": "47:30"}
    assert validate_goal(good, TODAY) == {}
    bad = validate_goal({**good, "nom": " ", "distance": "15k", "date_course": "2026-01-01", "temps_vise": "vite"}, TODAY)
    assert set(bad) == {"nom", "distance", "date_course", "temps_vise"}


def test_un_seul_objectif_actif(tmp_path):
    a = create_goal(tmp_path, {"nom": "A", "distance": "10k", "date_course": "2026-11-29"})
    b = create_goal(tmp_path, {"nom": "B", "distance": "semi", "date_course": "2027-03-01"})
    assert [g["actif"] for g in load_goals(tmp_path)] == [False, True]
    set_active(tmp_path, a["id"])
    assert active_goal(tmp_path, TODAY)["nom"] == "A"
    assert active_goal(tmp_path, date(2026, 12, 1)) is None  # course passée
    assert delete_goal(tmp_path, b["id"]) and len(load_goals(tmp_path)) == 1


def test_fractionne_converti_en_entrainement_garmin():
    workout = to_garmin_workout(sessions_of("fractionne")[0])
    steps = workout["workoutSegments"][0]["workoutSteps"]
    assert [s["stepType"]["stepTypeKey"] for s in steps] == ["warmup", "repeat", "cooldown"]
    repeat = steps[1]
    assert repeat["type"] == "RepeatGroupDTO" and repeat["numberOfIterations"] >= 5
    effort, recovery = repeat["workoutSteps"]
    assert effort["endCondition"]["conditionTypeKey"] == "distance" and effort["endConditionValue"] == 1000
    assert effort["targetType"]["workoutTargetTypeKey"] == "pace.zone"
    # Cible en m/s : la limite basse correspond à l'allure la plus lente (260 s/km)
    assert effort["targetValueOne"] == pytest.approx(1000 / 260, abs=1e-3)
    assert effort["targetValueOne"] < effort["targetValueTwo"]
    assert recovery["stepType"]["stepTypeKey"] == "recovery"
    orders = [steps[0]["stepOrder"], repeat["stepOrder"], effort["stepOrder"], recovery["stepOrder"], steps[2]["stepOrder"]]
    assert sorted(orders) == list(range(1, 6))  # numérotation unique, comme l'attend Garmin


def test_footing_converti_en_une_seule_etape():
    steps = to_garmin_workout(sessions_of("ef")[0])["workoutSegments"][0]["workoutSteps"]
    assert len(steps) == 1 and steps[0]["endCondition"]["conditionTypeKey"] == "distance"


def test_envoi_sans_doublon_et_remplacement(tmp_path):
    session = sessions_of("tempo")[0]
    client = FakeGarmin()
    assert send_session(client, session, tmp_path)["statut"] == "envoyee"
    assert client.scheduled == [(1001, session["date"])]
    assert send_session(client, session, tmp_path)["statut"] == "deja_envoyee"
    assert len(client.uploaded) == 1  # rien n'est renvoyé
    changed = {**session, "titre": "Footing facile (séance adaptée)"}
    assert fingerprint(changed) != fingerprint(session)
    assert send_session(client, changed, tmp_path)["statut"] == "remplacee"
    assert client.deleted == [1001] and len(client.uploaded) == 2


def test_objectifs_planning_et_montre_par_l_api(data_dir, tmp_path, monkeypatch):
    import importlib
    import shutil

    from fastapi.testclient import TestClient

    import api.main
    import ingestion.garmin_export

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    importlib.reload(api.main)
    client = TestClient(api.main.app)
    last = pd.to_datetime(pd.read_parquet(data / "gold" / "daily_features.parquet")["date"]).max().date()
    race = (last + timedelta(days=50)).isoformat()

    assert client.post("/objectifs", json={"objectif": {"nom": "x", "distance": "15k", "date_course": race}}).status_code == 422
    created = client.post("/objectifs", json={"objectif": {"nom": "Semi de test", "distance": "semi", "date_course": race,
                                                           "temps_vise": "1:45:00", "jours_tennis": [1, 3]}}).json()
    goal = client.get("/objectifs").json()["objectifs"][0]
    assert goal["nom"] == "Semi de test" and goal["jours_restants"] == 50 and goal["temps_vise"] == "1h45'00\""
    assert goal["statut"] in {"dans les temps", "proche", "en retard"}
    evolution = client.get(f"/objectifs/{created['objectif']['id']}/suivi", params={"semaines": 6}).json()["evolution"]
    assert len(evolution) == 6

    plan = client.get("/planning/actif").json()
    assert plan["objectif_actif"]["nom"] == "Semi de test" and plan["semaines"][-1]["phase"] == "Semaine de course"
    assert all("etapes" in s for w in plan["semaines"] for s in w["seances"])

    # 6 sorties par semaine, sans tennis : une séance est prévue le jour de référence
    client.post("/objectifs", json={"objectif": {"nom": "10 km", "distance": "10k", "date_course": race,
                                                 "seances_par_semaine": 6}})
    fake = FakeGarmin()
    monkeypatch.setattr(ingestion.garmin_export, "connect", lambda: fake)
    monkeypatch.setattr(api.main, "today_date", lambda: last)  # « aujourd'hui » = dernier jour des données de test
    today = client.get("/montre/seance-du-jour").json()
    assert today["seance"] is not None and today["entrainement_garmin"]["workoutSegments"]
    result = client.post("/montre/envoyer").json()
    assert result["statut"] == "envoyee" and fake.scheduled == [(1001, today["seance"]["date"])]
    assert client.post("/montre/envoyer").json()["statut"] == "deja_envoyee"
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)


# --- Distances libres -----------------------------------------------------------------------------

def test_objectif_de_distance_libre(tmp_path):
    from processing.goals import goal_km, goal_label

    assert validate_goal({"nom": "Foulées de Vincennes", "distance": None, "distance_km": 15, "date_course": "2026-11-29"},
                         TODAY) == {}
    assert "distance" in validate_goal({"nom": "x", "distance_km": 150, "date_course": "2026-11-29"}, TODAY)
    goal = create_goal(tmp_path, {"nom": "Foulées", "distance_km": 15, "date_course": "2026-11-29"})
    assert goal_km(goal) == 15 and goal_label(goal) == "15 km" and goal["distance"] is None
    semi = create_goal(tmp_path, {"nom": "Semi", "distance": "semi", "date_course": "2026-11-29"})
    assert goal_km(semi) == 21.0975 and goal_label(semi) == "semi-marathon"


def test_planning_et_prediction_d_une_distance_libre(data_dir, tmp_path, monkeypatch):
    import importlib
    import shutil

    from fastapi.testclient import TestClient

    import api.main

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    importlib.reload(api.main)
    client = TestClient(api.main.app)
    last = pd.to_datetime(pd.read_parquet(data / "gold" / "daily_features.parquet")["date"]).max().date()
    race = (last + timedelta(days=60)).isoformat()
    assert client.post("/objectifs", json={"objectif": {"nom": "15 km de test", "distance_km": 15,
                                                        "date_course": race}}).status_code == 200
    plan = client.get("/planning/actif").json()
    assert plan["objectif"]["libelle"] == "15 km" and plan["objectif"]["distance_km"] == 15
    race_day = [s for w in plan["semaines"] for s in w["seances"] if s["type"] == "course"][0]
    assert race_day["distance_km"] == 15 and race_day["titre"] == "Course : 15 km"
    preds = client.get("/predictions", params={"distance_km": 15, "ajuster_au_jour": False}).json()["predictions"]
    ten = client.get("/predictions", params={"distance": "10k", "ajuster_au_jour": False}).json()["predictions"]
    assert preds["personnalisee"]["temps_base_s"] > ten["10k"]["temps_base_s"]
    goal = client.get("/objectifs").json()["objectifs"][0]
    assert goal["libelle"] == "15 km"
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)
