"""Analyse d'une sortie : régularité, dérive cardiaque, zones, efficacité comparée."""

import pandas as pd

from processing.run_analysis import analyse_run, efficiency_history


def run(**kw):
    base = {"activity_id": 9, "start_time": "2026-10-04 10:00", "distance_m": 10000, "duration_s": 3000,
            "moving_duration_s": 3000, "avg_hr": 150, "max_hr": 165, "elevation_gain_m": 0,
            **{f"hr_z{z}_s": 0 for z in range(1, 6)}}
    return pd.Series({**base, **kw})


def laps(paces, hrs):
    return pd.DataFrame({"activity_id": 9, "lap": range(1, len(paces) + 1), "distance_m": 1000.0,
                         "duration_s": paces, "avg_hr": hrs, "max_hr": hrs, "elevation_gain_m": 0, "cadence": 170})


EMPTY = pd.DataFrame(columns=["activity_id", "kind", "date", "efficacite"])


def test_negative_split_et_derive_faible():
    out = analyse_run(run(), laps([310, 305, 300, 300, 295, 295, 290, 290], [145, 146, 147, 147, 148, 148, 149, 149]),
                      "tempo", EMPTY, 193)
    assert out["regularite"]["ecart_s"] < 0 and "négative split" in out["sections"]["allure"][0]
    assert out["derive"]["niveau"] == "faible"


def test_derive_forte_quand_la_fc_s_envole():
    out = analyse_run(run(), laps([300] * 8, [140, 142, 150, 155, 162, 166, 170, 172]), "ef", EMPTY, 193)
    assert out["derive"]["niveau"] == "forte" and out["derive"]["fc_2de_moitie"] > out["derive"]["fc_1re_moitie"]


def test_ef_trop_intense_signalee():
    out = analyse_run(run(hr_z2_s=600, hr_z3_s=1800, hr_z4_s=600), pd.DataFrame(), "ef", EMPTY, 193)
    assert "trop intense" in out["sections"]["zones"][0] and out["tours"] == []


def test_efficacite_comparee_aux_sorties_du_meme_type():
    history = pd.DataFrame({"activity_id": [1, 2, 3, 4], "kind": ["ef", "ef", "ef", "tempo"],
                            "date": pd.to_datetime(["2026-09-01", "2026-09-08", "2026-09-15", "2026-09-20"]),
                            "efficacite": [1.30, 1.30, 1.30, 2.0]})
    out = analyse_run(run(), pd.DataFrame(), "ef", history, 193)  # 10 km en 50 min à 150 bpm : 1,33 m/battement
    assert out["comparaison"]["sorties"] == 3 and out["comparaison"]["variation_pct"] > 2
    assert "plus vite pour le même effort" in out["sections"]["efficacite"][0]


def test_historique_sans_fractionne_ni_sortie_courte():
    runs = pd.DataFrame({"distance_m": [10000, 3000, 8000], "duration_s": [3000, 900, 2400],
                         "moving_duration_s": [3000, 900, 2400], "avg_hr": [150, 150, 165],
                         "elevation_gain_m": [None, 0, 0], "kind": ["ef", "ef", "fractionne"]})
    eff = efficiency_history(runs)["efficacite"]
    assert eff.iloc[0] > 0 and pd.isna(eff.iloc[1]) and pd.isna(eff.iloc[2])


def test_points_d_acces(data_dir, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    import api.main

    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)
    client = TestClient(api.main.app)
    runs = client.get("/courses", params={"limite": 5}).json()["courses"]
    assert len(runs) == 5
    out = client.get("/courses/analyse", params={"activity_id": runs[1]["activity_id"]}).json()
    assert out["activity_id"] == runs[1]["activity_id"] and out["tours"] and out["points"]
    assert client.get("/courses/analyse", params={"activity_id": 123456789}).status_code == 404


def test_le_denivele_est_ramene_sur_le_plat():
    # km 3 avec 40 m de D+ : plus lent en réel, mais même effort sur le plat
    flat = [300] * 8
    real = [300 + (40 * 7.92 / 1000 * 300 if i == 2 else 0) for i in range(8)]
    lap_df = laps(real, [150] * 8).assign(elevation_gain_m=[40 if i == 2 else 0 for i in range(8)])
    out = analyse_run(run(elevation_gain_m=40), lap_df, "ef", EMPTY, 193)
    third = out["tours"][2]
    assert third["allure_s"] > third["allure_plat_s"] == flat[2]
    assert out["regularite"]["variation_pct"] < 0.5  # régulier une fois la côte prise en compte


def test_commentaire_du_coach_garde_en_cache(data_dir, tmp_path, monkeypatch):
    import importlib
    import shutil

    from fastapi.testclient import TestClient

    import api.main

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RUNLAB_LLM", "fake")
    importlib.reload(api.main)
    client = TestClient(api.main.app)
    run_id = client.get("/courses").json()["courses"][0]["activity_id"]
    context = api.main.run_context(run_id)
    assert {"kilometres", "nuit_precedente_et_forme", "ressenti_declare", "sorties_similaires"} <= context.keys()
    assert client.get("/courses/commentaire", params={"activity_id": run_id}).json()["commentaire"] is None
    with client.stream("POST", "/courses/commentaire/flux", json={"activity_id": run_id}) as r:
        body = "".join(r.iter_text())
    assert "Réponse de test" in body and "[[MESURES]]" in body
    saved = client.get("/courses/commentaire", params={"activity_id": run_id}).json()
    assert saved["commentaire"].startswith("Réponse de test") and "[[" not in saved["commentaire"]
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)
