"""Tests des calculs de performance, vérifiés contre les tables publiées de Jack Daniels."""

import pandas as pd
import pytest

from processing.performance import (
    DISTANCES_M,
    TRAINING_ZONES,
    collect_performances,
    day_adjustment,
    estimate_vdot,
    flat_equivalent_m,
    garmin_predictions,
    hr_speed_vo2max,
    pace_at_fraction,
    personal_training_paces,
    predict_time_s,
    race_effort,
    recent_runs,
    vdot,
    vo2_at_speed,
)


def test_vdot_conforme_aux_tables_de_daniels():
    # Table de Daniels : un 5 km en 19'57" correspond à un VDOT de 50
    assert vdot(5000, 19 * 60 + 57) == pytest.approx(50, abs=0.2)


DANIELS_VDOT_50 = [("10k", 41 * 60 + 21), ("semi", 3600 + 31 * 60 + 35), ("marathon", 3 * 3600 + 10 * 60 + 49)]


@pytest.mark.parametrize("distance,expected_s", DANIELS_VDOT_50)
def test_temps_predits_conformes_aux_tables(distance, expected_s):
    assert predict_time_s(50, DISTANCES_M[distance]) == pytest.approx(expected_s, abs=15)


def test_prediction_coherente_avec_la_performance_d_origine():
    v = vdot(10000, 45 * 60)
    assert predict_time_s(v, 10000) == pytest.approx(45 * 60, abs=1)


def test_allures_ordonnees_de_la_plus_lente_a_la_plus_rapide():
    slowest = [pace_at_fraction(50, low) for low, _, _ in TRAINING_ZONES.values()]
    assert slowest == sorted(slowest, reverse=True)  # EF > marathon > seuil > fractionné > vitesse
    # Allure seuil de la table de Daniels pour un VDOT de 50 : 4'15"/km (borne rapide de la fourchette seuil)
    assert pace_at_fraction(50, 0.88) == pytest.approx(4 * 60 + 15, abs=3)


def test_ajustement_du_jour_borne_et_explique():
    adj, reasons = day_adjustment({"hrv_ecart_pct": -35, "sommeil_h": 4.5, "fraicheur_relative": -0.8})
    assert adj == pytest.approx(0.03)  # plafond de +3 %, même quand tout va mal
    assert len(reasons) == 3
    adj, reasons = day_adjustment({"hrv_ecart_pct": 2, "sommeil_h": 7.5, "fraicheur_relative": 0.0})
    assert adj == 0 and "normale" in reasons[0]


# --- Estimation de la forme ----------------------------------------------------------------------

TODAY = pd.Timestamp("2026-10-01").date()


def activities(*rows):
    cols = ["activity_id", "sport", "start_time", "distance_m", "duration_s", "vo2max", "fastest_5k_s", "fastest_10k_s"]
    return pd.DataFrame(rows, columns=cols)


def labels(mapping):
    return pd.DataFrame({"activity_id": list(mapping), "label": list(mapping.values()), "suggestion": "ef"})


def watch_vo2(acts):
    v = acts.loc[acts["vo2max"].notna(), ["start_time", "vo2max"]]
    return pd.DataFrame({"date": pd.to_datetime(v["start_time"]).dt.normalize(), "vo2max": v["vo2max"]})


def test_les_footings_ne_comptent_pas_mais_les_courses_si():
    acts = activities(
        (1, "running", "2026-09-20 08:00", 10000, 3600, None, 1700, 3550),  # footing EF : 5 km en 28'20"
        (2, "running", "2026-09-27 09:00", 10000, 2850, None, 1410, 2850),  # course : 10 km en 47'30"
    )
    perf = collect_performances(acts, labels({1: "ef", 2: "course"}))
    assert set(perf["source"]) == {"course"}  # rien n'est tiré du footing


def test_calibrage_vo2max_sur_les_courses():
    # VO2 max montre 50, mais la course réelle correspond à un VDOT plus bas : le rapport corrige la formule
    race_vdot = vdot(10000, 2850)
    acts = activities(
        (1, "running", "2026-06-01 08:00", 8000, 2900, 50.0, None, None),
        (2, "running", "2026-06-07 09:00", 10000, 2850, None, None, None),
        (3, "running", "2026-09-28 08:00", 8000, 2900, 51.0, None, None),
    )
    perf = collect_performances(acts, labels({2: "course"}))
    est = estimate_vdot(perf, {"vo2max_montre": watch_vo2(acts)}, TODAY)
    watch = est["composantes"]["vo2max_montre"]
    assert watch["calibrage"] == pytest.approx(race_vdot / 50, abs=0.001)
    assert watch["vdot_estime"] == pytest.approx(51 * race_vdot / 50, abs=0.1)
    # La course a 4 mois : elle compte encore, mais dépréciée
    assert est["composantes"]["performances"]["vdot_deprecie"] < race_vdot


def test_vo2max_par_relation_fc_vitesse_sur_un_footing():
    # 10 km en 50 min (200 m/min) à 75 % de la réserve cardiaque (repos 50, max 190 -> FC 155)
    acts = pd.DataFrame([{
        "activity_id": 1, "sport": "running", "start_time": "2026-09-28 08:00", "distance_m": 10000,
        "duration_s": 3000, "avg_hr": 155.0, "avg_speed_ms": 200 / 60, "elevation_gain_m": 40,
    }])
    est = hr_speed_vo2max(acts, labels({1: "ef"}), hr_rest=50, hr_max=190)
    flat_speed_m_min = flat_equivalent_m(10000, 40) / 50  # 40 m de D+ convertis en distance de plat
    expected = 3.5 + (vo2_at_speed(flat_speed_m_min) - 3.5) / 0.75
    assert est["vo2max"].iloc[0] == pytest.approx(expected, abs=0.01)
    # Le fractionné est exclu : ses moyennes mélangent efforts et récupérations
    assert hr_speed_vo2max(acts, labels({1: "fractionne"}), 50, 190).empty


def test_lecture_des_predictions_de_la_montre():
    raw = [{"calendarDate": "2026-10-01", "time5K": 1350, "time10K": 2850, "timeHalfMarathon": 6400, "timeMarathon": 13800}]
    assert garmin_predictions(raw) == {"5k": 1350.0, "10k": 2850.0, "semi": 6400.0, "marathon": 13800.0}
    assert garmin_predictions(None) == {}


def test_prediction_10k_coherente_avec_une_course_recente():
    acts = activities((1, "running", "2026-09-28 09:00", 10000, 2850, None, None, None))
    est = estimate_vdot(collect_performances(acts, labels({1: "course"})), {}, TODAY)
    assert predict_time_s(est["vdot"], 10000) == pytest.approx(2850, rel=0.01)  # 3 jours d'ancienneté : quasi identique


# --- Chrono des courses et allures personnelles ----------------------------------------------------

def test_course_chronometree_sur_la_distance_officielle():
    # Enregistrement de 10,6 km en 52 min (échauffement inclus), meilleur 10 km de l'activité : 47'30"
    run = pd.Series({"distance_m": 10600, "duration_s": 3120, "moving_duration_s": 3100, "fastest_10k_s": 2850.0})
    assert race_effort(run) == (10000.0, 2850.0)


def session(i, kind, pace_s, hr, fastest_1k=None, km=8):
    return {"activity_id": i, "sport": "running", "start_time": f"2026-09-{i:02d} 08:00", "distance_m": km * 1000,
            "duration_s": pace_s * km, "moving_duration_s": pace_s * km, "avg_hr": hr, "fastest_1k_s": fastest_1k,
            "kind": kind}


def test_allures_observees_et_recommandation():
    rows = [session(1, "ef", 345, 140), session(2, "ef", 350, 142), session(3, "ef", 340, 138), session(4, "ef", 355, 141),
            session(5, "tempo", 290, 168), session(6, "tempo", 295, 170),
            session(7, "fractionne", 330, 160, fastest_1k=235), session(8, "fractionne", 335, 158, fastest_1k=240)]
    acts = pd.DataFrame(rows)
    labels_df = pd.DataFrame({"activity_id": acts["activity_id"], "label": acts["kind"], "suggestion": "ef"})
    runs = recent_runs(acts.drop(columns="kind"), labels_df, TODAY, 120)
    paces = personal_training_paces(runs, hr_max=193, vdot_value=43)

    ef = paces["zones"]["ef"]
    assert ef["recommandation"]["source"] == "observe"
    assert ef["observe"]["allure_mediane_s"] == 348  # médiane de 340, 345, 350, 355
    # Fractionné : allure des meilleurs km, pas l'allure moyenne de la séance
    assert paces["zones"]["fractionne"]["observe"]["allure_mediane"].startswith("3'5")
    # Modèle FC -> allure : appris sur EF + tempo, plus rapide quand la FC monte
    model_ef = ef["modele_fc"]
    assert model_ef["allure_rapide"] < model_ef["allure_lente"]
    assert paces["allure_max"]["meilleur_1km"] == "3'55\"/km"


# --- Dénivelé ------------------------------------------------------------------------------------

def test_equivalence_de_scarf():
    assert flat_equivalent_m(10000, 100) == pytest.approx(10792)


def test_une_sortie_vallonnee_n_est_plus_penalisee():
    # Même effort cardiaque, même temps : 10 km plats, ou 10 km avec 150 m de D+ (donc plus durs)
    base = {"activity_id": 1, "sport": "running", "start_time": "2026-09-28 08:00", "distance_m": 10000,
            "duration_s": 3000, "avg_hr": 155.0, "avg_speed_ms": 10000 / 3000}
    flat = hr_speed_vo2max(pd.DataFrame([{**base, "elevation_gain_m": 0}]), labels({1: "ef"}), 50, 190)
    hilly = hr_speed_vo2max(pd.DataFrame([{**base, "elevation_gain_m": 150}]), labels({1: "ef"}), 50, 190)
    assert hilly["vo2max"].iloc[0] > flat["vo2max"].iloc[0]  # courir aussi vite en côte demande plus de capacité


def test_le_gap_garmin_est_prioritaire():
    base = {"activity_id": 1, "sport": "running", "start_time": "2026-09-28 08:00", "distance_m": 10000,
            "duration_s": 3000, "avg_hr": 155.0, "avg_speed_ms": 10000 / 3000, "elevation_gain_m": 150}
    with_gap = hr_speed_vo2max(pd.DataFrame([{**base, "avg_gap_speed_ms": 3.5}]), labels({1: "ef"}), 50, 190)
    expected = 3.5 + (vo2_at_speed(3.5 * 60) - 3.5) / ((155 - 50) / 140)
    assert with_gap["vo2max"].iloc[0] == pytest.approx(expected, abs=0.01)


def test_course_vallonnee_convertie_en_plat():
    acts = pd.DataFrame([{"activity_id": 1, "sport": "running", "start_time": "2026-09-28 09:00", "distance_m": 10000,
                          "duration_s": 3000, "elevation_gain_m": 120, "fastest_10k_s": 3000.0}])
    perf = collect_performances(acts, labels({1: "course"}))
    assert perf["distance_m"].iloc[0] == pytest.approx(flat_equivalent_m(10000, 120))
    assert perf["vdot"].iloc[0] > vdot(10000, 3000)  # 50 min avec du D+ vaut mieux que 50 min sur le plat


# --- Dérive cardiaque et dépréciation liée à la charge ------------------------------------------

def test_sortie_longue_exclue_de_la_relation_fc_vitesse():
    base = {"sport": "running", "start_time": "2026-09-28 08:00", "avg_hr": 152.0, "elevation_gain_m": 0}
    acts = pd.DataFrame([
        {**base, "activity_id": 1, "distance_m": 8000, "duration_s": 45 * 60, "avg_speed_ms": 8000 / 2700},
        {**base, "activity_id": 2, "distance_m": 14000, "duration_s": 95 * 60, "avg_speed_ms": 14000 / 5700},
    ])
    est = hr_speed_vo2max(acts, labels({1: "ef", 2: "ef"}), hr_rest=42, hr_max=193)
    assert len(est) == 1  # la sortie de 95 min (dérive cardiaque) n'entre pas dans l'estimation


def test_pas_de_depreciation_si_la_charge_est_maintenue():
    from processing.performance import MAX_DECAY, decay_factors

    perf = pd.DataFrame({"date": pd.to_datetime(["2026-05-17"]), "vdot": [43.0]})
    dates = pd.date_range("2026-05-01", "2026-10-01")
    maintained = pd.Series(40.0, index=dates)
    halved = pd.Series([40.0 if d < pd.Timestamp("2026-09-01") else 20.0 for d in dates], index=dates)
    assert decay_factors(perf, TODAY, maintained).iloc[0] == 0  # forme de fond maintenue : rien n'est perdu
    full = decay_factors(perf, TODAY).iloc[0]  # sans série de charge : dépréciation complète
    assert 0 < full <= MAX_DECAY
    assert decay_factors(perf, TODAY, halved).iloc[0] == pytest.approx(full / 2)  # charge divisée par 2
