"""Tests du chargement incrémental (watermark) et de la fusion des activités."""

import json
from datetime import date

from ingestion.garmin_export import INITIAL_DAYS, MAX_DAYS, REFRESH_DAYS, compute_start, merge_activities

TODAY = date(2026, 10, 10)


def write_days(raw_dir, *days):
    (raw_dir / "daily").mkdir(parents=True, exist_ok=True)
    for d in days:
        (raw_dir / "daily" / f"{d}.json").write_text("{}")


def test_dossier_vide_chargement_initial(tmp_path):
    assert (TODAY - compute_start(TODAY, tmp_path)).days == INITIAL_DAYS


def test_reprend_au_dernier_jour_connu(tmp_path):
    # Dernière synchro le 3 octobre, PC éteint une semaine : on repart du 1er, sans trou
    write_days(tmp_path, "2026-09-30", "2026-10-02", "2026-10-03")
    assert compute_start(TODAY, tmp_path) == date(2026, 10, 3 - REFRESH_DAYS)


def test_synchro_quotidienne_ne_recupere_que_quelques_jours(tmp_path):
    write_days(tmp_path, "2026-10-09")
    assert (TODAY - compute_start(TODAY, tmp_path)).days == REFRESH_DAYS + 1


def test_garde_fou_un_an_maximum(tmp_path):
    write_days(tmp_path, "2020-01-01")
    assert (TODAY - compute_start(TODAY, tmp_path)).days == MAX_DAYS


def test_fusion_sans_doublon_et_mise_a_jour(tmp_path):
    old = [{"activityId": 1, "startTimeLocal": "2026-10-01"}, {"activityId": 2, "startTimeLocal": "2026-10-02", "v": "ancien"}]
    (tmp_path / "activities.json").write_text(json.dumps(old))
    new = [{"activityId": 2, "startTimeLocal": "2026-10-02", "v": "nouveau"}, {"activityId": 3, "startTimeLocal": "2026-10-03"}]
    merged = merge_activities(new, tmp_path)
    assert [a["activityId"] for a in merged] == [1, 2, 3]
    assert merged[1]["v"] == "nouveau"
