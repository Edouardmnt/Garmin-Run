"""Modifications du planning proposées par le coach, validées par l'utilisateur ; exercices pour une douleur."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date

import pytest

from processing.adjustments import MarkerFilter, apply_adjustments, parse_proposal, split_marker, validate
from processing.coach import detect_topics
from processing.planning import step
from processing.rehab import detect_zones, mentions_pain, pain_data

TODAY = date(2026, 10, 5)  # un lundi
PACES = {"ef": {"rapide_s": 300, "lente_s": 330, "fc_cible": [140, 150]},
         "tempo": {"rapide_s": 270, "lente_s": 285, "fc_cible": [160, 165]}}


def sess(day, kind, title, km=8.0, passee=False):
    return {"date": day, "jour": "?", "type": kind, "titre": title, "description": "desc", "distance_km": km,
            "duree_min": 45, "allure": "5'00\"/km", "fc_cible": None, "objectif": "", "passee": passee,
            "etapes": [step("echauffement", duree_s=900), {"type": "repetition", "repetitions": 6, "etapes": [
                step("effort", distance_m=1000), step("recuperation", duree_s=120)]}, step("effort", duree_s=600)]}


def plan():
    return {"semaines": [
        {"numero": 1, "debut": "2026-10-05", "phase": "Développement", "volume_km": 0, "seances": [
            sess("2026-10-04", "ef", "Footing passé", passee=True),
            sess("2026-10-06", "fractionne", "Fractionné"), sess("2026-10-08", "ef", "Footing"),
            sess("2026-10-11", "longue", "Sortie longue", 14)]},
        {"numero": 2, "debut": "2026-10-12", "phase": "Développement", "volume_km": 0, "seances": [
            sess("2026-10-13", "tempo", "Tempo"), sess("2026-10-18", "course", "Course : 10 km", 10)]}],
        "notes": []}


def accept(adj):
    return {**adj, "statut": "accepte"}


def test_deplacer_une_seance():
    adj, error = validate({"date": "2026-10-11", "action": "deplacer", "nouvelle_date": "2026-10-10"}, plan(), TODAY)
    assert error is None and adj["statut"] == "propose" and adj["apres"]["date"] == "2026-10-10"
    assert "Déplacer « Sortie longue » du dimanche 11/10 au samedi 10/10" == adj["libelle"]
    out = apply_adjustments(plan(), [accept(adj)], PACES, TODAY)
    days = [s["date"] for s in out["semaines"][0]["seances"]]
    assert "2026-10-10" in days and "2026-10-11" not in days
    moved = next(s for s in out["semaines"][0]["seances"] if s["date"] == "2026-10-10")
    assert moved["jour"] == "Samedi" and moved["ajustement"]["id"] == adj["id"]


def test_deplacement_d_une_semaine_a_l_autre():
    adj, _ = validate({"date": "2026-10-11", "action": "deplacer", "nouvelle_date": "2026-10-12"}, plan(), TODAY)
    out = apply_adjustments(plan(), [accept(adj)], PACES, TODAY)
    assert out["semaines"][1]["seances"][0]["titre"] == "Sortie longue"
    assert out["semaines"][0]["volume_km"] == 24.0  # volumes recalculés (sortie longue partie)


@pytest.mark.parametrize("raw, message", [
    ({"date": "2026-10-11", "action": "deplacer", "nouvelle_date": "2026-10-08"}, "déjà une séance"),
    ({"date": "2026-10-04", "action": "alleger"}, "passée"),
    ({"date": "2026-10-11", "action": "deplacer", "nouvelle_date": "2026-10-01"}, "passé"),
    ({"date": "2026-10-18", "action": "repos"}, "jour de course"),
    ({"date": "2026-10-06", "action": "intensifier"}, "déjà une séance intense"),
    ({"date": "2026-10-09", "action": "repos"}, "Aucune séance"),
    ({"date": "2026-10-06", "action": "sprinter"}, "Action inconnue"),
    ({"date": "2026-10-11", "action": "deplacer", "nouvelle_date": "2026-12-25"}, "au-delà du planning"),
])
def test_propositions_refusees(raw, message):
    adj, error = validate(raw, plan(), TODAY if raw["date"] != "2026-10-04" else date(2026, 10, 5))
    assert adj is None and message in error


def test_risques_signales():
    # Tempo déplacé le lendemain du fractionné : deux séances dures d'affilée ; mercredi = jour de tennis
    adj, _ = validate({"date": "2026-10-13", "action": "deplacer", "nouvelle_date": "2026-10-07"}, plan(), TODAY, {2})
    assert any("autre séance dure" in w for w in adj["avertissements"])
    assert any("tennis" in w for w in adj["avertissements"])
    adj, _ = validate({"date": "2026-10-08", "action": "intensifier"}, plan(), TODAY, verdict="orange")
    assert any("forme" in w for w in adj["avertissements"])


def test_alleger_intensifier_raccourcir_repos():
    a, _ = validate({"date": "2026-10-06", "action": "alleger"}, plan(), TODAY, paces=PACES)
    b, _ = validate({"date": "2026-10-08", "action": "intensifier"}, plan(), TODAY, paces=PACES)
    c, _ = validate({"date": "2026-10-13", "action": "raccourcir", "facteur": 0.1}, plan(), TODAY)
    d, _ = validate({"date": "2026-10-11", "action": "repos"}, plan(), TODAY)
    assert c["facteur"] == 0.4  # facteur borné
    out = apply_adjustments(plan(), [accept(x) for x in (a, b, c, d)], PACES, TODAY)
    sessions = {s["date"]: s for w in out["semaines"] for s in w["seances"]}
    assert sessions["2026-10-06"]["type"] == "ef" and sessions["2026-10-06"]["distance_km"] == 5.6
    assert sessions["2026-10-08"]["type"] == "tempo" and sessions["2026-10-08"]["etapes"][1]["duree_s"] == 900
    short = sessions["2026-10-13"]
    assert short["distance_km"] == 3.2 and short["etapes"][1]["repetitions"] == 2  # 6 × 1000 m -> 2
    assert short["etapes"][0]["duree_s"] == 900  # l'échauffement ne change pas
    assert "2026-10-11" not in sessions and any("Repos décidé avec le coach" in n for n in out["notes"])


def test_seuls_les_ajustements_valides_s_appliquent():
    adj, _ = validate({"date": "2026-10-06", "action": "repos"}, plan(), TODAY)
    for status in ("propose", "refuse", "annule"):
        out = apply_adjustments(plan(), [{**adj, "statut": status}], PACES, TODAY)
        assert any(s["date"] == "2026-10-06" for s in out["semaines"][0]["seances"])


def test_marqueur_jamais_affiche_meme_coupe_en_morceaux():
    f = MarkerFilter()
    chunks = ["Je te propose de décaler ", "ta sortie. Tu valides ?\n[", "[PROPO", 'SITION]] {"date": "2026-10-11", ',
              '"action": "repos"}']
    visible = "".join(f.feed(c) for c in chunks) + f.flush()
    assert visible == "Je te propose de décaler ta sortie. Tu valides ?" and "[[" not in visible
    assert parse_proposal(f.hidden) == {"date": "2026-10-11", "action": "repos"}


def test_lecture_tolerante_de_la_proposition():
    text = 'Ok.\n[[PROPOSITION]] ```json\n{“date”: “2026-10-11”, “action”: “alleger”}\n```'
    visible, hidden = split_marker(text)
    assert visible == "Ok." and parse_proposal(hidden)["action"] == "alleger"
    assert parse_proposal("[[PROPOSITION]] pas de json") is None and parse_proposal(None) is None


# --- Douleurs ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("question, zones", [
    ("J'ai mal au genou", ["genou_avant"]), ("douleur à l'extérieur du genou en descente", ["genou_exterieur"]),
    ("Mon tendon d'Achille tire le matin", ["achille"]), ("mal sous le pied au réveil", ["pied"]),
    ("J'adore la course à pied", []), ("C'est normal d'être essoufflé ?", []),
])
def test_zones_reperees(question, zones):
    assert detect_zones(question) == zones


def test_une_douleur_passe_en_premier_avec_le_planning():
    assert detect_topics("J'ai mal au tibia depuis ma sortie d'hier")[:2] == ["douleur", "planning"]
    assert not mentions_pain("Quelle allure pour mon footing ?")
    data = pain_data("tendinite d'Achille")
    assert data["zones"][0]["exercices"] and data["signaux_alerte"] and "kinésithérapeute" in data["zones"][0]["specialiste"]
    assert pain_data("j'ai mal")["zone_inconnue"]


# --- Parcours complet par l'API ----------------------------------------------------------------------

@pytest.fixture
def api(data_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    subprocess.run([sys.executable, "scripts/make_run_labels.py"], env={**os.environ, "RUNLAB_DATA_DIR": str(data)},
                   check=True, capture_output=True)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RUNLAB_LLM", "fake")
    monkeypatch.setenv("RUNLAB_WATCH_AUTO", "0")
    importlib.reload(api.main)
    yield api.main, TestClient(api.main.app)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)


def ask(client, question):
    with client.stream("POST", "/coach/question/flux", json={"question": question}) as r:
        return "".join(r.iter_text())


def test_le_coach_propose_l_utilisateur_valide(api):
    main, client = api
    body = ask(client, "Je suis crevé, allège ma prochaine séance")
    assert "[[PROPOSITION]]" in body and "PROPOSITION]] {\"date\"" not in body.split("\n[[")[0]
    adj = json.loads(body.split("[[PROPOSITION]] ")[1].splitlines()[0])
    assert adj["statut"] == "propose" and adj["action"] == "alleger"
    day = adj["date"]

    before = client.get("/planning/actif").json()
    assert next(s for w in before["semaines"] for s in w["seances"] if s["date"] == day).get("ajustement") is None

    done = client.post(f"/planning/ajustements/{adj['id']}/valider").json()
    assert done["ajustement"]["statut"] == "accepte"
    after = client.get("/planning/actif").json()
    changed = next(s for w in after["semaines"] for s in w["seances"] if s["date"] == day)
    assert changed["titre"] == "Footing facile (allégé)" and changed["ajustement"]["id"] == adj["id"]
    assert [a["id"] for a in after["ajustements"]] == [adj["id"]]
    assert client.post(f"/planning/ajustements/{adj['id']}/valider").status_code == 409  # pas deux fois

    client.post(f"/planning/ajustements/{adj['id']}/annuler")
    back = client.get("/planning/actif").json()
    assert next(s for w in back["semaines"] for s in w["seances"] if s["date"] == day).get("ajustement") is None


def test_proposition_impossible_signalee(api):
    _, client = api
    body = ask(client, "Décale ma prochaine séance")  # le faux modèle écrit une date illisible
    assert "[[PROPOSITION_REFUSEE]]" in body and "illisible" in body


def test_refuser(api):
    _, client = api
    adj = json.loads(ask(client, "Remplace ma prochaine séance par du repos").split("[[PROPOSITION]] ")[1].splitlines()[0])
    assert client.post(f"/planning/ajustements/{adj['id']}/refuser").json()["ajustement"]["statut"] == "refuse"
    assert client.get("/planning/ajustements", params={"statut": "refuse"}).json()["ajustements"][0]["id"] == adj["id"]


def test_la_montre_est_mise_a_jour_si_la_seance_du_jour_change(api, monkeypatch):
    main, client = api
    monkeypatch.setenv("RUNLAB_WATCH_AUTO", "1")
    plan = client.get("/planning/actif").json()
    first = next(s for w in plan["semaines"] for s in w["seances"] if s["date"] >= os.environ["RUNLAB_TODAY"])
    monkeypatch.setattr(main, "today_date", lambda: date.fromisoformat(first["date"]))
    sent = []
    monkeypatch.setattr("ingestion.garmin_export.connect", lambda: "client")
    monkeypatch.setattr(main, "send_session", lambda c, s, d: sent.append(s) or {"statut": "remplacee"})
    adj = client.post("/planning/ajustements", json={"proposition": {"date": first["date"], "action": "raccourcir",
                                                                     "facteur": 0.5}}).json()["ajustement"]
    done = client.post(f"/planning/ajustements/{adj['id']}/valider").json()
    assert done["montre"]["statut"] == "remplacee" and sent[0]["ajustement"]["id"] == adj["id"]


def test_exercices_pour_une_douleur(api):
    _, client = api
    body = ask(client, "J'ai mal à l'extérieur du genou depuis hier")
    data = json.loads(body.split("[[EXERCICES]] ")[1].splitlines()[0])
    assert data["zones"][0]["id"] == "genou_exterieur" and data["signaux_alerte"]
    assert client.get("/coach/exercices", params={"question": "mal au tibia"}).json()["zones"][0]["id"] == "tibia"


# --- Secours par mots-clés -------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("Supprime ma prochaine séance et remplace-la par du repos.", {"date": "2026-10-06", "action": "repos"}),
    ("Je ne peux pas courir à ma prochaine séance, décale-la au lendemain.",
     {"date": "2026-10-06", "action": "deplacer", "nouvelle_date": "2026-10-07"}),
    ("décale ma sortie longue à samedi", {"date": "2026-10-11", "action": "deplacer", "nouvelle_date": "2026-10-10"}),
    ("déplace la sortie longue de dimanche à samedi",
     {"date": "2026-10-11", "action": "deplacer", "nouvelle_date": "2026-10-10"}),
    ("Je suis crevé, tu peux alléger ma prochaine séance ?", {"date": "2026-10-06", "action": "alleger"}),
    ("repos demain", {"date": "2026-10-06", "action": "repos"}),
    ("raccourcis mon tempo", {"date": "2026-10-13", "action": "raccourcir"}),
    ("allège jeudi", {"date": "2026-10-08", "action": "alleger"}),
    ("décale ma séance de mardi", None),        # déplacer, mais où ? le coach doit demander
    ("repos mercredi", None),                   # pas de séance mercredi : rien à proposer
    ("Quelle allure pour mon footing ?", None),  # aucune demande de modification
])
def test_demande_explicite_comprise(question, expected):
    from processing.adjustments import fallback_proposal

    found = fallback_proposal(question, plan(), TODAY)
    if expected is None:
        assert found is None
    else:
        assert {k: found[k] for k in expected} == expected


def test_secours_quand_le_modele_n_ecrit_pas_de_proposition(api):
    _, client = api
    body = ask(client, "Raccourcis ma prochaine séance s'il te plaît")  # le faux modèle n'écrit aucune proposition
    adj = json.loads(body.split("[[PROPOSITION]] ")[1].splitlines()[0])
    assert adj["action"] == "raccourcir" and adj["origine"] == "demande"
    body = ask(client, "Je suis crevé, allège ma prochaine séance")  # ici, c'est le modèle qui propose
    assert json.loads(body.split("[[PROPOSITION]] ")[1].splitlines()[0])["origine"] == "coach"
