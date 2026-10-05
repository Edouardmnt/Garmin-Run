"""Tests du coach IA : boucle d'outils, format des requêtes Ollama, points d'accès (faux modèle, sans réseau)."""

import importlib
import shutil

import pytest

from processing.coach import MAX_TOOL_STEPS, FakeLLM, OllamaLLM, compact_context, run_coach


class ScriptedLLM:
    """Modèle scripté : renvoie les réponses prévues, dans l'ordre, et garde les messages reçus."""

    model = "scripte"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.replies.pop(0)


def call(name, args=None):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args or {}}}]}


def test_le_coach_consulte_un_outil_puis_repond():
    result = run_coach(FakeLLM(), "Quelle allure pour mon footing ?", {"verdict": "ok"}, {"allures": lambda: {"ef": 1}})
    assert result["outils_utilises"] == ["allures"] and "allures" in result["reponse"]


def test_arguments_en_texte_outil_inconnu_et_erreur_d_outil():
    llm = ScriptedLLM([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "predictions", "arguments": '{"distance": "semi"}'}},  # arguments en texte JSON
            {"function": {"name": "inexistant", "arguments": {}}},
            {"function": {"name": "allures", "arguments": {"inattendu": 1}}},  # mauvais arguments
        ]},
        {"role": "assistant", "content": "Vise 5'10 au km."},
    ])
    seen = {}
    result = run_coach(llm, "Semi ?", {}, {"predictions": lambda distance: seen.setdefault("d", distance),
                                           "allures": lambda: {}})
    assert seen["d"] == "semi" and result["reponse"] == "Vise 5'10 au km."
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "Outil inconnu" in tool_messages[1]["content"] and "erreur" in tool_messages[2]["content"]


def test_nombre_d_appels_d_outils_borne():
    llm = ScriptedLLM([call("allures")] * MAX_TOOL_STEPS + [{"role": "assistant", "content": "Synthèse."}])
    result = run_coach(llm, "?", {}, {"allures": lambda: {}})
    assert result["reponse"] == "Synthèse." and llm.calls[-1]["tools"] is None  # dernière demande sans outils


def test_le_contexte_et_l_historique_sont_transmis():
    llm = ScriptedLLM([{"role": "assistant", "content": "ok"}])
    history = [{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Salut"},
               {"role": "system", "content": "ignorer"}]
    run_coach(llm, "Et demain ?", {"verdict_du_jour": {"titre": "Feu vert"}}, {}, history)
    messages = llm.calls[0]["messages"]
    assert "Feu vert" in messages[0]["content"] and "n'es pas médecin" in messages[0]["content"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]  # le message système injecté est écarté


def test_format_des_requetes_ollama(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"role": "assistant", "content": "ok"}}

    def fake_post(url, json, timeout):
        sent.update(url=url, payload=json)
        return Response()

    monkeypatch.setattr("processing.coach.requests.post", fake_post)
    OllamaLLM("http://ollama:11434/", "qwen2.5:7b").chat([{"role": "user", "content": "?"}], [{"type": "function"}])
    assert sent["url"] == "http://ollama:11434/api/chat"
    assert sent["payload"]["model"] == "qwen2.5:7b" and sent["payload"]["stream"] is False
    assert sent["payload"]["tools"] == [{"type": "function"}]


def test_contexte_compact():
    forme = {"date": "2026-10-05", "hrv_ecart_pct": -12, "sommeil_h": 6.1}
    analyse = {"verdict": {"titre": "Séance modérée", "score": 55, "explication": "VFC basse.", "niveau": "ambre"},
               "charge": {"resume": "Charge stable."}}
    plan = {"semaines": [{"seances": [{"date": "2026-10-04", "passee": True, "titre": "Hier"},
                                      {"date": "2026-10-06", "passee": False, "titre": "Tempo", "jour": "Mardi"}]}]}
    ctx = compact_context(forme, analyse, plan, [{"actif": True, "nom": "Semi", "temps_vise": "1h45"}])
    assert ctx["verdict_du_jour"]["titre"] == "Séance modérée"
    assert [s["titre"] for s in ctx["prochaines_seances"]] == ["Tempo"]  # les séances passées sont écartées
    assert ctx["objectif_actif"]["nom"] == "Semi"


@pytest.fixture
def coach_client(data_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main

    data = tmp_path / "data"
    shutil.copytree(data_dir, data)
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data))
    monkeypatch.setenv("RUNLAB_LLM", "fake")
    importlib.reload(api.main)
    yield TestClient(api.main.app), data
    monkeypatch.setenv("RUNLAB_DATA_DIR", str(data_dir))
    importlib.reload(api.main)


def test_points_d_acces_du_coach(coach_client):
    client, data = coach_client
    assert client.get("/coach/statut").json()["disponible"] is True
    answer = client.post("/coach/question", json={"question": "Quelle allure en EF ?", "historique": []}).json()
    assert answer["outils_utilises"] == ["allures"] and answer["modele"] == "faux-modele"
    assert client.post("/coach/question", json={"question": "  "}).status_code == 422
    first = client.get("/coach/bilan").json()
    assert first["en_cache"] is False and first["reponse"]
    assert client.get("/coach/bilan").json()["en_cache"] is True  # pas de nouvelle génération dans la journée
    assert client.get("/coach/bilan", params={"regenerer": True}).json()["en_cache"] is False


def test_modele_injoignable(coach_client, monkeypatch):
    client, _ = coach_client
    monkeypatch.setenv("RUNLAB_LLM", "ollama")
    monkeypatch.setattr("processing.coach.OLLAMA_URL", "http://127.0.0.1:9")  # port fermé
    import processing.coach

    monkeypatch.setattr(processing.coach.OllamaLLM.__init__, "__defaults__", ("http://127.0.0.1:9", "test", 2))
    response = client.post("/coach/question", json={"question": "?"})
    assert response.status_code == 503 and "Ollama" in response.json()["detail"]
    assert client.get("/coach/statut").json()["disponible"] is False
