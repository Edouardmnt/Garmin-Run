"""Coach IA de Foulée : un modèle de langage local (Ollama) qui s'appuie sur les données de l'API.

- Contexte compact fourni d'office (forme, verdict, analyses, objectif, prochaines séances) : un petit
  modèle local n'a pas à deviner quoi consulter pour les questions courantes.
- Outils pour le reste (prédictions sur une distance, allures, nutrition, séances...) : le modèle les
  appelle lui-même (tool calling), et la réponse indique les données consultées.
- Garde-fous dans les consignes : pas de diagnostic médical, pas de chiffre inventé, rien n'est modifié.

Configuration : OLLAMA_URL (http://localhost:11434 par défaut), OLLAMA_MODEL (qwen2.5:7b par défaut),
RUNLAB_LLM=fake pour un faux modèle déterministe (tests et CI, sans réseau).
"""

import json
import os
from collections.abc import Callable

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "300"))  # un modèle local sur CPU peut être lent
MAX_TOOL_STEPS = 4

SYSTEM_PROMPT = """Tu es le coach de course à pied de l'application Foulée. Tu tutoies l'utilisateur et tu réponds en français.

Règles :
- Appuie-toi uniquement sur le CONTEXTE ci-dessous et sur les résultats des outils. N'invente jamais un chiffre.
  Si une donnée manque, appelle l'outil adapté ; si elle n'existe pas, dis-le simplement.
- Cite les chiffres utiles (allures en min/km, temps, VFC, charge) pour justifier tes conseils.
- Tu n'es pas médecin : en cas de douleur, de symptôme ou de malaise, recommande d'arrêter et de consulter
  un professionnel de santé, sans poser de diagnostic.
- Tu ne modifies rien : tu proposes, l'utilisateur décide.
- Sois concis : 3 à 8 phrases, ou une courte liste si c'est plus clair. Pas de formules creuses.
- Les allures sont en équivalent plat ; le planning respecte déjà les jours de tennis et la forme du jour.

CONTEXTE (données de l'utilisateur, à jour) :
{context}"""

BILAN_PROMPT = """Fais le bilan de ma semaine d'entraînement en 5 à 8 phrases : ce qui s'est bien passé, les points
de vigilance (récupération, sommeil, stress, charge), et un conseil concret pour les prochains jours en lien avec
mon objectif. Termine par la séance du jour ou la prochaine séance prévue, avec son allure."""

TOOLS = [
    {"name": "predictions", "description": "Temps et allure prédits pour une course (distance classique ou libre, D+).",
     "parameters": {"type": "object", "properties": {
         "distance": {"type": "string", "enum": ["5k", "10k", "semi", "marathon"]},
         "distance_km": {"type": "number", "description": "Distance libre en km, à la place de distance"},
         "denivele_m": {"type": "integer", "description": "D+ du parcours en mètres"}}}},
    {"name": "allures", "description": "Allures d'entraînement personnelles (EF, tempo, fractionné) et FC cibles.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "nutrition", "description": "Nutrition et hydratation pour une course selon la durée prévue et la température.",
     "parameters": {"type": "object", "properties": {
         "distance": {"type": "string", "enum": ["5k", "10k", "semi", "marathon"]},
         "distance_km": {"type": "number"}, "temperature_c": {"type": "number"}}}},
    {"name": "seances", "description": "Dernières sorties de course réalisées (date, distance, allure, FC, type).",
     "parameters": {"type": "object", "properties": {"limite": {"type": "integer"}}}},
    {"name": "planning", "description": "Planning complet de l'objectif actif, semaine par semaine.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "objectifs", "description": "Objectifs de course, temps prédit et écart au temps visé.",
     "parameters": {"type": "object", "properties": {}}},
]


def ollama_tools() -> list[dict]:
    return [{"type": "function", "function": t} for t in TOOLS]


# --- Modèles ---------------------------------------------------------------------------------------

class OllamaLLM:
    """Client minimal de l'API de chat d'Ollama (non streamé)."""

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, timeout: int = TIMEOUT_S):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": 0.3}}
        if tools:
            payload["tools"] = tools
        response = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]

    def status(self) -> dict:
        """Ollama joignable ? Modèle téléchargé ?"""
        try:
            models = requests.get(f"{self.url}/api/tags", timeout=5).json().get("models", [])
        except requests.RequestException as exc:
            return {"disponible": False, "modele": self.model, "url": self.url, "erreur": str(exc)}
        names = [m.get("name", "") for m in models]
        present = any(n == self.model or n.split(":")[0] == self.model for n in names)
        return {"disponible": present, "modele": self.model, "url": self.url, "modeles_installes": names,
                "erreur": None if present else f"Modèle absent : lance « ollama pull {self.model} »"}


class FakeLLM:
    """Faux modèle déterministe : consulte les allures, puis répond en citant ce qu'il a lu (tests, CI)."""

    model = "faux-modele"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        tool_results = [m for m in messages if m["role"] == "tool"]
        if tools and not tool_results:
            return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "allures", "arguments": {}}}]}
        seen = ", ".join(m.get("tool_name", "?") for m in tool_results) or "contexte"
        return {"role": "assistant", "content": f"Réponse de test fondée sur : {seen}."}

    def status(self) -> dict:
        return {"disponible": True, "modele": self.model, "url": "local", "erreur": None}


def get_llm():
    return FakeLLM() if os.getenv("RUNLAB_LLM") == "fake" else OllamaLLM()


# --- Contexte et boucle d'outils -------------------------------------------------------------------

def compact_context(forme: dict, analyse: dict, plan: dict, goals: list[dict]) -> dict:
    """Ce qu'un coach doit savoir d'emblée, en quelques centaines de mots."""
    upcoming = [s for w in plan.get("semaines", []) for s in w["seances"] if not s.get("passee")][:4]
    active = next((g for g in goals if g.get("actif")), None)
    return {
        "date_des_donnees": forme.get("date"),
        "verdict_du_jour": {k: analyse["verdict"][k] for k in ("titre", "score", "explication")},
        "forme": {k: forme.get(k) for k in ("hrv_ecart_pct", "sommeil_h", "fc_repos", "fraicheur_tsb",
                                            "ajustement_chrono_pct")},
        "analyses": {k: analyse[k]["resume"] for k in ("charge", "forme", "recuperation", "sommeil", "stress",
                                                       "activites") if k in analyse},
        "objectif_actif": None if active is None else {k: active.get(k) for k in (
            "nom", "libelle", "date_course", "jours_restants", "temps_vise", "temps_predit", "statut")},
        "prochaines_seances": [{k: s.get(k) for k in ("date", "jour", "titre", "description", "allure", "fc_cible")}
                               for s in upcoming],
        "adaptations_du_planning": plan.get("personnalisation", []),
    }


def run_coach(llm, user_message: str, context: dict, tools: dict[str, Callable[..., dict]],
              history: list[dict] | None = None) -> dict:
    """Une question -> une réponse, avec les appels d'outils nécessaires (au plus MAX_TOOL_STEPS)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=json.dumps(context, ensure_ascii=False))}]
    messages += [m for m in (history or []) if m.get("role") in ("user", "assistant")][-6:]
    messages.append({"role": "user", "content": user_message})
    used = []
    for _ in range(MAX_TOOL_STEPS):
        reply = llm.chat(messages, ollama_tools())
        calls = reply.get("tool_calls") or []
        if not calls:
            return {"reponse": reply.get("content", "").strip(), "outils_utilises": used,
                    "modele": getattr(llm, "model", "?")}
        messages.append({"role": "assistant", "content": reply.get("content", ""), "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):  # certains modèles renvoient les arguments en texte JSON
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name in tools:
                try:
                    result = tools[name](**args)
                except Exception as exc:  # un outil mal appelé ne doit pas faire échouer la conversation
                    result = {"erreur": str(exc)}
                used.append(name)
            else:
                result = {"erreur": f"Outil inconnu : {name}"}
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, ensure_ascii=False, default=str)[:6000]})
    # Trop d'appels d'outils : on demande une réponse finale, sans outils
    final = llm.chat(messages + [{"role": "user", "content": "Réponds maintenant avec ce que tu as."}], None)
    return {"reponse": final.get("content", "").strip(), "outils_utilises": used, "modele": getattr(llm, "model", "?")}
