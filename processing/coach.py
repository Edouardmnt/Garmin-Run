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
import re
from collections.abc import Callable

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "300"))  # un modèle local sur CPU peut être lent
MAX_TOOL_STEPS = 4
# "direct" (par défaut) : un seul appel au modèle, avec les données choisies d'après la question.
# "outils" : le modèle choisit lui-même ses outils (plus souple, mais plusieurs appels : réservé aux GPU).
COACH_MODE = os.getenv("COACH_MODE", "direct")
OLLAMA_OPTIONS = {"temperature": 0.3, "num_ctx": 4096, "num_predict": 450}
KEEP_ALIVE = "30m"  # le modèle reste en mémoire entre deux questions

SYSTEM_PROMPT = """Tu es le coach de course à pied de l'application Foulée. Tu tutoies l'utilisateur et tu réponds en français.

Règles :
- Appuie-toi uniquement sur le CONTEXTE ci-dessous et sur les résultats des outils. N'invente jamais un chiffre.
  Si une donnée manque, appelle l'outil adapté ; si elle n'existe pas, dis-le simplement.
- Commence par le chiffre clé qui répond à la question (temps prédit, allure, moyenne de sommeil, écart de VFC,
  charge...), recopié tel qu'il figure dans les données, puis justifie ton conseil avec 1 ou 2 autres chiffres.
- Tu n'es pas médecin : ne pose jamais de diagnostic. En cas de douleur, si DONNÉES UTILES contient « douleur »,
  cite 2 ou 3 exercices de cette liste (nom et dosage), rappelle de ne pas dépasser une douleur de 3 sur 10,
  donne les signaux qui doivent faire consulter rapidement, et conseille de voir le spécialiste indiqué.
  En cas de malaise ou de symptôme inquiétant : arrêter et consulter.
- Tu ne modifies rien toi-même : tu proposes, l'utilisateur valide d'un bouton. N'écris jamais « c'est fait ».

Modifier le planning : si l'utilisateur demande de changer une séance (jour, intensité, durée, repos), ou si
c'est utile (douleur, fatigue, mauvaise nuit), propose UNE modification, explique-la en une ou deux phrases et
demande-lui de la valider. Termine alors ta réponse par une dernière ligne, exactement au format :
[[PROPOSITION]] {{"date": "AAAA-MM-JJ", "action": "deplacer", "nouvelle_date": "AAAA-MM-JJ", "raison": "..."}}
- date : celle de la séance concernée, prise dans prochaines_seances ; aide-toi du calendrier pour les jours ;
- action : deplacer (avec nouvelle_date), alleger, intensifier, raccourcir ou allonger (avec "facteur", ex. 0.7
  ou 1.2), repos ;
- pas de ligne [[PROPOSITION]] si aucune modification n'est utile.
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

    def _payload(self, messages: list[dict], tools: list[dict] | None, stream: bool) -> dict:
        payload = {"model": self.model, "messages": messages, "stream": stream, "options": OLLAMA_OPTIONS,
                   "keep_alive": KEEP_ALIVE}
        if tools:
            payload["tools"] = tools
        return payload

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        response = requests.post(f"{self.url}/api/chat", json=self._payload(messages, tools, False), timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]

    def stream(self, messages: list[dict]):
        """Réponse mot à mot. Le dernier élément est un dictionnaire de mesures (durée, vitesse)."""
        with requests.post(f"{self.url}/api/chat", json=self._payload(messages, None, True),
                           stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
                if chunk.get("done"):
                    yield metrics(chunk)

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


def metrics(done_chunk: dict) -> dict:
    """Mesures renvoyées par Ollama en fin de réponse (durées en nanosecondes)."""
    total = done_chunk.get("total_duration", 0) / 1e9
    tokens, gen = done_chunk.get("eval_count", 0), done_chunk.get("eval_duration", 0) / 1e9
    return {"duree_s": round(total, 1), "jetons": tokens, "jetons_par_s": round(tokens / gen, 1) if gen else None,
            "contexte_jetons": done_chunk.get("prompt_eval_count")}


class FakeLLM:
    """Faux modèle déterministe : consulte les allures, puis répond en citant ce qu'il a lu (tests, CI)."""

    model = "faux-modele"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        tool_results = [m for m in messages if m["role"] == "tool"]
        if tools and not tool_results:
            return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "allures", "arguments": {}}}]}
        seen = ", ".join(m.get("tool_name", "?") for m in tool_results) or "contexte"
        return {"role": "assistant", "content": f"Réponse de test fondée sur : {seen}."}

    def stream(self, messages: list[dict]):
        data = [m["content"] for m in messages if m["role"] == "system"][0]
        topics = data.split("DONNÉES UTILES :")[-1][:120] if "DONNÉES UTILES" in data else "contexte"
        for word in f"Réponse de test fondée sur : {topics.strip()[:60]}.".split(" "):
            yield word + " "
        question = _normalize(messages[-1]["content"])
        upcoming = re.findall(r'"date": "(\d{4}-\d{2}-\d{2})", "jour"', data)
        if upcoming and any(w in question for w in ("allege", "decale", "repos")):
            action = "repos" if "repos" in question else "alleger"
            proposal = {"date": upcoming[0], "action": action, "raison": "demande de l'utilisateur"}
            if "decale" in question:
                proposal = {"date": upcoming[0], "action": "deplacer", "nouvelle_date": "AUTRE", "raison": "test"}
            yield " Tu valides ?\n[[PROP"  # le marqueur arrive en plusieurs morceaux, comme avec un vrai modèle
            yield "OSITION]] " + json.dumps(proposal)
        yield {"duree_s": 0.1, "jetons": 8, "jetons_par_s": 80.0, "contexte_jetons": 100}

    def status(self) -> dict:
        return {"disponible": True, "modele": self.model, "url": "local", "erreur": None}


def get_llm():
    return FakeLLM() if os.getenv("RUNLAB_LLM") == "fake" else OllamaLLM()


# --- Contexte et boucle d'outils -------------------------------------------------------------------

def compact_context(forme: dict, analyse: dict, plan: dict, goals: list[dict], today: str | None = None) -> dict:
    """Ce qu'un coach doit savoir d'emblée, en quelques centaines de mots."""
    from datetime import date, timedelta

    day0 = date.fromisoformat(today) if today else None
    upcoming = [s for w in plan.get("semaines", []) for s in w["seances"]
                if not s.get("passee") and (day0 is None or s["date"] >= today)][:4]
    active = next((g for g in goals if g.get("actif")), None)
    names = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    calendar = None if day0 is None else [f"{names[(day0 + timedelta(days=i)).weekday()]} "
                                          f"{(day0 + timedelta(days=i)).isoformat()}" for i in range(10)]
    return {
        "aujourd_hui": today,
        "calendrier": calendar,
        "date_des_donnees": forme.get("date"),
        "verdict_du_jour": {k: analyse["verdict"][k] for k in ("titre", "score", "explication")},
        "forme": {k: forme.get(k) for k in ("hrv_ecart_pct", "sommeil_h", "fc_repos", "fraicheur_tsb",
                                            "ajustement_chrono_pct")},
        "analyses": {k: analyse[k]["resume"] for k in ("charge", "forme", "recuperation", "sommeil", "stress",
                                                       "activites") if k in analyse},
        "objectif_actif": None if active is None else {k: active.get(k) for k in (
            "nom", "libelle", "date_course", "jours_restants", "temps_vise", "temps_predit", "statut")},
        "prochaines_seances": [{k: s.get(k) for k in ("date", "jour", "type", "titre", "description", "allure",
                                                      "fc_cible")}
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


# --- Mode direct : un seul appel, avec les données choisies d'après la question -------------------

PLAN_CHANGE_WORDS = ("decal", "deplac", "repos", "annul", "remplac", "allege", "alleg", "intens", "plus court",
                     "plus long", "raccourc", "allong", "supprim", "change", "modifi", "pas dispo", "peux pas courir",
                     "fatigue", "crev")

TOPIC_KEYWORDS = {
    "allures": ("allure", "footing", "endurance", " ef", "fraction", "tempo", "seuil", "vma", "rythme", "vite", "lent"),
    "predictions": ("temps", "chrono", "predi", "objectif", "semi", "marathon", "10 km", "5 km", "10k", "5k", "record"),
    "nutrition": ("mang", "boire", "bois", "gel", "nutrition", "hydrat", "ravito", "repas", "sucre", "eau", "glucide"),
    "seances": ("hier", "dernier", "derniere", "sortie", "fait", "realise"),
    "planning": ("semaine", "planning", "programme", "demain", "prochain", "seance", "ce soir", "aujourd",
                 "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", *PLAN_CHANGE_WORDS),
}


def _normalize(text: str) -> str:
    table = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
    return " " + text.lower().translate(table)


def detect_topics(question: str) -> list[str]:
    """Sujets de la question, repérés par mots-clés : au plus trois, pour garder un contexte court."""
    from processing.rehab import mentions_pain

    text = _normalize(question)
    topics = [topic for topic, words in TOPIC_KEYWORDS.items() if any(w in text for w in words)]
    if mentions_pain(question):  # une douleur passe avant tout le reste, avec le planning pour l'alléger
        topics = ["douleur", "planning"] + [t for t in topics if t != "planning"]
    return topics[:3]


def detect_distance(question: str) -> dict:
    """Distance évoquée dans la question, pour les prédictions et la nutrition."""

    text = _normalize(question)
    if "semi" in text:
        return {"distance": "semi"}
    if "marathon" in text:
        return {"distance": "marathon"}
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:km|k\b)", text)
    if match:
        km = float(match.group(1).replace(",", "."))
        presets = {5.0: "5k", 10.0: "10k"}
        return {"distance": presets[km]} if km in presets else {"distance_km": km}
    return {}


def gather_topic_data(user_message: str, fetchers: dict) -> tuple[list[str], dict]:
    """Sujets de la question et données correspondantes : exactement ce que le modèle lira (et ce qu'on vérifie)."""
    topics = detect_topics(user_message)
    extra = {}
    for topic in topics:
        try:
            extra[topic] = fetchers[topic](user_message)
        except Exception as exc:  # une donnée indisponible ne doit pas empêcher de répondre
            extra[topic] = {"erreur": str(exc)}
    return topics, extra


def run_coach_direct_stream(llm, user_message: str, context: dict, fetchers: dict, history: list[dict] | None = None):
    """Génère la réponse mot à mot, en un seul appel au modèle. Premier élément : la liste des sujets consultés."""
    topics, extra = gather_topic_data(user_message, fetchers)
    data = json.dumps(context, ensure_ascii=False)
    if extra:
        data += "\n\nDONNÉES UTILES : " + json.dumps(extra, ensure_ascii=False, default=str)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=data)}]
    messages += [m for m in (history or []) if m.get("role") in ("user", "assistant")][-4:]
    messages.append({"role": "user", "content": user_message})
    yield topics
    yield from llm.stream(messages)


def answer_direct(llm, user_message: str, context: dict, fetchers: dict, history: list[dict] | None = None) -> dict:
    """Version complète (non streamée) du mode direct, pour l'API classique et le bilan."""
    stream = run_coach_direct_stream(llm, user_message, context, fetchers, history)
    topics = next(stream)
    parts, stats = [], None
    for item in stream:
        if isinstance(item, dict):
            stats = item
        else:
            parts.append(item)
    from processing.adjustments import parse_proposal, split_marker

    text, hidden = split_marker("".join(parts).strip())
    return {"reponse": text.strip(), "outils_utilises": topics, "modele": getattr(llm, "model", "?"),
            "mesures": stats, "proposition_brute": parse_proposal(hidden), "texte_complet": "".join(parts).strip()}


# --- Commentaire d'une sortie ---------------------------------------------------------------------

RUN_PROMPT = """Commente ma sortie analysée dans DONNÉES UTILES, comme un coach qui me connaît, en 6 à 10 phrases :
1. ce qui s'est bien passé, chiffres à l'appui ;
2. les points d'attention (allure, dérive cardiaque, zones, dénivelé), en tenant compte de ma nuit précédente,
   de mon ressenti déclaré et de mon objectif ;
3. un conseil concret et chiffré pour ma prochaine séance du même type.
Compare avec mes sorties similaires quand c'est utile. Parle des allures équivalentes sur le plat quand le
parcours était vallonné. N'invente aucun chiffre absent des données."""


def run_commentary_stream(llm, run_data: dict, context: dict):
    """Commentaire personnalisé d'une sortie, mot à mot, en un seul appel au modèle."""
    data = json.dumps(context, ensure_ascii=False) + "\n\nDONNÉES UTILES : " + json.dumps(
        run_data, ensure_ascii=False, default=str)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=data)},
                {"role": "user", "content": RUN_PROMPT}]
    yield from llm.stream(messages)

