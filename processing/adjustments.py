"""Ajustements du planning proposés par le coach, appliqués seulement une fois validés par l'utilisateur.

Cycle de vie d'un ajustement (enregistré dans $RUNLAB_DATA_DIR/planning/ajustements.json) :
  propose  -> le coach l'a suggéré dans la discussion, rien ne change encore ;
  accepte  -> l'utilisateur a validé : le planning (et la montre) en tiennent compte ;
  refuse   -> l'utilisateur a dit non ;
  annule   -> validé puis annulé depuis le planning.

Actions possibles sur une séance à venir :
  deplacer    (nouvelle_date)            alleger      (séance dure -> footing facile ; footing -> plus court, plus lent)
  intensifier (footing -> tempo court)   raccourcir / allonger (facteur sur la distance et la durée)
  repos       (la séance est retirée)

Le code vérifie tout ce que le modèle propose (la séance existe, la date est libre et à venir, la course ne
bouge pas...) et signale les risques (deux séances dures d'affilée, jour de tennis, forme du jour).
"""

import copy
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from processing.planning import COOLDOWN_S, DAY_NAMES, HARD, WARMUP_S, fmt_pace, pace_range, step

ACTIONS = {"deplacer", "alleger", "intensifier", "raccourcir", "allonger", "repos"}
ACTION_NAMES = {"deplacer": "Déplacer", "alleger": "Alléger", "intensifier": "Intensifier",
                "raccourcir": "Raccourcir", "allonger": "Allonger", "repos": "Remplacer par du repos"}
FACTOR_BOUNDS = {"raccourcir": (0.4, 0.95, 0.7), "allonger": (1.05, 1.3, 1.15)}  # min, max, par défaut
MARKER = "[[PROPOSITION]]"


# --- Stockage --------------------------------------------------------------------------------------

def adjustments_file(data_dir: Path) -> Path:
    return data_dir / "planning" / "ajustements.json"


def load_adjustments(data_dir: Path) -> list[dict]:
    path = adjustments_file(data_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_adjustments(data_dir: Path, items: list[dict]) -> None:
    path = adjustments_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # écriture atomique


def set_status(data_dir: Path, adjustment_id: str, status: str) -> dict | None:
    items = load_adjustments(data_dir)
    found = next((a for a in items if a["id"] == adjustment_id), None)
    if found is None:
        return None
    found["statut"] = status
    found[f"{status}_le"] = datetime.now().isoformat(timespec="seconds")
    save_adjustments(data_dir, items)
    return found


# --- Lecture de la proposition écrite par le modèle -----------------------------------------------

def split_marker(text: str) -> tuple[str, str | None]:
    """Sépare le texte visible de la proposition technique (ligne [[PROPOSITION]] {...})."""
    cut = text.find("[[")
    if cut < 0:
        return text, None
    return text[:cut].rstrip(), text[cut:]


def parse_proposal(hidden: str | None) -> dict | None:
    """Le JSON qui suit [[PROPOSITION]], toléré dans un bloc de code ou avec des guillemets typographiques."""
    if not hidden or "PROPOSITION" not in hidden:
        return None
    match = re.search(r"\{.*\}", hidden.replace("“", '"').replace("”", '"'), re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class MarkerFilter:
    """Filtre un flux de texte : laisse passer la réponse, retient tout ce qui suit le premier « [[ »."""

    def __init__(self):
        self.hidden, self._buffer, self._hiding = "", "", False

    def feed(self, chunk: str) -> str:
        if self._hiding:
            self.hidden += chunk
            return ""
        self._buffer += chunk
        cut = self._buffer.find("[[")
        if cut >= 0:
            visible, self.hidden, self._hiding = self._buffer[:cut], self._buffer[cut:], True
            self._buffer = ""
            return visible.rstrip()
        # Garde en réserve un « [ » final (début possible du marqueur) et les blancs qui le précèdent
        keep = len(self._buffer) - len(self._buffer.rstrip("[ \n"))
        visible, self._buffer = self._buffer[:len(self._buffer) - keep], self._buffer[len(self._buffer) - keep:]
        return visible

    def flush(self) -> str:
        rest, self._buffer = self._buffer, ""
        return "" if self._hiding else rest


# --- Transformation d'une séance ------------------------------------------------------------------

def _scale_steps(steps: list[dict], factor: float) -> list[dict]:
    """Applique un facteur aux étapes d'effort (distance, durée, répétitions) ; échauffement et retour inchangés."""
    out = []
    for s in steps:
        s = copy.deepcopy(s)
        if s["type"] == "repetition":
            s["repetitions"] = max(1, round(s["repetitions"] * factor))
        elif s["type"] == "effort":
            if s.get("distance_m"):
                s["distance_m"] = round(s["distance_m"] * factor)
            if s.get("duree_s"):
                s["duree_s"] = round(s["duree_s"] * factor)
        out.append(s)
    return out


def transform(session: dict, adjustment: dict, paces: dict) -> dict | None:
    """Séance après ajustement (None : séance retirée, jour de repos)."""
    action = adjustment["action"]
    s = copy.deepcopy(session)
    ef = paces.get("ef", {})
    tempo = paces.get("tempo", {})
    if action == "repos":
        return None
    if action == "deplacer":
        new_day = date.fromisoformat(adjustment["nouvelle_date"])
        s.update({"date": new_day.isoformat(), "jour": DAY_NAMES[new_day.weekday()]})
    elif action == "alleger":
        km = max(4.0, round(s["distance_km"] * 0.7, 1))
        slow = {"rapide_s": (ef.get("lente_s") or 360), "lente_s": (ef.get("lente_s") or 360) + 30}
        s.update({"type": "ef", "titre": "Footing facile (allégé)", "distance_km": km,
                  "duree_min": round(km * (slow["rapide_s"] + 15) / 60),
                  "description": f"{km:g} km très faciles, sans forcer, à la place de « {session['titre']} ».",
                  "allure": f"{fmt_pace(slow['rapide_s'])} ou plus lent", "fc_cible": s.get("fc_cible"),
                  "objectif": "Récupérer en gardant l'habitude de courir.",
                  "etapes": [step("effort", distance_m=km * 1000, allure=slow)]})
    elif action == "intensifier":
        tempo_min = 15
        km = s["distance_km"]
        s.update({"type": "tempo", "titre": "Footing avec bloc tempo",
                  "description": f"Échauffement 15 min, puis {tempo_min} min à allure tempo, retour au calme 10 min "
                                 f"(environ {km:g} km au total).",
                  "allure": pace_range(tempo) if tempo else s.get("allure"),
                  "fc_cible": None if not tempo.get("fc_cible") else f"{tempo['fc_cible'][0]}–{tempo['fc_cible'][1]} bpm",
                  "objectif": "Un peu de rythme, sans en faire une séance complète.",
                  "etapes": [step("echauffement", duree_s=WARMUP_S, allure=ef),
                             step("effort", duree_s=tempo_min * 60, allure=tempo),
                             step("retour_au_calme", duree_s=COOLDOWN_S, allure=ef)]})
    else:  # raccourcir / allonger
        factor = adjustment["facteur"]
        s["distance_km"] = round(s["distance_km"] * factor, 1)
        if s.get("duree_min"):
            s["duree_min"] = round(s["duree_min"] * factor)
        s["etapes"] = _scale_steps(s.get("etapes", []), factor)
        word = "raccourcie" if factor < 1 else "allongée"
        s["description"] = f"Version {word} ({s['distance_km']:g} km) : {session['description']}"
    s["ajustement"] = {"id": adjustment["id"], "libelle": describe(adjustment, session)}
    return s


def describe(adjustment: dict, session: dict) -> str:
    """Phrase lisible : « Déplacer Tempo du mardi 06/10 au mercredi 07/10 »."""
    day = date.fromisoformat(session["date"])
    when = f"{DAY_NAMES[day.weekday()].lower()} {day.strftime('%d/%m')}"
    action = adjustment["action"]
    if action == "deplacer":
        new = date.fromisoformat(adjustment["nouvelle_date"])
        return f"Déplacer « {session['titre']} » du {when} au {DAY_NAMES[new.weekday()].lower()} {new.strftime('%d/%m')}"
    if action in FACTOR_BOUNDS:
        pct = round((adjustment["facteur"] - 1) * 100)
        return f"{ACTION_NAMES[action]} « {session['titre']} » du {when} ({pct:+d} %)"
    return f"{ACTION_NAMES[action]} : « {session['titre']} » du {when}"


# --- Validation d'une proposition -----------------------------------------------------------------

def _sessions(plan: dict) -> list[dict]:
    return [s for w in plan.get("semaines", []) for s in w["seances"]]


def validate(raw: dict, plan: dict, today: date, tennis_days: set[int] | None = None,
             verdict: str = "vert", paces: dict | None = None) -> tuple[dict | None, str | None]:
    """Vérifie une proposition du modèle. Renvoie (ajustement prêt à enregistrer, None) ou (None, raison du refus)."""
    action = str(raw.get("action", "")).strip().lower()
    if action not in ACTIONS:
        return None, f"Action inconnue : « {action} »."
    try:
        day = date.fromisoformat(str(raw.get("date")))
    except ValueError:
        return None, "Date de séance illisible."
    sessions = _sessions(plan)
    session = next((s for s in sessions if s["date"] == day.isoformat()), None)
    if session is None:
        return None, f"Aucune séance prévue le {day.strftime('%d/%m')}."
    if day < today:
        return None, "Cette séance est déjà passée."
    if session["type"] == "course":
        return None, "Le jour de course ne se modifie pas depuis la discussion : change ton objectif."

    adjustment = {"id": uuid.uuid4().hex[:8], "date": day.isoformat(), "action": action,
                  "raison": str(raw.get("raison", "")).strip()[:300], "statut": "propose",
                  "cree_le": datetime.now().isoformat(timespec="seconds")}
    warnings = []
    new_day = day
    if action == "deplacer":
        try:
            new_day = date.fromisoformat(str(raw.get("nouvelle_date")))
        except ValueError:
            return None, "Nouvelle date illisible."
        if new_day < today:
            return None, "On ne peut pas déplacer une séance dans le passé."
        if new_day == day:
            return None, "La nouvelle date est la même que l'ancienne."
        last = max(s["date"] for s in sessions)
        if new_day.isoformat() > last:
            return None, "Cette date est au-delà du planning."
        if any(s["date"] == new_day.isoformat() for s in sessions):
            return None, f"Il y a déjà une séance le {new_day.strftime('%d/%m')} : propose un autre jour."
        if any(s["type"] == "course" and abs((date.fromisoformat(s["date"]) - new_day).days) <= 2 for s in sessions):
            warnings.append("C'est très proche du jour de course.")
        adjustment["nouvelle_date"] = new_day.isoformat()
    elif action in FACTOR_BOUNDS:
        low, high, default = FACTOR_BOUNDS[action]
        try:
            factor = float(raw.get("facteur", default))
        except (TypeError, ValueError):
            factor = default
        adjustment["facteur"] = round(min(max(factor, low), high), 2)
    elif action == "intensifier":
        if session["type"] in HARD:
            return None, "C'est déjà une séance intense : propose plutôt de l'allonger."
        if verdict != "vert":
            warnings.append("Ta forme du jour n'est pas au vert : intensifier est risqué.")
    elif action == "alleger" and session["type"] not in HARD and session["distance_km"] <= 4:
        return None, "Cette séance est déjà très légère : propose plutôt du repos."

    # Risques après ajustement : séance dure entourée d'autres séances dures, jour de tennis
    preview = transform(session, adjustment, paces or {})
    if preview is not None and preview["type"] in HARD:
        others = {s["date"] for s in sessions if s["type"] in HARD and s["date"] != day.isoformat()}
        around = {(new_day + timedelta(days=d)).isoformat() for d in (-1, 1)}
        if others & around:
            warnings.append("Elle se retrouve à côté d'une autre séance dure : pense à bien récupérer.")
    if preview is not None and tennis_days and new_day.weekday() in tennis_days:
        warnings.append("C'est un jour de tennis.")

    adjustment["avertissements"] = warnings
    adjustment["libelle"] = describe(adjustment, session)
    adjustment["avant"] = {k: session.get(k) for k in ("date", "jour", "titre", "distance_km", "allure")}
    adjustment["apres"] = None if preview is None else {k: preview.get(k) for k in
                                                         ("date", "jour", "titre", "distance_km", "allure")}
    return adjustment, None


# --- Application au planning ----------------------------------------------------------------------

def apply_adjustments(plan: dict, adjustments: list[dict], paces: dict, today: date) -> dict:
    """Planning après les ajustements VALIDÉS, dans l'ordre où ils ont été proposés."""
    plan = copy.deepcopy(plan)
    weeks = plan.get("semaines", [])
    applied, removed = [], []
    for adj in sorted((a for a in adjustments if a["statut"] == "accepte"), key=lambda a: a["cree_le"]):
        week = next((w for w in weeks if any(s["date"] == adj["date"] for s in w["seances"])), None)
        if week is None:
            continue  # la séance n'existe plus (planning recalculé, objectif changé) : ajustement sans effet
        index = next(i for i, s in enumerate(week["seances"]) if s["date"] == adj["date"])
        session = week["seances"].pop(index)
        if session.get("passee"):  # séance passée : on n'y touche plus
            week["seances"].insert(index, session)
            continue
        new = transform(session, adj, paces)
        applied.append(adj["id"])
        if new is None:
            removed.append(describe(adj, session))
            continue
        new["passee"] = new["date"] < today.isoformat()
        target = next((w for w in weeks if w["debut"] <= new["date"] < (date.fromisoformat(w["debut"])
                                                                        + timedelta(days=7)).isoformat()), week)
        target["seances"].append(new)
        target["seances"].sort(key=lambda s: s["date"])
    for w in weeks:
        w["volume_km"] = round(sum(s["distance_km"] for s in w["seances"]), 1)
    plan["ajustements_appliques"] = applied
    if removed:
        plan.setdefault("notes", []).extend(f"Repos décidé avec le coach : {r.split(' : ', 1)[-1]}." for r in removed)
    return plan


# --- Secours : la demande explicite de l'utilisateur, quand le modèle n'a pas écrit de proposition -----------

_WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_REQUESTS = [  # de la plus spécifique à la plus générale
    ("deplacer", ("decal", "deplac", "repousse", "reporte", "avance")),
    ("repos", ("repos", "supprim", "annul", "enleve", "retire", "saute")),
    ("raccourcir", ("raccourc", "plus court", "moins long", "reduis", "reduire")),
    ("allonger", ("allong", "plus long", "rallonge")),
    ("intensifier", ("intensif", "plus dur", "plus intense", "plus rapide")),
    ("alleger", ("alleg", "plus leger", "plus facile", "plus cool", "tranquille")),
]
_KINDS = {"longue": ("sortie longue",), "tempo": ("tempo", "seuil"), "fractionne": ("fraction", "vma", "interval"),
          "ef": ("footing", "endurance"), "specifique": ("allure course", "specifique")}


def _plain(text: str) -> str:
    table = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
    return " " + text.lower().translate(table).replace("’", "'") + " "


def _next_weekday(start: date, weekday: int) -> date:
    return start + timedelta(days=(weekday - start.weekday()) % 7)


def requested_action(question: str) -> str | None:
    """Modification explicitement demandée dans la question (« décale », « repos », « allège »...), ou None."""
    text = _plain(question)
    return next((a for a, words in _REQUESTS if any(w in text for w in words)), None)


def fallback_proposal(question: str, plan: dict, today: date) -> dict | None:
    """Traduit une demande explicite (« décale ma sortie longue à samedi », « repos demain ») en proposition.

    Ne sert qu'en secours : si le modèle a écrit une proposition valide, c'est elle qui compte.
    Sans verbe de modification clair, ou sans séance identifiable, rien n'est proposé.
    """
    text = _plain(question)
    action = requested_action(question)
    if action is None:
        return None
    upcoming = [s for s in _sessions(plan) if s["date"] >= today.isoformat() and s["type"] != "course"]
    if not upcoming:
        return None

    # La séance visée : type nommé, jour nommé, « demain », « aujourd'hui », sinon la prochaine
    target_part = re.split(r"\s(?:a|au|vers|pour)\s(?=(?:" + "|".join(_WEEKDAYS) + r"|demain|lendemain))", text)[0] \
        if action == "deplacer" else text
    source = None
    for kind, words in _KINDS.items():
        if any(w in target_part for w in words):
            source = next((s for s in upcoming if s["type"] == kind), None)
            break
    if source is None:
        day = None
        if "aujourd" in target_part or "ce soir" in target_part or "ce matin" in target_part:
            day = today
        elif re.search(r"(?<!apres-)(?<!apres )demain", target_part) and "lendemain" not in target_part:
            day = today + timedelta(days=1)
        else:
            named = [i for i, w in enumerate(_WEEKDAYS) if re.search(r"\b" + w + r"\b", target_part)]
            if named:
                day = _next_weekday(today, named[0])
        if day is not None:
            source = next((s for s in upcoming if s["date"] == day.isoformat()), None)
            if source is None:
                return None  # le jour nommé n'a pas de séance : mieux vaut ne rien proposer
    source = source or upcoming[0]
    proposal = {"date": source["date"], "action": action, "raison": "Ta demande dans la discussion."}

    if action == "deplacer":
        start = date.fromisoformat(source["date"])
        # la destination suit « à / au / vers / pour » ; sans elle, seuls « lendemain » et « veille » sont compris
        rest = text[len(target_part):] if len(target_part) < len(text) else (
            text if ("lendemain" in text or "veille" in text) else "")
        if "lendemain" in rest:
            new_day = start + timedelta(days=1)
        elif "veille" in rest:
            new_day = start - timedelta(days=1)
        else:
            named = [i for i, w in enumerate(_WEEKDAYS) if re.search(r"\b" + w + r"\b", rest)]
            if named:
                new_day = _next_weekday(today, named[-1])
            elif "demain" in rest:
                new_day = today + timedelta(days=1)
            else:
                return None  # déplacer, mais où ? le coach doit poser la question
        proposal["nouvelle_date"] = new_day.isoformat()
    return proposal
