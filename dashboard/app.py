"""Foulée, le tableau de bord : l'interface ne lit jamais les données, elle interroge l'API.

Lancement local (API démarrée sur le port 8000) :  streamlit run dashboard/app.py
Sans API séparée :  $env:RUNLAB_API_URL = "inprocess"; streamlit run dashboard/app.py
"""

import html
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("RUNLAB_API_URL", "http://127.0.0.1:8000")
AUTO_SYNC = os.getenv("RUNLAB_AUTO_SYNC", "1") == "1"  # synchronisation à l'ouverture si les données ont vieilli
AUTO_SYNC_AFTER_MIN = 30
SLOW_SYNC_S = 90  # au-delà, on explique que l'attente vient de Garmin
DISTANCES = {"5 km": "5k", "10 km": "10k", "Semi": "semi", "Marathon": "marathon"}
DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

ENCRE, GRIS, FILET, ACCENT = "#14181F", "#6B7280", "#E3E5E8", "#1F5C4A"
OK, VIGILANCE, ALERTE = "#1F7A5A", "#B7791F", "#B4432F"
SPORT_COLORS = {"running": ENCRE, "tennis": ACCENT, "strength": "#9AA1AB", "walking": "#C9CDD3",
                "hiking": "#B5BAC2", "other": "#DADDE2"}
SPORT_NAMES = {"running": "Course", "tennis": "Tennis", "strength": "Musculation", "walking": "Marche",
               "hiking": "Randonnée", "other": "Autre"}
TYPE_COLORS = {"ef": "#9AA1AB", "longue": ENCRE, "tempo": VIGILANCE, "fractionne": ALERTE,
               "specifique": ACCENT, "course": ENCRE}
TYPE_NAMES = {"ef": "Endurance fondamentale", "longue": "Sortie longue", "tempo": "Tempo", "fractionne": "Fractionné",
              "specifique": "Allure course", "course": "Course"}
LEVEL_COLORS = {"vert": OK, "ambre": VIGILANCE, "rouge": ALERTE}

st.set_page_config(page_title="Foulée", page_icon="🏃", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# --- Accès à l'API -------------------------------------------------------------------------------

@st.cache_resource
def inprocess_client():
    import sys

    from fastapi.testclient import TestClient

    # « streamlit run » n'ajoute que le dossier dashboard/ au chemin Python : on ajoute la racine du dépôt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.main import app

    return TestClient(app)


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, params: tuple = ()) -> dict:
    if API_URL == "inprocess":
        response = inprocess_client().get(path, params=dict(params))
    else:
        response = requests.get(f"{API_URL}{path}", params=dict(params), timeout=30)
    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.status_code
        raise RuntimeError(str(detail))
    return response.json()


COACH_TIMEOUT_S = 330  # un modèle local sur processeur peut mettre plusieurs minutes à répondre


def post(path: str, payload: dict, timeout: int = 30) -> tuple[bool, dict]:
    """Envoie des données à l'API (questionnaire, coach). Renvoie (succès, réponse)."""
    try:
        if API_URL == "inprocess":
            response = inprocess_client().post(path, json=payload)
        else:
            response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    except requests.ConnectionError:
        return False, {"detail": f"L'API ne répond pas à l'adresse {API_URL}."}
    except requests.Timeout:
        return False, {"detail": "Le coach a mis trop de temps à répondre. Réessaie, ou choisis un modèle plus léger."}
    return response.status_code == 200, response.json()


def get_slow(path: str, **params) -> tuple[bool, dict]:
    """Lecture longue et non mise en cache (bilan du coach)."""
    try:
        if API_URL == "inprocess":
            response = inprocess_client().get(path, params=params)
        else:
            response = requests.get(f"{API_URL}{path}", params=params, timeout=COACH_TIMEOUT_S)
    except (requests.ConnectionError, requests.Timeout) as exc:
        return False, {"detail": str(exc)}
    return response.status_code == 200, response.json()


def delete(path: str) -> bool:
    if API_URL == "inprocess":
        return inprocess_client().delete(path).status_code == 200
    try:
        return requests.delete(f"{API_URL}{path}", timeout=30).status_code == 200
    except requests.ConnectionError:
        return False


def get(path: str, **params) -> dict | None:
    try:
        return api_get(path, tuple(sorted((k, v) for k, v in params.items() if v is not None)))
    except requests.ConnectionError:
        st.error(f"L'API ne répond pas à l'adresse {API_URL}. Lance-la avec : uvicorn api.main:app")
    except RuntimeError as exc:
        st.warning(f"Données indisponibles : {exc}")
    return None


# --- Éléments visuels réutilisables --------------------------------------------------------------

def esc(text) -> str:
    return html.escape(str(text))


def hero(legend: str, value: str, details: list[tuple[str, str]]) -> None:
    """L'élément fort de la page : un grand temps en chiffres fins, et ses détails en dessous."""
    cells = "".join(f"<div>{esc(label)}<b>{esc(v)}</b></div>" for label, v in details)
    st.markdown(f'<div class="chrono"><div class="legende">{esc(legend)}</div><div class="temps">{esc(value)}</div>'
                f'<div class="details">{cells}</div></div>', unsafe_allow_html=True)


def verdict_card(verdict: dict) -> None:
    color = LEVEL_COLORS[verdict["niveau"]]
    st.markdown(f"""<div class="verdict">
        <div class="etat"><span class="point" style="background:{color}"></span>{esc(verdict['titre'])}</div>
        <div class="jauge"><span style="width:{verdict['score']}%;background:{color}"></span></div>
        <div class="score">Indice de forme du jour : {verdict['score']} sur 100</div>
        <p class="conseil">{esc(verdict['conseil'])}</p>
        <p class="raison">{esc(verdict['explication'])}</p></div>""", unsafe_allow_html=True)


def sessions_table(sessions: list[dict]) -> None:
    """Carnet d'entraînement : une séance par ligne, colonnes alignées."""
    rows = []
    for s in sessions:
        color = TYPE_COLORS.get(s["type"], GRIS)
        when = date.fromisoformat(s["date"]).strftime("%d/%m")
        figures = f"{s['distance_km']:g} km".replace(".", ",") + (f", {s['duree_min']} min" if s.get("duree_min") else "")
        detail = s["allure"] if s.get("allure") and s["allure"] != "—" else ""
        if s.get("fc_cible"):
            detail += f"<br>FC {esc(s['fc_cible'])}"
        if s.get("passee"):
            done = s.get("realisee")
            detail = ("Réalisée" if done else "Non réalisée" if done is False else "Passée") + (f"<br>{detail}" if detail else "")
        row_class = ' class="passee"' if s.get("passee") else ""
        badge = '<span class="modifiee">Modifiée avec ton coach</span>' if s.get("ajustement") else ""
        rows.append(f"""<tr{row_class}><td class="jour"><b>{esc(s['jour'])}</b><span>{when}</span></td>
            <td class="type"><i style="background:{color}"></i>{esc(TYPE_NAMES.get(s['type'], s['type']))}</td>
            <td><div class="titre">{esc(s['titre'])}</div><div class="description">{esc(s['description'])}</div>
            <div class="objectif">{esc(s['objectif'])}</div>{badge}</td>
            <td class="chiffres">{esc(figures)}<span>{detail}</span></td></tr>""")
    st.markdown(f'<table class="carnet">{"".join(rows)}</table>', unsafe_allow_html=True)


def explain(how_to_read: str, insight: dict | None) -> None:
    left, right = st.columns(2, gap="large")
    left.markdown(f'<div class="lecture"><h4>Comment le lire</h4><p>{how_to_read}</p></div>', unsafe_allow_html=True)
    points = "".join(f"<li>{esc(p)}</li>" for p in (insight or {}).get("points", [])) or "<li>Pas assez de données.</li>"
    right.markdown(f'<div class="donnees"><h4>Ce que disent tes données</h4><ul>{points}</ul></div>',
                   unsafe_allow_html=True)


def resume(title: str, text: str) -> str:
    return f'<div class="resume"><h4>{esc(title)}</h4><p>{esc(text)}</p></div>'


def figure(fig: go.Figure, height: int = 320) -> None:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=36, b=0), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Archivo, Helvetica Neue, Arial, sans-serif", color=ENCRE, size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=GRIS)),
                      hovermode="x unified", hoverlabel=dict(bgcolor="white", font_family="Archivo"), bargap=0.35)
    fig.update_xaxes(showgrid=False, linecolor=FILET, tickfont=dict(color=GRIS), tickformat="%d/%m", hoverformat="%d/%m/%Y")
    fig.update_yaxes(gridcolor="#EEF0F2", zeroline=False, tickfont=dict(color=GRIS), title_font=dict(color=GRIS))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def history(days: int) -> pd.DataFrame | None:
    data = get("/historique", jours=days)
    if not data:
        return None
    df = pd.DataFrame(data["series"])
    df["date"] = pd.to_datetime(df["date"])
    df.attrs["sports"] = data["sports"]
    return df.set_index("date")


# --- Questionnaire après sortie ------------------------------------------------------------------

def questionnaire_card() -> None:
    thanks = st.session_state.pop("merci_questionnaire", None)
    if thanks:  # message conservé après le rafraîchissement automatique
        st.success(thanks)
    data = get("/questionnaire")
    run = (data or {}).get("en_attente")
    if not run:
        return
    when = date.fromisoformat(run["date"]).strftime("%d/%m")
    title = f"Ta sortie du {when} : {run['distance_km']:g} km en {run['duree_min']} min ({run['allure']})".replace(".", ",")
    with st.container(border=True):
        st.subheader(f"{len(run['questions'])} questions sur ta dernière sortie")
        st.caption(title + ". Tes réponses corrigent le type de séance, signalent les douleurs et "
                   "améliorent tes allures, ton planning et tes prédictions.")
        with st.form("questionnaire"):
            answers = {q["id"]: st.radio(q["question"], q["options"], index=None, key=f"q_{q['id']}")
                       for q in run["questions"]}
            sent = st.form_submit_button("Envoyer mes réponses", type="primary")
        if sent:
            if None in answers.values():
                st.warning("Réponds à toutes les questions avant d'envoyer.")
            else:
                ok, body = post(f"/questionnaire/{run['activity_id']}", {"reponses": answers})
                if ok:
                    api_get.clear()  # prédictions, allures et planning sont recalculés avec tes réponses
                    st.session_state["merci_questionnaire"] = body["message"]
                    st.rerun()  # le questionnaire disparaît (ou laisse place au suivant)
                else:
                    st.error(f"Réponses non enregistrées : {body.get('detail')}")


# --- Montre --------------------------------------------------------------------------------------

def watch_block() -> None:
    data = get("/montre/seance-du-jour")
    if data is None:
        return
    st.header("Ta montre")
    session = data["seance"]
    if session is None:
        st.markdown(resume("Aujourd'hui", "Pas de séance de course prévue : rien à envoyer sur ta montre."),
                    unsafe_allow_html=True)
        return
    sent = data["envois"].get(session["date"])
    left, right = st.columns([3, 1], gap="large")
    status = (f"Déjà dans ton calendrier Garmin : {sent['titre']}." if sent
              else "Pas encore envoyée. Elle part automatiquement chaque matin à 6 h si le cluster tourne.")
    left.markdown(resume(f"Séance du jour : {session['titre']}", status), unsafe_allow_html=True)
    if right.button("Envoyer sur ma montre", use_container_width=True):
        ok, body = post("/montre/envoyer", {})
        if ok:
            api_get.clear()
            st.success(body["message"] + " Synchronise ta montre avec l'application Garmin Connect pour la voir.")
        else:
            st.error(f"Envoi impossible : {body.get('detail')}")


# --- Accueil -------------------------------------------------------------------------------------

def page_home() -> None:
    forme, analyse = get("/forme"), get("/analyse")
    preds = get("/predictions", distance="10k", ajuster_au_jour=True)
    if not (forme and analyse and preds):
        return
    day = date.fromisoformat(forme["date"]).strftime("%d/%m/%Y")
    st.title("Ta journée")
    st.caption(f"Données à jour au {day}")
    questionnaire_card()

    left, right = st.columns([6, 5], gap="large")
    with left:
        p = preds["predictions"]["10k"]
        hero("Ton 10 km si tu courais aujourd'hui", p["temps_ajuste"],
             [("Allure", p["allure_course"]), ("Sur le plat, au repos", p["temps_base"]),
              ("Prédiction de la montre", p["prediction_montre"] or "—")])
    with right:
        verdict_card(analyse["verdict"])

    st.header("Prochaines séances")
    plan = get("/planning/actif")
    upcoming = [s for w in (plan or {}).get("semaines", []) for s in w["seances"] if not s.get("passee")][:3]
    if upcoming:
        sessions_table(upcoming)
    goal = (plan or {}).get("objectif_actif")
    if goal:
        race_day = date.fromisoformat(goal["date_course"]).strftime("%d/%m/%Y")
        st.caption(f"Préparation : {goal['nom']}, le {race_day}.")
    else:
        st.caption("Planning général sur deux semaines. Crée un objectif dans l'onglet Objectifs "
                   "pour une préparation sur mesure.")
    watch_block()

    st.header("Ta semaine")
    c1, c2, c3 = st.columns(3, gap="large")
    for col, key, title in ((c1, "charge", "Charge"), (c2, "forme", "Forme"), (c3, "recuperation", "Récupération")):
        col.markdown(resume(title, analyse[key]["resume"]), unsafe_allow_html=True)

    st.header("Les chiffres du jour")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VFC de la nuit", f"{forme['hrv']:.0f} ms" if forme["hrv"] else "—",
              None if forme["hrv_ecart_pct"] is None else f"{forme['hrv_ecart_pct']:+.0f} % vs ta normale")
    sleep_delta = None if None in (forme["sommeil_h"], forme["sommeil_normal_h"]) else \
        f"{forme['sommeil_h'] - forme['sommeil_normal_h']:+.1f} h vs ta normale"
    m2.metric("Sommeil", f"{forme['sommeil_h']:.1f} h" if forme["sommeil_h"] else "—", sleep_delta)
    m3.metric("FC de repos", f"{forme['fc_repos']:.0f} bpm" if forme["fc_repos"] else "—")
    m4.metric("Fraîcheur", f"{forme['fraicheur_tsb']:+.0f}" if forme["fraicheur_tsb"] is not None else "—",
              help="Positive : reposé. Négative : plus fatigué que d'habitude.")


# --- Ma forme : graphiques expliqués -------------------------------------------------------------

def page_fitness() -> None:
    st.title("Ma forme")
    days = st.select_slider("Période affichée", [30, 60, 90, 180, 365], value=90, format_func=lambda d: f"{d} jours")
    df, analyse = history(days), get("/analyse")
    if df is None or analyse is None:
        return

    st.header("Charge d'entraînement")
    weekly = df[[f"load_{s}" for s in df.attrs["sports"]]].resample("W").sum()
    fig = go.Figure()
    for sport in df.attrs["sports"]:
        fig.add_bar(x=weekly.index, y=weekly[f"load_{sport}"].round(0), name=SPORT_NAMES.get(sport, sport),
                    marker_color=SPORT_COLORS.get(sport, "#C9CCD8"))
    fig.update_layout(barmode="stack", yaxis_title="TRIMP par semaine")
    figure(fig)
    explain("Chaque barre est une semaine. Sa hauteur est ta charge totale, mesurée en TRIMP : la durée de tes séances "
            "pondérée par leur intensité cardiaque. Les couleurs montrent la part de chaque sport. Une hausse brutale "
            "d'une semaine à l'autre augmente le risque de blessure ; une progression régulière fait progresser.",
            analyse["charge"])

    st.header("Forme de fond et fatigue")
    fig = go.Figure()
    fig.add_bar(x=df.index, y=df["tsb"].round(1), name="Fraîcheur (TSB)",
                marker_color=[OK if v >= 0 else (VIGILANCE if v > -0.3 * c else ALERTE) for v, c in zip(df["tsb"], df["ctl"])],
                opacity=0.35)
    fig.add_scatter(x=df.index, y=df["ctl"].round(1), name="Forme de fond (CTL)", line=dict(color=ACCENT, width=2.5))
    fig.add_scatter(x=df.index, y=df["atl"].round(1), name="Fatigue récente (ATL)", line=dict(color=GRIS, width=1.5))
    figure(fig, 360)
    explain("La ligne verte est ta forme de fond : la moyenne de ta charge sur environ 6 semaines. La ligne grise est "
            "ta fatigue récente, sur environ 1 semaine. Les barres montrent leur différence, ta fraîcheur : vertes quand "
            "tu es reposé, ocre quand tu accumules un peu de fatigue (normal en préparation), brique quand la fatigue "
            "devient importante. Avant une course, on cherche une forme de fond haute et des barres vertes.",
            analyse["forme"])

    st.header("Récupération")
    base = df["hrv_last_night"].rolling(28, min_periods=7).mean()
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=(base * 1.1).round(1), line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=df.index, y=(base * 0.9).round(1), fill="tonexty", fillcolor="rgba(20,24,31,.06)",
                    line=dict(width=0), name="Ta zone normale (±10 %)", hoverinfo="skip")
    low = df["hrv_last_night"] < base * 0.9
    fig.add_scatter(x=df.index, y=df["hrv_last_night"], mode="markers", name="VFC de la nuit",
                    marker=dict(color=[ALERTE if x else ENCRE for x in low], size=6))
    fig.add_scatter(x=df.index, y=df["hrv_7j"].round(1), name="Moyenne 7 jours", line=dict(color=ACCENT, width=2))
    fig.update_layout(yaxis_title="VFC (ms)")
    figure(fig, 360)
    explain("Chaque point est la variabilité de ta fréquence cardiaque (VFC) pendant une nuit : plus elle est haute par "
            "rapport à ta normale, mieux tu récupères. La bande grise est ta zone normale, calculée sur tes 4 dernières "
            "semaines. Un point brique est une nuit nettement sous ta normale : fatigue, stress, alcool ou maladie. Une "
            "seule nuit basse n'est pas grave ; plusieurs d'affilée méritent d'alléger l'entraînement.",
            analyse["recuperation"])

    st.header("Sommeil et fréquence cardiaque de repos")
    fig = go.Figure()
    fig.add_bar(x=df.index, y=df["sleep_h"].round(1), name="Sommeil (h)",
                marker_color=[ALERTE if (v or 0) < 6 else (ACCENT if (v or 0) >= 7.5 else "#C9CDD3") for v in df["sleep_h"]])
    fig.add_scatter(x=df.index, y=df["resting_hr"], name="FC de repos (bpm)", yaxis="y2", line=dict(color=ENCRE, width=1.5))
    fig.update_layout(yaxis=dict(title="Heures de sommeil"),
                      yaxis2=dict(title="FC de repos", overlaying="y", side="right", showgrid=False))
    figure(fig)
    explain("Les barres sont tes nuits : brique sous 6 heures, vertes à partir de 7 h 30. La ligne est ta fréquence "
            "cardiaque au repos. Une FC de repos qui monte de plus de 3 battements au-dessus de ta moyenne pendant "
            "plusieurs jours est souvent le premier signe d'une fatigue ou d'un début de maladie.",
            analyse["sommeil"])


# --- Nuits et journées ---------------------------------------------------------------------------

def page_nights() -> None:
    st.title("Nuits & journées")
    df, analyse = history(30), get("/analyse")
    if df is None or analyse is None:
        return

    st.header("Ta dernière nuit")
    night = analyse["nuit"].get("nuit")
    if night:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Sommeil", f"{night['sommeil_h']:.1f} h".replace(".", ","))
        c2.metric("Profond", f"{night['profond_pct']} %")
        c3.metric("Paradoxal", f"{night['paradoxal_pct']} %")
        c4.metric("Éveillé", f"{night['eveil_min']} min")
        c5.metric("Coucher", night["coucher"] or "—")
        c6.metric("Lever", night["lever"] or "—")

    nights = df.dropna(subset=["sleep_h"]).tail(14)
    fig = go.Figure()
    for col, name, color in (("deep_sleep_s", "Profond", ACCENT), ("rem_sleep_s", "Paradoxal", "#5E8C7E"),
                             ("light_sleep_s", "Léger", "#C9CDD3"), ("awake_s", "Éveillé", VIGILANCE)):
        if col in nights:
            fig.add_bar(x=nights.index, y=(nights[col] / 3600).round(2), name=name, marker_color=color)
    fig.update_layout(barmode="stack", yaxis_title="Heures")
    figure(fig, 320)
    explain("Chaque barre est une nuit des deux dernières semaines, découpée en phases. Le sommeil profond (vert) "
            "répare le corps après l'effort : chez l'adulte, il représente souvent 15 à 25 % de la nuit. Le sommeil "
            "paradoxal consolide la mémoire et la coordination. Une part ocre importante signale des réveils.",
            analyse["nuit"])

    st.header("Ta journée et ton stress")
    days = df.tail(30)
    fig = go.Figure()
    if "avg_stress" in days:
        colors = ["#C9CDD3" if v <= 25 else "#9AA1AB" if v <= 50 else VIGILANCE if v <= 75 else ALERTE
                  for v in days["avg_stress"].fillna(0)]
        fig.add_bar(x=days.index, y=days["avg_stress"], name="Stress moyen", marker_color=colors)
    if "body_battery_max" in days:
        fig.add_scatter(x=days.index, y=days["body_battery_max"], name="Body Battery au réveil", yaxis="y2",
                        line=dict(color=ACCENT, width=2))
    fig.update_layout(yaxis=dict(title="Stress (0-100)", range=[0, 100]),
                      yaxis2=dict(title="Body Battery", overlaying="y", side="right", range=[0, 100], showgrid=False))
    figure(fig)
    explain("Les barres montrent ton stress moyen de chaque journée, mesuré par la montre à partir de ta fréquence "
            "cardiaque : gris clair sous 25 (repos), gris jusqu'à 50 (faible), ocre jusqu'à 75 (moyen), brique au-delà. "
            "La ligne verte est ta Body Battery au réveil : ce que ta nuit a rechargé. Un stress élevé la veille et "
            "une recharge faible sont deux bonnes raisons d'alléger la séance du jour.",
            {"points": analyse["journee"]["points"] + analyse["stress"]["points"]})

    st.header("Tes activités de la semaine")
    week = analyse["activites"].get("semaine", [])
    if week:
        fig = go.Figure(go.Bar(x=[w["minutes"] for w in week], y=[SPORT_NAMES.get(w["sport"], w["sport"]) for w in week],
                               orientation="h", marker_color=[SPORT_COLORS.get(w["sport"], "#C9CCD8") for w in week],
                               text=[f"{w['seances']} séance(s)" for w in week], textposition="inside"))
        fig.update_layout(xaxis_title="Minutes sur les 7 derniers jours", showlegend=False)
        figure(fig, 220)
    explain("Le temps passé dans chaque sport sur les 7 derniers jours. Le tennis et la musculation fatiguent aussi : "
            "le planning en tient compte pour placer tes séances de course.", analyse["activites"])


# --- Coach ---------------------------------------------------------------------------------------

SUGGESTIONS = ["Je peux faire mon fractionné ce soir ?", "Quelle allure pour mon prochain footing ?",
               "Je suis fatigué, allège ma prochaine séance", "Que manger avant ma prochaine course ?"]


def stream_post(path: str, payload: dict, meta: dict):
    """Lit une réponse envoyée mot à mot par l'API. Les en-têtes et la fin technique vont dans `meta`."""
    if API_URL == "inprocess":
        with inprocess_client().stream("POST", path, json=payload) as response:
            meta["sujets"] = response.headers.get("X-Coach-Sujets", "")
            yield from response.iter_text()
    else:
        with requests.post(f"{API_URL}{path}", json=payload, stream=True, timeout=COACH_TIMEOUT_S) as response:
            response.encoding = "utf-8"
            meta["sujets"] = response.headers.get("X-Coach-Sujets", "")
            yield from response.iter_content(chunk_size=None, decode_unicode=True)


def visible_text(chunks, meta: dict):
    """Ne laisse passer à l'écran que le texte de la réponse : la ligne technique finale va dans `meta`."""
    buffer, tail = "", None
    for chunk in chunks:
        if tail is not None:
            tail += chunk
            continue
        buffer += chunk
        cut = buffer.find("\n[[")
        if cut >= 0:
            if buffer[:cut]:
                yield buffer[:cut]
            tail, buffer = buffer[cut:], ""
        elif len(buffer) > 3:  # garde 3 caractères au cas où le marqueur arrive en deux morceaux
            yield buffer[:-3]
            buffer = buffer[-3:]
    if buffer:
        yield buffer
    for line in (tail or "").splitlines():
        if line.startswith("[[MESURES]]"):
            meta["mesures"] = json.loads(line.removeprefix("[[MESURES]]").strip())
        elif line.startswith("[[ERREUR]]"):
            meta["erreur"] = line.removeprefix("[[ERREUR]]").strip()
        elif line.startswith("[[PROPOSITION]]"):
            meta["proposition"] = json.loads(line.removeprefix("[[PROPOSITION]]").strip())
        elif line.startswith("[[PROPOSITION_REFUSEE]]"):
            meta["proposition_refusee"] = line.removeprefix("[[PROPOSITION_REFUSEE]]").strip()
        elif line.startswith("[[EXERCICES]]"):
            meta["exercices"] = json.loads(line.removeprefix("[[EXERCICES]]").strip())


TOPIC_NAMES = {"allures": "Allures", "predictions": "Prédictions", "nutrition": "Nutrition", "seances": "Séances",
               "planning": "Planning", "sortie": "Sortie", "kilometres": "Kilomètres", "zones": "Zones cardiaques",
               "nuit": "Nuit précédente", "ressenti": "Ton ressenti", "sorties similaires": "Sorties similaires",
               "douleur": "Douleur et exercices"}


def footnote(meta: dict) -> str:
    """Sources et vitesse d'une réponse, en petites étiquettes (HTML)."""
    topics = [TOPIC_NAMES.get(t, t) for t in meta.get("sujets", "").split(",") if t]
    chips = "".join(f'<span class="etiquette">{esc(t)}</span>' for t in ["Contexte du jour", *topics])
    m = meta.get("mesures") or {}
    speed = ""
    if m.get("duree_s") is not None:
        rate = f", {m['jetons_par_s']} jetons/s".replace(".", ",") if m.get("jetons_par_s") else ""
        speed = f'<span class="vitesse">{m["duree_s"]:.0f} s{rate}</span>'
    return f'<div class="sources">{chips}{speed}</div>'


def check_note(check: dict | None) -> str:
    """Vérification des chiffres de la réponse : étiquette discrète si tout est retrouvé, alerte sinon (HTML)."""
    if not check:
        return ""
    if check.get("chiffres_non_verifies"):
        figures = ", ".join(check["chiffres_non_verifies"][:6])
        return (f'<p class="verif-alerte">À vérifier : {esc(figures)}. Ces chiffres ne figurent pas dans tes '
                "données : le coach les a peut-être calculés, ou inventés.</p>")
    if check.get("chiffres_cites"):
        n = check["chiffres_cites"]
        return f'<p class="verif-ok">{n} chiffre{"s" if n > 1 else ""} vérifié{"s" if n > 1 else ""} dans tes données</p>'
    return ""


def stream_answer(payload: dict, path: str = "/coach/question/flux", verify: dict | None = None) -> dict:
    """Affiche la réponse au fil de sa rédaction, puis ses sources, sa vitesse et la vérification de ses chiffres."""
    meta = {}
    try:
        text = st.write_stream(visible_text(stream_post(path, payload, meta), meta))
    except (requests.ConnectionError, requests.Timeout) as exc:
        meta["erreur"] = str(exc)
        text = ""
    text = text if isinstance(text, str) else "".join(text)
    note = footnote(meta)
    if meta.get("erreur"):
        st.error(f"Le coach n'a pas pu répondre : {meta['erreur']}")
    else:
        if verify is not None and text.strip():
            ok, check = post("/coach/verification", {"reponse": text, **verify})
            note += check_note(check if ok else None)
        st.markdown(note, unsafe_allow_html=True)
        coach_extras(meta)
    return {"content": text, "note": note, "erreur": meta.get("erreur"), "proposition": meta.get("proposition"),
            "proposition_refusee": meta.get("proposition_refusee"), "exercices": meta.get("exercices")}


def coach_extras(message: dict) -> None:
    """Sous une réponse : exercices pour une douleur, proposition de modification du planning."""
    if message.get("exercices"):
        exercises_card(message["exercices"])
    if message.get("proposition"):
        proposal_card(message["proposition"])
    elif message.get("proposition_refusee"):
        st.markdown(f'<p class="verif-alerte">Le coach a proposé une modification impossible : '
                    f'{esc(message["proposition_refusee"])} Reformule ta demande.</p>', unsafe_allow_html=True)


def fresh_get(path: str) -> dict:
    """Lecture sans cache (état qui change au clic)."""
    try:
        if API_URL == "inprocess":
            return inprocess_client().get(path).json()
        return requests.get(f"{API_URL}{path}", timeout=10).json()
    except (requests.ConnectionError, requests.Timeout, ValueError):
        return {}


def session_line(s: dict | None) -> str:
    if not s:
        return "<b>Repos</b><small>Séance retirée du planning</small>"
    when = date.fromisoformat(s["date"]).strftime("%d/%m")
    km = f"{s['distance_km']:g} km".replace(".", ",")
    return f"<b>{esc(s['titre'])}</b><small>{esc(s['jour'])} {when} · {km} · {esc(s.get('allure') or '')}</small>"


def proposal_card(adj: dict) -> None:
    """Avant / après, risques signalés, et les boutons : rien ne change sans ta validation."""
    statuses = {a["id"]: a["statut"] for a in fresh_get("/planning/ajustements").get("ajustements", [])}
    status = statuses.get(adj["id"], adj["statut"])
    alerts = "".join(f"<li>{esc(w)}</li>" for w in adj.get("avertissements", []))
    reason = f'<p class="prop-raison">{esc(adj["raison"])}</p>' if adj.get("raison") else ""
    head = "D'après ta demande" if adj.get("origine") == "demande" else "Proposition du coach"
    st.markdown(f"""<div class="proposition"><div class="prop-tete">{head}</div>
        <p class="prop-libelle">{esc(adj["libelle"])}</p>{reason}
        <div class="prop-avant-apres"><div><span>Avant</span>{session_line(adj["avant"])}</div>
        <div class="prop-fleche">→</div><div><span>Après</span>{session_line(adj["apres"])}</div></div>
        {f'<ul class="prop-alertes">{alerts}</ul>' if alerts else ''}</div>""", unsafe_allow_html=True)
    if status == "propose":
        with st.container(key=f"prop_{adj['id']}", horizontal=True):
            if st.button("Valider", key=f"valider_{adj['id']}", type="primary"):
                ok, answer = post(f"/planning/ajustements/{adj['id']}/valider", {})
                if ok:
                    api_get.clear()
                    watch = (answer.get("montre") or {}).get("message")
                    st.session_state["toast"] = answer["message"] + (f" {watch}" if watch else "")
                else:
                    st.session_state["toast"] = answer.get("detail", "Validation impossible.")
                st.rerun()
            if st.button("Refuser", key=f"refuser_{adj['id']}"):
                post(f"/planning/ajustements/{adj['id']}/refuser", {})
                st.rerun()
    else:
        labels = {"accepte": "Validée : ton planning est à jour.", "refuse": "Refusée : le planning ne change pas.",
                  "annule": "Validée puis annulée depuis le planning."}
        st.markdown(f'<p class="prop-statut prop-{status}">{labels.get(status, status)}</p>', unsafe_allow_html=True)


def exercises_card(data: dict) -> None:
    """Exercices issus d'une base fixe (pas générés par le modèle), signaux d'alerte, et rappel de consulter."""
    if data.get("zones"):
        for zone in data["zones"]:
            items = "".join(f'<li><b>{esc(e["nom"])}</b> <span class="dosage">{esc(e["dosage"])}</span>'
                            f'<br><small>{esc(e["consigne"])}</small></li>' for e in zone["exercices"])
            st.markdown(f"""<div class="exercices"><div class="prop-tete">Exercices souvent proposés en kiné :
                {esc(zone["nom"].lower())}</div><ol>{items}</ol>
                <p><b>Côté course.</b> {esc(zone["course"])}</p>
                <p><b>À qui t'adresser.</b> {esc(zone["specialiste"][0].upper() + zone["specialiste"][1:])}.</p></div>""",
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="exercices"><p>Précise où tu as mal (genou, tibia, mollet, tendon d\'Achille, pied, '
                    "hanche, arrière de la cuisse, dos) pour que le coach te propose des exercices adaptés.</p></div>",
                    unsafe_allow_html=True)
    flags = "".join(f"<li>{esc(f)}</li>" for f in data.get("signaux_alerte", []))
    rules = " ".join(esc(r) for r in data.get("regles", []))
    st.markdown(f'<div class="alerte-sante"><b>Consulte rapidement un médecin si :</b><ul>{flags}</ul>'
                f"<p>{rules}</p></div>", unsafe_allow_html=True)


def coach_today() -> None:
    """Ce que le coach voit aujourd'hui : verdict et chiffres du jour, en une ligne."""
    forme, analyse = get("/forme"), get("/analyse")
    if not (forme and analyse):
        return
    verdict = analyse["verdict"]
    color = LEVEL_COLORS[verdict["niveau"]]
    hrv = "—" if forme["hrv_ecart_pct"] is None else f"{forme['hrv_ecart_pct']:+.0f} %"
    sleep = "—" if forme["sommeil_h"] is None else f"{forme['sommeil_h']:.1f} h".replace(".", ",")
    fresh = "—" if forme["fraicheur_tsb"] is None else f"{forme['fraicheur_tsb']:+.0f}"
    cells = [("VFC vs normale", hrv), ("Sommeil", sleep), ("Fraîcheur", fresh)]
    figures = "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in cells)
    st.markdown(f'<div class="coach-jour"><div class="coach-verdict"><span class="point" style="background:{color}">'
                f'</span>{esc(verdict["titre"])}<small>{verdict["score"]} sur 100</small></div>'
                f'<div class="coach-chiffres">{figures}</div></div>', unsafe_allow_html=True)


def page_coach() -> None:
    if "toast" in st.session_state:
        st.toast(st.session_state.pop("toast"))
    status = get("/coach/statut") or {}
    st.title("Ton coach")
    st.markdown(f'<p class="mention-ia">Intelligence artificielle : modèle {esc(status.get("modele", "local"))}, '
                "exécuté sur ta machine. Tes données ne la quittent pas. Ses conseils ne remplacent pas l'avis d'un "
                "professionnel de santé, et il ne modifie rien à ta place.</p>", unsafe_allow_html=True)
    if not status.get("disponible"):
        model = esc(status.get("modele", ""))
        st.markdown(f'<div class="donnees" style="margin-top:1rem"><h4>Coach indisponible</h4><p>'
                    f'{esc(status.get("erreur") or "Le modèle local ne répond pas.")}<br>Vérifie qu\'Ollama est lancé '
                    f'et que le modèle est téléchargé : <code>ollama pull {model}</code>.</p></div>',
                    unsafe_allow_html=True)
        return
    coach_today()

    # Bilan : un article, avec ses sources
    with st.container(key="tete_bilan"):  # titre et action sur une même ligne, soulignée d'un seul filet
        head, action = st.columns([4, 1], vertical_alignment="center")
        head.markdown('<div class="titre-section">Bilan de la semaine</div>', unsafe_allow_html=True)
        with action.container(key="action_bilan"):
            redo = st.button("Rédiger un nouveau bilan" if "bilan" in st.session_state else "Rédiger mon bilan")
    with st.container(key="bilan"):
        if redo:
            st.session_state.pop("bilan", None)
            answer = stream_answer({"bilan": True}, verify={"bilan": True})
            if not answer["erreur"]:
                st.session_state["bilan"] = answer
        elif "bilan" in st.session_state:
            st.markdown(st.session_state["bilan"]["content"])
            st.markdown(st.session_state["bilan"]["note"], unsafe_allow_html=True)
            coach_extras(st.session_state["bilan"])
        else:
            st.markdown('<p class="vide">Le coach résume ta semaine (charge, récupération, sommeil, stress) et te '
                        "propose la suite, en lien avec ton objectif.</p>", unsafe_allow_html=True)

    # Conversation
    st.header("Discussion")
    question = None
    with st.container(key="suggestions"):
        cols = st.columns(len(SUGGESTIONS))
        for col, suggestion in zip(cols, SUGGESTIONS):
            if col.button(suggestion, use_container_width=True):
                question = suggestion
    question = st.chat_input("Pose ta question au coach") or question

    history = st.session_state.setdefault("coach_messages", [])
    if not history and not question:
        st.markdown('<p class="vide">Pose une question sur ta forme, tes allures, ta prochaine séance ou ta course : '
                    "le coach répond avec tes données.</p>", unsafe_allow_html=True)
    for i, message in enumerate(history):
        with st.container(key=f"msg_{i}_{message['role']}"):
            st.markdown(message["content"])
            if message.get("note"):
                st.markdown(message["note"], unsafe_allow_html=True)
            coach_extras(message)
    if question:
        with st.container(key=f"msg_{len(history)}_user"):
            st.markdown(question)
        with st.container(key=f"msg_{len(history) + 1}_assistant"):
            past = [{"role": m["role"], "content": m["content"]} for m in history]
            answer = stream_answer({"question": question, "historique": past},
                                   verify={"question": question, "historique": past})
        if not answer["erreur"]:
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer["content"], "note": answer["note"],
                         **{k: answer[k] for k in ("proposition", "proposition_refusee", "exercices") if answer[k]}}]


# --- Planning ------------------------------------------------------------------------------------

def page_goals() -> None:
    st.title("Mes objectifs")
    data = get("/objectifs")
    goals = (data or {}).get("objectifs", [])
    active = next((g for g in goals if g["actif"] and g["jours_restants"] >= 0), None)

    if active:
        left, right = st.columns([6, 5], gap="large")
        with left:
            race_day = date.fromisoformat(active["date_course"]).strftime("%d/%m/%Y")
            details = [("Jours restants", str(active["jours_restants"])), ("Temps visé", active["temps_vise"] or "—")]
            if active.get("ecart_s") is not None:
                gap = active["ecart_s"]
                details.append(("Écart", f"{'+' if gap > 0 else '−'}{abs(gap) // 60}'{abs(gap) % 60:02d}\""))
            hero(f"{active['nom']}, le {race_day} : ton temps prédit aujourd'hui", active["temps_predit"], details)
        with right:
            follow = get(f"/objectifs/{active['id']}/suivi", semaines=12)
            if follow and follow["evolution"]:
                df = pd.DataFrame(follow["evolution"])
                df["date"] = pd.to_datetime(df["date"])
                fig = go.Figure()
                fig.add_scatter(x=df["date"], y=df["temps_predit_s"] / 60, name="Temps prédit",
                                line=dict(color=ENCRE, width=2), mode="lines+markers", marker=dict(size=5),
                                text=df["temps_predit"], hovertemplate="%{text}<extra></extra>")
                if active.get("temps_vise_s"):
                    fig.add_hline(y=active["temps_vise_s"] / 60, line=dict(color=ACCENT, width=1.5, dash="dot"),
                                  annotation_text="Temps visé", annotation_font_color=ACCENT)
                fig.update_layout(yaxis_title="minutes", showlegend=False)
                figure(fig, 260)
                st.caption("Ton temps prédit sur cette course, semaine après semaine. Sous la ligne verte : "
                           "tu es dans les temps.")

    st.header("Nouvel objectif")
    with st.form("formulaire_objectif", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Nom de la course", placeholder="Semi de Paris")
        distance = c2.selectbox("Distance", [*DISTANCES, "Autre distance"], index=1)
        race_date = c3.date_input("Date", value=date.today() + timedelta(weeks=10),
                                  min_value=date.today() + timedelta(days=3), format="DD/MM/YYYY")
        c4, c5, c6 = st.columns(3)
        target = c4.text_input("Temps visé (facultatif)", placeholder="47:30 ou 1:45:00")
        dplus = c5.number_input("D+ du parcours (m)", 0, 5000, 0, step=10)
        per_week = c6.slider("Sorties de course par semaine", 2, 6, 3)
        c7, c8, c9 = st.columns(3)
        other_km = c7.number_input("Autre distance (km)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, format="%.1f",
                                   help="Utilisée si tu choisis « Autre distance » : 15 km, 20 km, trail de 30 km…")
        tennis = c8.multiselect("Jours de tennis", DAYS, placeholder="Aucun")
        long_day = c9.selectbox("Jour de la sortie longue", DAYS, index=6)
        submitted = st.form_submit_button("Créer l'objectif", type="primary")
    if submitted:
        ok, body = post("/objectifs", {"objectif": {
            "nom": name, "distance": DISTANCES.get(distance), "distance_km": other_km if distance not in DISTANCES else None,
            "date_course": race_date.isoformat(), "temps_vise": target,
            "denivele_m": int(dplus), "seances_par_semaine": per_week,
            "jours_tennis": [DAYS.index(d) for d in tennis], "jour_sortie_longue": DAYS.index(long_day)}})
        if ok:
            api_get.clear()
            st.success(f"Objectif créé : {body['objectif']['nom']}. Ton planning s'est adapté.")
            st.rerun()
        else:
            detail = body.get("detail")
            st.error(" ".join(detail.values()) if isinstance(detail, dict) else str(detail))

    if goals:
        st.header("Tous mes objectifs")
        for g in goals:
            race_day = date.fromisoformat(g["date_course"]).strftime("%d/%m/%Y")
            cols = st.columns([4, 2, 2, 2, 1, 1], vertical_alignment="center")
            label = g.get("libelle", g.get("distance") or "")
            cols[0].markdown(f"**{esc(g['nom'])}**  \n<span style='color:{GRIS}'>{esc(label)}, {race_day}"
                             f"{'  (actif)' if g['actif'] else ''}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"Prédit  \n**{g['temps_predit']}**")
            cols[2].markdown(f"Visé  \n**{g['temps_vise'] or '—'}**")
            cols[3].markdown(f"<span style='color:{GRIS}'>{esc(g['statut'])}</span>", unsafe_allow_html=True)
            if not g["actif"] and g["jours_restants"] >= 0 and cols[4].button("Activer", key=f"act_{g['id']}"):
                post(f"/objectifs/{g['id']}/activer", {})
                api_get.clear()
                st.rerun()
            if cols[5].button("Suppr.", key=f"del_{g['id']}"):
                delete(f"/objectifs/{g['id']}")
                api_get.clear()
                st.rerun()


def page_planning() -> None:
    if "toast" in st.session_state:
        st.toast(st.session_state.pop("toast"))
    st.title("Mon planning")
    plan = get("/planning/actif")
    if not plan:
        return
    goal, weeks = plan["objectif"], plan["semaines"]
    active = plan.get("objectif_actif")
    left, right = st.columns([6, 5], gap="large")
    with left:
        volume = f"{plan['volume_actuel_km_semaine']:g} km par semaine".replace(".", ",")
        details = [("Allure", goal["allure_course"]), ("Volume actuel", volume)]
        if goal["denivele_m"]:
            details.append(("Dénivelé", f"{goal['denivele_m']} m"))
        if active:
            race_day = date.fromisoformat(active["date_course"]).strftime("%d/%m/%Y")
            legend = f"{active['nom']}, le {race_day} : temps visé par le plan"
        else:
            legend = "Sans objectif : deux semaines de développement, base 10 km"
        hero(legend, goal["temps_vise"], details)
    with right:
        phase_colors = {"Développement": "#C9CDD3", "Spécifique": ACCENT, "Affûtage": "#9AA1AB", "Semaine de course": ENCRE}
        fig = go.Figure(go.Bar(x=[f"S{w['numero']}" for w in weeks], y=[w["volume_km"] for w in weeks],
                               marker_color=[phase_colors[w["phase"]] for w in weeks],
                               text=[w["phase"] for w in weeks], hovertemplate="%{x} : %{y} km<br>%{text}<extra></extra>"))
        fig.update_layout(yaxis_title="km par semaine", showlegend=False, bargap=0.45)
        fig.update_traces(textposition="none")
        figure(fig, 230)
        st.caption("Gris clair : développement. Vert : spécifique. Gris : affûtage. Noir : semaine de course.")

    if plan.get("personnalisation"):
        st.markdown('<div class="donnees"><h4>Adapté à tes derniers jours</h4><ul>'
                    + "".join(f"<li>{esc(n)}</li>" for n in plan["personnalisation"]) + "</ul></div>",
                    unsafe_allow_html=True)
    if plan.get("ajustements"):
        st.markdown('<div class="titre-section">Modifié avec ton coach</div>', unsafe_allow_html=True)
        for adj in plan["ajustements"]:
            line, action = st.columns([5, 1], vertical_alignment="center")
            line.markdown(f'<p class="ajustement-ligne">{esc(adj["libelle"])}</p>', unsafe_allow_html=True)
            if action.button("Annuler", key=f"annuler_{adj['id']}"):
                ok, answer = post(f"/planning/ajustements/{adj['id']}/annuler", {})
                api_get.clear()
                watch = (answer.get("montre") or {}).get("message") if ok else None
                st.session_state["toast"] = "Séance d'origine rétablie." + (f" {watch}" if watch else "")
                st.rerun()
    notes = [esc(n) for n in plan["notes"]] + ["Objectif, jours de tennis et nombre de sorties : onglet Objectifs. "
                                               "Nutrition du jour de course : onglet Prédictions."]
    st.markdown('<div class="lecture" style="margin-top:1rem"><h4>À savoir</h4><p>' + "<br>".join(notes)
                + "</p></div>", unsafe_allow_html=True)

    for week in weeks:
        start_day = date.fromisoformat(week["debut"]).strftime("%d/%m")
        st.header(f"Semaine {week['numero']}, à partir du {start_day}")
        st.caption(f"{week['phase']}, {week['volume_km']:g} km".replace(".", ","))
        if week["seances"]:
            sessions_table(week["seances"])
        else:
            st.write("Semaine terminée.")


# --- Prédictions ---------------------------------------------------------------------------------

METHOD_NAMES = {"application": "Foulée (affiché)", "sans_correction": "Sans correction",
                "recalibree": "Recalibrage seul", "questionnaires": "Questionnaires seuls",
                "vo2max_montre": "VO2 max de la montre", "relation_fc_vitesse": "Relation FC / vitesse",
                "performances": "Performances seules", "riegel_derniere": "Riegel, dernière performance"}


def pct(value: float, sign: bool = False) -> str:
    """Pourcentage au format français : 3,4 ou +3,4."""
    return (f"{value:+.1f}" if sign else f"{value:.1f}").replace(".", ",")


def prediction_quality() -> None:
    """Fiabilité mesurée : ce que Foulée aurait prédit la veille de chaque performance réelle."""
    data = get("/qualite/predictions")
    st.header("Fiabilité de ces prédictions")
    if not data or not data["n"]:
        st.markdown('<p class="vide">Pas encore assez de performances pour mesurer la fiabilité.</p>',
                    unsafe_allow_html=True)
        return
    app_score = data["methodes"]["application"]
    bias = app_score["biais_pct"]
    if bias < -1:
        trend = "plutôt trop rapides (optimistes)"
    elif bias > 1:
        trend = "plutôt trop lentes (prudentes)"
    else:
        trend = "sans biais net"
    cols = st.columns(4)
    cols[0].metric("Erreur moyenne", f"{pct(app_score['mape_pct'])} %")
    cols[1].metric("Écart médian", f"{app_score['erreur_mediane_s'] // 60}'{app_score['erreur_mediane_s'] % 60:02d}\"")
    cols[2].metric("À ±3 % près", f"{app_score['part_a_3pct']} %")
    cols[3].metric("Performances testées", data["n"])

    pts = data["points"]
    fig = go.Figure()
    lo = min(min(p["reel_s"], p["predit_s"]) for p in pts) / 60 * 0.95
    hi = max(max(p["reel_s"], p["predit_s"]) for p in pts) / 60 * 1.05
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=FILET, width=1.5), showlegend=False,
                    hoverinfo="skip")
    for source in dict.fromkeys(p["source"] for p in pts):
        group = [p for p in pts if p["source"] == source]
        fig.add_scatter(x=[p["reel_s"] / 60 for p in group], y=[p["predit_s"] / 60 for p in group], mode="markers",
                        name=source.capitalize(), marker=dict(size=9, color=ENCRE if source == "course" else ACCENT,
                                                              opacity=1 if source == "course" else .55),
                        text=[f"{p['date']} : {p['distance_km']:g} km, réel {p['reel']}, prédit {p['predit']} "
                              f"({pct(p['erreur_pct'], sign=True)} %)" for p in group], hovertemplate="%{text}<extra></extra>")
    fig.update_layout(xaxis_title="Temps réel (min)", yaxis_title="Temps prédit la veille (min)", hovermode="closest")
    fig.update_xaxes(tickformat=None)
    figure(fig, 320)

    ranking = sorted(data["methodes"].items(), key=lambda x: x[1]["mape_pct"])
    races = data["par_source"].get("course")
    points = [f"Sur {data['n']} performances, Foulée se trompe en moyenne de {pct(app_score['mape_pct'])} %, "
              f"et ses prédictions sont {trend} ({pct(bias, sign=True)} %)."]
    if races:
        points.append(f"Sur tes {races['n']} course(s), les seuls efforts à fond, l'erreur est de {pct(races['mape_pct'])} % "
                      f"(biais {pct(races['biais_pct'], sign=True)} %). Les meilleurs 5 et 10 km de séance, courus sans "
                      "forcer, paraissent toujours plus lents que la prédiction : c'est normal.")
    best = ranking[0]
    if best[0] != "application":
        points.append(f"La méthode la plus juste sur ton historique est « {METHOD_NAMES[best[0]]} » "
                      f"({pct(best[1]['mape_pct'])} % d'erreur) : une piste pour mieux pondérer les estimations.")
    no_corr = data["methodes"].get("sans_correction")
    if no_corr:
        gain = no_corr["mape_pct"] - app_score["mape_pct"]
        points.append(f"Foulée corrige ses prédictions à partir de ses erreurs passées sur tes courses : sans cette correction, "
                      f"l'erreur serait de {pct(no_corr['mape_pct'])} % (biais {pct(no_corr['biais_pct'], sign=True)} %) ; "
                      f"elle {'la réduit' if gain > 0 else 'ne la réduit pas'} à {pct(app_score['mape_pct'])} %.")
    riegel = data["methodes"].get("riegel_derniere")
    if riegel:
        better = app_score["mape_pct"] < riegel["mape_pct"]
        points.append(f"Face à la référence naïve (formule de Riegel sur ta dernière performance, "
                      f"{pct(riegel['mape_pct'])} %), Foulée fait {'mieux' if better else 'moins bien'}.")
    explain("Chaque point est une performance réelle : en abscisse ton chrono, en ordonnée le temps que Foulée aurait "
            "prédit la veille, avec les seules données connues à ce moment-là. Sur la diagonale, la prédiction est "
            "parfaite ; au-dessus, elle était trop lente ; en dessous, trop rapide. Les points noirs sont tes courses, "
            "les verts les meilleurs 5 et 10 km de tes séances dures, souvent courus sans être à fond.",
            {"points": points})
    with st.expander("Comparer toutes les méthodes"):
        st.dataframe(pd.DataFrame([{"Méthode": METHOD_NAMES[m], "Erreur moyenne (%)": s["mape_pct"],
                                    "Biais (%)": s["biais_pct"], "À ±3 % (%)": s["part_a_3pct"], "n": s["n"]}
                                   for m, s in ranking]), hide_index=True, width="stretch")


def page_predictions() -> None:
    st.title("Mes prédictions")
    c1, c2, c3 = st.columns([3, 3, 2])
    label = c1.radio("Distance", [*DISTANCES, "Autre"], index=1, horizontal=True)
    dplus = c2.slider("D+ du parcours (m)", 0, 1500, 0, step=10)
    adjust = c3.toggle("Avec ma forme du jour", value=True)
    other_km = None
    if label == "Autre":
        other_km = st.number_input("Distance (km)", min_value=1.0, max_value=100.0, value=15.0, step=0.5)
        label = f"{other_km:g} km".replace(".", ",")
    key = "personnalisee" if other_km else DISTANCES[label]
    preds = get("/predictions", distance="10k" if other_km else key, ajuster_au_jour=adjust, denivele_m=dplus,
                distance_km=other_km)
    if not preds:
        return
    p = preds["predictions"][key]

    hero(f"Ton {label.lower()} prédit", p["temps_ajuste"],
         [("Allure", p["allure_course"]), ("Sur le plat", p["temps_plat"]),
          ("Prédiction de la montre", p["prediction_montre"] or "—"),
          ("Forme du jour", f"{preds['ajustement_du_jour_pct']:+.1f} %".replace(".", ","))])
    if preds["avertissements"]:
        st.markdown('<div class="lecture" style="margin-top:1.4rem"><h4>À savoir</h4><p>'
                    + "<br>".join(esc(w) for w in preds["avertissements"]) + "</p></div>", unsafe_allow_html=True)

    prediction_quality()

    st.header("Comment ce temps est calculé")
    st.markdown(f"Ton niveau est résumé par un **VDOT de {preds['vdot']}**, la moyenne des estimations suivantes. "
                "Chaque estimation est corrigée par un **calibrage** sur tes vraies courses.")
    names = {"vo2max_montre": "VO2 max de ta montre", "relation_fc_vitesse": "Relation FC / vitesse sur toutes tes sorties",
             "performances": "Ta meilleure performance"}
    rows = []
    for source, comp in preds["estimation"].items():  # ne pas réutiliser `key` : c'est la distance choisie
        if source == "performances":
            detail = f"{comp['source']} du {comp['date']} : {comp['distance_km']:g} km en {comp['temps']}"
        else:
            detail = f"{comp['mesures_utilisees']} mesures du {comp['periode']}, calibrage x{comp['calibrage']}"
        rows.append({"Source": names.get(source, source), "Détail": detail,
                     "VDOT": comp.get("vdot_estime", comp.get("vdot_deprecie"))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Forme du jour : " + " ".join(preds["explications_ajustement"]))

    st.header("Nutrition et hydratation")
    temperature = st.slider("Température prévue le jour de la course (°C)", -5, 40, 15)
    food = get("/nutrition", distance="10k" if other_km else key, temperature_c=temperature, denivele_m=dplus,
               distance_km=other_km)
    if not food:
        return
    c1, c2, c3 = st.columns(3)
    carbs, fluids = food["glucides_g_par_heure"], food["boisson_ml_par_heure"]
    c1.metric("Glucides pendant la course", f"{carbs[0]}–{carbs[1]} g/h" if carbs[1] else "Aucun")
    c2.metric("Boisson", f"{fluids[0]}–{fluids[1]} ml/h" if fluids[1] else "Pas nécessaire")
    c3.metric("Sodium", "300–600 mg/h" if food["sodium"] else "Pas nécessaire")
    before, during, after = st.columns(3, gap="large")
    for col, title, items in ((before, "Avant", food["avant"]), (during, "Pendant", food["pendant"]),
                              (after, "Après", food["apres"])):
        col.markdown(f'<div class="lecture"><h4>{title}</h4><p>' + "<br><br>".join(esc(i) for i in items)
                     + "</p></div>", unsafe_allow_html=True)
    if food["reperes"]:
        st.subheader(f"Tes repères sur la course (temps prévu : {food['temps_prevu']})")
        st.dataframe(pd.DataFrame(food["reperes"]).rename(columns={"minute": "Minute", "km": "Kilomètre",
                                                                    "action": "À faire"}),
                     hide_index=True, use_container_width=True)
    st.caption(food["avertissement"])


# --- Allures -------------------------------------------------------------------------------------

def page_paces() -> None:
    st.title("Mes allures")
    data, preds = get("/allures"), get("/predictions", distance="toutes", ajuster_au_jour=False)
    if not (data and preds):
        return
    zones = data["zones"]
    rows = [(TYPE_NAMES[k], zones[k]["recommandation"], TYPE_COLORS[k]) for k in ("ef", "tempo", "fractionne")]
    for label, key in (("Allure semi", "semi"), ("Allure 10 km", "10k"), ("Allure 5 km", "5k")):
        pace = preds["predictions"][key]["temps_base_s"] / {"semi": 21.0975, "10k": 10, "5k": 5}[key]
        rows.append((label, {"allure_rapide_s": pace - 2, "allure_lente_s": pace + 2,
                             "allure_rapide": preds["predictions"][key]["allure_course"]}, ENCRE))

    fig = go.Figure()
    for name, reco, color in reversed(rows):
        if reco.get("allure_rapide_s") is None:
            continue
        fast, slow = reco["allure_rapide_s"], reco["allure_lente_s"]
        fig.add_bar(y=[name], x=[max(slow - fast, 4) / 60], base=[fast / 60], orientation="h", marker_color=color, width=0.35,
                    hovertemplate=f"{name} : {reco.get('allure_rapide')} – {reco.get('allure_lente', '')}<extra></extra>",
                    showlegend=False)
    ticks = list(range(180, 450, 15))
    fig.update_xaxes(tickvals=[t / 60 for t in ticks], ticktext=[f"{t // 60}'{t % 60:02d}" for t in ticks],
                     autorange="reversed", title="Allure (min/km), de la plus lente à la plus rapide")
    figure(fig, 320)

    purposes = {"ef": "La base de tout : 70 à 80 % de ton volume. Tu dois pouvoir parler en courant.",
                "tempo": "Effort soutenu et continu, 20 à 40 minutes : repousse ton seuil.",
                "fractionne": "Répétitions rapides avec récupération : développe ta VMA."}
    names = {"observe": "tes séances", "modele_fc": "ton modèle FC → allure", "theorique_vdot": "la théorie (VDOT)"}
    rows_html = []
    for key in ("ef", "tempo", "fractionne"):
        reco = zones[key]["recommandation"]
        fc = f"FC {reco['fc_cible'][0]}–{reco['fc_cible'][1]} bpm" if reco["fc_cible"] else "FC : pas encore de modèle"
        rows_html.append(f"""<tr><td class="type"><i style="background:{TYPE_COLORS[key]}"></i>{TYPE_NAMES[key]}</td>
            <td><div class="description">{purposes[key]}</div>
            <div class="objectif">Calculée à partir de {names.get(reco['source'], '—')}.</div></td>
            <td class="chiffres">{esc(reco['allure_rapide'] or '—')} – {esc(reco['allure_lente'] or '—')}
            <span>{esc(fc)}</span></td></tr>""")
    st.header("Tes zones d'entraînement")
    st.markdown(f'<table class="carnet">{"".join(rows_html)}</table>', unsafe_allow_html=True)
    best = (data.get("allure_max") or {}).get("meilleur_1km", "—")
    st.caption(f"Allure max (ton meilleur kilomètre récent) : {best}. Allures en équivalent plat : "
               "en côte, ralentis pour garder le même effort.")


# --- Séances -------------------------------------------------------------------------------------

def pace_axis(fig: go.Figure, values_s: list[float], **kwargs) -> None:
    """Axe d'allure lisible (min/km), le plus rapide en haut."""
    lo, hi = min(values_s) - 10, max(values_s) + 10
    step = 15 if hi - lo > 60 else 5
    ticks = list(range(int(lo // step * step), int(hi) + step, step))
    fig.update_layout(yaxis=dict(tickvals=ticks, ticktext=[f"{t // 60}'{t % 60:02d}" for t in ticks],
                                 range=[hi, lo], title="Allure (min/km)", **kwargs))


def coach_comment(activity_id: int) -> None:
    """L'avis du coach IA sur la sortie : déjà rédigé, ou à demander (une dizaine de secondes)."""
    head, action = st.columns([4, 1], vertical_alignment="center")
    head.markdown('<div class="titre-section">L\'avis de ton coach</div>', unsafe_allow_html=True)
    saved = get("/courses/commentaire", activity_id=activity_id) or {}
    with action.container(key="action_avis"):
        ask = st.button("Demander à nouveau" if saved.get("commentaire") else "Demander son avis",
                        key=f"avis_{activity_id}")
    with st.container(key="bilan"):  # même mise en forme que le bilan du coach : un article à filet vert
        if ask:
            answer = stream_answer({"activity_id": activity_id}, "/courses/commentaire/flux",
                                   verify={"activity_id": activity_id})
            if not answer["erreur"]:
                api_get.clear()
        elif saved.get("commentaire"):
            st.markdown(saved["commentaire"])
            st.markdown(footnote({"sujets": ",".join(saved.get("sujets", [])), "mesures": saved.get("mesures")})
                        + check_note(saved.get("verification")), unsafe_allow_html=True)
        else:
            st.markdown('<p class="vide">Le coach lit tout : tes kilomètres, ton dénivelé, tes zones, ta nuit '
                        "précédente, ton ressenti et tes sorties similaires, puis te dit ce qu'il en pense.</p>",
                        unsafe_allow_html=True)


def run_analysis_section() -> None:
    """Analyse d'une sortie : allure et FC au km, zones cardiaques, efficacité comparée."""
    runs = (get("/courses", limite=15) or {}).get("courses", [])
    if not runs:
        return
    labels = {r["activity_id"]: (f"{date.fromisoformat(r['date'][:10]).strftime('%d/%m')}  ·  "
                                 f"{r['distance_km']:g} km  ·  {TYPE_NAMES.get(r['type'], 'Sortie')}").replace(".", ",")
              for r in runs}
    chosen = st.selectbox("Sortie", list(labels), format_func=labels.get, key="sortie_analysee")
    data = get("/courses/analyse", activity_id=chosen)
    if not data:
        return
    r = data["resume"]
    when = date.fromisoformat(r["date"][:10]).strftime("%d/%m/%Y")
    fc = f"{r['fc_moyenne']} bpm ({r['fc_pct_max']} % FC max)" if r["fc_moyenne"] else "—"
    hero(f"{r['type_libelle'].capitalize()} du {when}, à {r['date'][11:]}", r["allure"].removesuffix("/km"),
         [("Distance", f"{r['distance_km']:g} km".replace(".", ",")), ("Durée", f"{r['duree_min']} min"),
          ("FC moyenne", fc), ("Dénivelé", f"{r['denivele_m']} m")])

    coach_comment(chosen)

    sections = data["sections"]
    st.header("Allure et fréquence cardiaque, kilomètre par kilomètre")
    laps = data["tours"]
    if laps:
        km = [f"{i + 1}" for i in range(len(laps))]
        paces = [t["allure_s"] for t in laps]
        fig = go.Figure()
        conf = data.get("conformite")
        reco_note = ""
        if conf:
            fast, slow = conf["rapide_s"], conf["lente_s"]
            fig.add_hrect(y0=fast, y1=slow, fillcolor="rgba(31,92,74,.07)", line_width=0)
            reco_note = f" La bande verte pâle est ta fourchette conseillée pour ce type de séance ({conf['fourchette']})."
        fig.add_scatter(x=km, y=[t["fc"] for t in laps], name="FC moyenne (bpm)", yaxis="y2", mode="lines+markers",
                        line=dict(color=GRIS, width=1.5, dash="dot"), marker=dict(size=5),
                        hovertemplate="km %{x} : %{y} bpm<extra></extra>")
        flats = [t["allure_plat_s"] for t in laps]
        hilly = any(t["denivele_m"] >= 15 for t in laps)
        if hilly:
            fig.add_scatter(x=km, y=paces, name="Allure réelle", mode="lines+markers",
                            line=dict(color="#9AA1AB", width=1.5), marker=dict(size=5),
                            text=[f"{t['allure']}, D+ {t['denivele_m']} m" for t in laps],
                            hovertemplate="km %{x} : %{text}<extra></extra>")
        fig.add_scatter(x=km, y=flats if hilly else paces, name="Allure équivalente plat" if hilly else "Allure",
                        mode="lines+markers", line=dict(color=ACCENT, width=2.5), marker=dict(size=7),
                        text=[t["allure_plat"] if hilly else t["allure"] for t in laps],
                        hovertemplate="km %{x} : %{text}<extra></extra>")
        pace_axis(fig, paces + flats + ([fast, slow] if conf else []))
        hrs = [t["fc"] for t in laps if t["fc"]]
        fig.update_layout(yaxis2=dict(title="FC (bpm)", overlaying="y", side="right", showgrid=False, tickformat="d",
                                      range=[min(hrs) - 10, max(hrs) + 10] if hrs else None),
                          xaxis_title="Kilomètre", hovermode="x unified")
        fig.update_xaxes(tickformat=None, type="category")
        figure(fig, 320)
        explain("La ligne verte est ton allure à chaque kilomètre (plus haut = plus rapide), la ligne grise pointillée "
                "ta fréquence cardiaque moyenne sur ce kilomètre. Une allure stable avec une FC qui monte doucement est "
                "normale : c'est la dérive cardiaque. Si la FC grimpe nettement alors que l'allure baisse, l'effort "
                "était trop élevé pour la durée." + reco_note
                + (" Sur ce parcours vallonné, la ligne grise est ton allure réelle et la verte ton allure ramenée sur "
                   "le plat (1 m de montée compte comme 7,92 m de plat) : c'est elle qui dit si ton effort était "
                   "régulier." if hilly else ""), {"points": sections["allure"]})
    else:
        st.markdown('<p class="vide">Pas de détail au kilomètre pour cette sortie (tapis, ou sortie de plus de '
                    "60 jours).</p>", unsafe_allow_html=True)

    if data.get("zones"):
        st.header("Zones cardiaques")
        zones = data["zones"]
        colors = ["#C9CDD3", "#5E8C7E", ACCENT, VIGILANCE, ALERTE]
        fig = go.Figure(go.Bar(x=[z["pct"] for z in zones], y=[z["nom"] for z in zones], orientation="h",
                               marker_color=colors, text=[f"{z['pct']} %  ·  {z['minutes']} min" for z in zones],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y} : %{x} %<extra></extra>"))
        fig.update_layout(xaxis=dict(visible=False, range=[0, max(z["pct"] for z in zones) * 1.3]),
                          yaxis=dict(autorange="reversed"), showlegend=False, bargap=0.45)
        figure(fig, 250)
        explain("Le temps passé dans chacune des 5 zones cardiaques définies par ta montre. Une endurance fondamentale "
                "doit rester surtout en zones 1 et 2 ; un tempo vise les zones 3 et 4 ; un fractionné monte en zones "
                "4 et 5 sur les répétitions.", {"points": sections["zones"]})

    st.header("Ton efficacité, sortie après sortie")
    history = [x for x in runs if x["efficacite"]]
    if len(history) >= 3:
        fig = go.Figure()
        for kind in dict.fromkeys(x["type"] for x in history):
            pts = [x for x in history if x["type"] == kind]
            fig.add_scatter(x=[x["date"][:10] for x in pts], y=[x["efficacite"] for x in pts], mode="markers",
                            name=TYPE_NAMES.get(kind, "Autre"), marker=dict(size=9, color=TYPE_COLORS.get(kind, GRIS)),
                            text=[f"{x['distance_km']:g} km à {x['allure']}, {x['fc_moyenne']} bpm" for x in pts],
                            hovertemplate="%{x} : %{y:.2f} m/battement<br>%{text}<extra></extra>")
        current = next((x for x in history if x["activity_id"] == chosen), None)
        if current:
            fig.add_scatter(x=[current["date"][:10]], y=[current["efficacite"]], mode="markers", showlegend=False,
                            marker=dict(size=17, color="rgba(0,0,0,0)", line=dict(color=ENCRE, width=2)),
                            hoverinfo="skip")
        fig.update_layout(yaxis_title="mètres par battement", hovermode="closest")
        figure(fig, 280)
    explain("Chaque point est une sortie : la distance que tu parcours pour un battement de cœur, avec la montée "
            "convertie en distance de plat. À allure égale, une FC plus basse fait monter le point : c'est le "
            "signe le plus direct de progression en endurance. Compare les points d'un même type de séance ; la "
            "sortie analysée est entourée.", {"points": sections["efficacite"]})


def page_sessions() -> None:
    st.title("Mes séances")
    run_analysis_section()
    st.header("Historique")
    data = get("/seances", limite=50)
    if not data:
        return
    df = pd.DataFrame(data["seances"])
    kinds = sorted(df["type"].dropna().unique())
    chosen = st.multiselect("Types de séance", kinds, default=kinds, format_func=lambda k: TYPE_NAMES.get(k, k))
    df = df[df["type"].isin(chosen)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Séances", len(df))
    c2.metric("Kilomètres", f"{df['distance_km'].sum():.0f}")
    c3.metric("FC moyenne", f"{df['fc_moyenne'].mean():.0f} bpm" if df["fc_moyenne"].notna().any() else "—")
    table = df.assign(type=df["type"].map(lambda k: TYPE_NAMES.get(k, k))).rename(columns={
        "date": "Date", "distance_km": "Distance (km)", "duree_min": "Durée (min)", "allure": "Allure",
        "fc_moyenne": "FC moyenne", "type": "Type", "type_source": "Type déterminé par"})
    st.dataframe(table, hide_index=True, use_container_width=True)


# --- Synchronisation à l'ouverture -----------------------------------------------------------------

def fresh_status() -> dict:
    """État de la synchronisation, sans cache (il change pendant qu'on le suit)."""
    try:
        if API_URL == "inprocess":
            return inprocess_client().get("/sync/statut").json()
        return requests.get(f"{API_URL}/sync/statut", timeout=10).json()
    except (requests.ConnectionError, requests.Timeout, ValueError):
        return {}


def age_text(minutes: int | None) -> str:
    if minutes is None:
        return "jamais synchronisées"
    if minutes < 1:
        return "synchronisées à l'instant"
    if minutes < 60:
        return f"synchronisées il y a {minutes} min"
    if minutes < 1440:
        return f"synchronisées il y a {minutes // 60} h"
    return f"synchronisées il y a {minutes // 1440} j"


def sync_status_line() -> None:
    """Zone qui se rafraîchit seule : barre pendant la synchronisation, âge des données ensuite.

    La page entière reste utilisable pendant ce temps ; à la fin, elle se recharge avec les nouvelles données.
    """
    status = fresh_status()
    if not status:
        return
    if status.get("en_cours"):
        st.session_state["synchro_suivie"] = True
        step = status.get("etape") or "Finalisation"
        elapsed = status.get("depuis_s")
        clock = "" if elapsed is None else f" · {elapsed // 60} min {elapsed % 60:02d} s" if elapsed >= 60 else f" · {elapsed} s"
        st.progress(max(float(status.get("progression") or 0), 0.03),
                    text=f"Mise à jour : {step[0].lower()}{step[1:]}…{clock}")
        if elapsed is not None and elapsed > SLOW_SYNC_S:
            st.caption("Garmin répond lentement aujourd'hui. Tu peux utiliser Foulée en attendant : "
                       "la page se mettra à jour toute seule à la fin.")
        return
    if st.session_state.pop("synchro_suivie", False):  # elle vient de se terminer
        api_get.clear()
        if status.get("erreur"):
            st.session_state["synchro_erreur"] = status["erreur"]
        st.rerun(scope="app")
    with st.container(key="etat_synchro", horizontal=True, horizontal_alignment="center",
                      vertical_alignment="center", gap="small"):
        error = st.session_state.get("synchro_erreur")
        text = (f"Mise à jour incomplète, données {age_text(status.get('age_min'))}" if error
                else f"Données {age_text(status.get('age_min'))}")
        st.markdown(f'<p class="age-donnees">{esc(text)}</p>', unsafe_allow_html=True, width="content")
        if st.button("Mettre à jour", width="content"):
            st.session_state.pop("synchro_erreur", None)
            ok, _ = post("/sync", {})
            if ok:
                st.session_state["synchro_suivie"] = True
                st.rerun(scope="app")  # relance la zone en mode « rafraîchissement automatique »


def sync_header() -> None:
    """Synchronisation automatique à l'ouverture si les données ont vieilli, sans bloquer la page."""
    if AUTO_SYNC and "synchro_ouverture" not in st.session_state:
        st.session_state["synchro_ouverture"] = True
        ok, answer = post(f"/sync?si_plus_ancienne_que_min={AUTO_SYNC_AFTER_MIN}", {})
        if ok and (answer.get("lancee") or answer.get("en_cours")):
            st.session_state["synchro_suivie"] = True
    # Pendant une synchronisation, la zone se rafraîchit chaque seconde ; sinon, elle reste immobile
    running = st.session_state.get("synchro_suivie", False)
    st.fragment(sync_status_line, run_every=1 if running else None)()


PAGES = {"Accueil": page_home, "Coach": page_coach, "Objectifs": page_goals, "Planning": page_planning,
         "Ma forme": page_fitness, "Nuits & journées": page_nights, "Prédictions": page_predictions,
         "Allures": page_paces, "Séances": page_sessions}

st.markdown('<div class="marque">Foulée</div>', unsafe_allow_html=True)
sync_header()
choice = st.radio("Navigation", list(PAGES), horizontal=True, label_visibility="collapsed", key="page")
PAGES[choice]()
