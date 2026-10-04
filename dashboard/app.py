"""Tableau de bord Garmin-Run : l'interface ne lit jamais les données, elle interroge l'API.

Lancement local (API démarrée sur le port 8000) :  streamlit run dashboard/app.py
Sans API séparée :  $env:RUNLAB_API_URL = "inprocess"; streamlit run dashboard/app.py
"""

import html
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("RUNLAB_API_URL", "http://127.0.0.1:8000")
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

st.set_page_config(page_title="Garmin-Run", page_icon="🏃", layout="wide")
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


def post(path: str, payload: dict) -> tuple[bool, dict]:
    """Envoie des données à l'API (questionnaire). Renvoie (succès, réponse)."""
    try:
        if API_URL == "inprocess":
            response = inprocess_client().post(path, json=payload)
        else:
            response = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    except requests.ConnectionError:
        return False, {"detail": f"L'API ne répond pas à l'adresse {API_URL}."}
    return response.status_code == 200, response.json()


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
        rows.append(f"""<tr><td class="jour"><b>{esc(s['jour'])}</b><span>{when}</span></td>
            <td class="type"><i style="background:{color}"></i>{esc(TYPE_NAMES.get(s['type'], s['type']))}</td>
            <td><div class="titre">{esc(s['titre'])}</div><div class="description">{esc(s['description'])}</div>
            <div class="objectif">{esc(s['objectif'])}</div></td>
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
                    api_get.clear()
                    st.success(body["message"])
                else:
                    st.error(f"Réponses non enregistrées : {body.get('detail')}")


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
    objective = st.session_state.get("objectif", {})
    plan = get("/planning", **objective)
    upcoming = [s for w in (plan or {}).get("semaines", []) for s in w["seances"]][:3]
    if upcoming:
        sessions_table(upcoming)
        if not objective:
            st.caption("Planning général sur deux semaines. "
                       "Fixe un objectif dans l'onglet Planning pour une préparation sur mesure.")

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


# --- Planning ------------------------------------------------------------------------------------

def page_planning() -> None:
    st.title("Mon planning")
    saved = st.session_state.get("objectif", {})
    with st.form("formulaire_objectif"):
        c1, c2, c3 = st.columns(3)
        current = list(DISTANCES.values()).index(saved.get("distance", "10k"))
        distance = c1.selectbox("Course visée", list(DISTANCES), index=current)
        default_date = date.fromisoformat(saved["date_course"]) if saved.get("date_course") else date.today() + timedelta(weeks=8)
        race_date = c2.date_input("Date de la course", value=default_date, min_value=date.today() + timedelta(days=3),
                                  format="DD/MM/YYYY")
        dplus = c3.number_input("D+ du parcours (m)", 0, 5000, int(saved.get("denivele_m", 0)), step=10)
        c4, c5, c6 = st.columns(3)
        per_week = c4.slider("Sorties de course par semaine", 2, 6, int(saved.get("seances_par_semaine", 3)))
        tennis = c5.multiselect("Jours de tennis", DAYS, placeholder="Aucun",
                                default=[DAYS[int(d)] for d in saved.get("jours_tennis", "").split(",") if d])
        long_day = c6.selectbox("Jour de la sortie longue", DAYS, index=int(saved.get("jour_sortie_longue", 6)))
        submitted = st.form_submit_button("Construire mon planning", type="primary")
    if submitted:
        st.session_state["objectif"] = {
            "distance": DISTANCES[distance], "date_course": race_date.isoformat(), "denivele_m": int(dplus),
            "seances_par_semaine": per_week, "jours_tennis": ",".join(str(DAYS.index(d)) for d in tennis),
            "jour_sortie_longue": DAYS.index(long_day),
        }
    objective = st.session_state.get("objectif")
    if not objective:
        st.markdown('<div class="lecture" style="margin-top:1.2rem"><p>Renseigne ta course, ses contraintes et tes jours '
                    'de tennis, puis construis ton planning : il part de ton volume réel et de tes allures personnelles.</p>'
                    '</div>', unsafe_allow_html=True)
        return

    plan = get("/planning", **objective)
    if not plan:
        return
    goal = plan["objectif"]
    weeks = plan["semaines"]
    race_day = date.fromisoformat(goal["date_course"]).strftime("%d/%m/%Y")
    left, right = st.columns([6, 5], gap="large")
    with left:
        volume = f"{plan['volume_actuel_km_semaine']:g} km par semaine".replace(".", ",")
        details = [("Allure", goal["allure_course"]), ("Volume actuel", volume)]
        if goal["denivele_m"]:
            details.append(("Dénivelé", f"{goal['denivele_m']} m"))
        hero(f"Objectif {distance.lower()} le {race_day}", goal["temps_vise"], details)
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
    notes = [esc(n) for n in plan["notes"]] + ["Nutrition et hydratation du jour de course : onglet Prédictions."]
    st.markdown('<div class="lecture" style="margin-top:1rem"><h4>À savoir</h4><p>' + "<br>".join(notes)
                + "</p></div>", unsafe_allow_html=True)

    for week in weeks:
        start = date.fromisoformat(week["debut"]).strftime("%d/%m")
        st.header(f"Semaine {week['numero']}, à partir du {start}")
        st.caption(f"{week['phase']}, {week['volume_km']:g} km".replace(".", ","))
        if week["seances"]:
            sessions_table(week["seances"])
        else:
            st.write("Semaine terminée.")


# --- Prédictions ---------------------------------------------------------------------------------

def page_predictions() -> None:
    st.title("Mes prédictions")
    c1, c2, c3 = st.columns([3, 3, 2])
    label = c1.radio("Distance", list(DISTANCES), index=1, horizontal=True)
    dplus = c2.slider("D+ du parcours (m)", 0, 1500, 0, step=10)
    adjust = c3.toggle("Avec ma forme du jour", value=True)
    preds = get("/predictions", distance=DISTANCES[label], ajuster_au_jour=adjust, denivele_m=dplus)
    if not preds:
        return
    p = preds["predictions"][DISTANCES[label]]

    hero(f"Ton {label.lower()} prédit", p["temps_ajuste"],
         [("Allure", p["allure_course"]), ("Sur le plat", p["temps_plat"]),
          ("Prédiction de la montre", p["prediction_montre"] or "—"),
          ("Forme du jour", f"{preds['ajustement_du_jour_pct']:+.1f} %".replace(".", ","))])
    if preds["avertissements"]:
        st.markdown('<div class="lecture" style="margin-top:1.4rem"><h4>À savoir</h4><p>'
                    + "<br>".join(esc(w) for w in preds["avertissements"]) + "</p></div>", unsafe_allow_html=True)

    st.header("Comment ce temps est calculé")
    st.markdown(f"Ton niveau est résumé par un **VDOT de {preds['vdot']}**, la moyenne des estimations suivantes. "
                "Chaque estimation est corrigée par un **calibrage** sur tes vraies courses.")
    names = {"vo2max_montre": "VO2 max de ta montre", "relation_fc_vitesse": "Relation FC / vitesse sur toutes tes sorties",
             "performances": "Ta meilleure performance"}
    rows = []
    for key, comp in preds["estimation"].items():
        if key == "performances":
            detail = f"{comp['source']} du {comp['date']} : {comp['distance_km']:g} km en {comp['temps']}"
        else:
            detail = f"{comp['mesures_utilisees']} mesures du {comp['periode']}, calibrage x{comp['calibrage']}"
        rows.append({"Source": names.get(key, key), "Détail": detail,
                     "VDOT": comp.get("vdot_estime", comp.get("vdot_deprecie"))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Forme du jour : " + " ".join(preds["explications_ajustement"]))

    st.header("Nutrition et hydratation")
    temperature = st.slider("Température prévue le jour de la course (°C)", -5, 40, 15)
    food = get("/nutrition", distance=DISTANCES[label], temperature_c=temperature, denivele_m=dplus)
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

def page_sessions() -> None:
    st.title("Mes séances")
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


PAGES = {"Accueil": page_home, "Ma forme": page_fitness, "Nuits & journées": page_nights, "Planning": page_planning,
         "Prédictions": page_predictions, "Allures": page_paces, "Séances": page_sessions}

st.markdown('<div class="marque">Garmin-Run</div>', unsafe_allow_html=True)
choice = st.radio("Navigation", list(PAGES), horizontal=True, label_visibility="collapsed", key="page")
PAGES[choice]()
