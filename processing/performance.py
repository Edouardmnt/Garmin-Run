"""Performance de course : VDOT, temps prédits, allures d'entraînement et ajustement du jour.

Méthode : formules de Daniels et Gilbert (VDOT). Une performance (distance, temps) donne un VDOT,
indicateur de capacité aérobie ; le VDOT donne en retour un temps sur n'importe quelle distance et
des allures d'entraînement. Référence : J. Daniels, "Daniels' Running Formula".

L'ajustement du jour (nuit, fatigue accumulée) est une heuristique volontairement prudente et bornée,
à calibrer quand les données personnelles seront suffisantes.
"""

import math
from datetime import date

import numpy as np
import pandas as pd

DISTANCES_M = {"5k": 5000.0, "10k": 10000.0, "semi": 21097.5, "marathon": 42195.0}

# Allures d'entraînement : fourchettes de % de VDOT (intensités de Daniels)
TRAINING_ZONES = {
    "ef": (0.62, 0.74, "Endurance fondamentale : footing facile, conversation possible"),
    "marathon": (0.75, 0.84, "Allure marathon"),
    "seuil": (0.83, 0.88, "Tempo / seuil : effort soutenu et continu, 20 à 40 min"),
    "fractionne": (0.95, 1.00, "Fractionné long (3 à 5 min) : allure proche de la VMA aérobie"),
    "vitesse": (1.05, 1.10, "Fractionné court (200 à 400 m) : vitesse et économie de course"),
}
MIN_PACE_S, MAX_PACE_S = 150, 600  # garde-fou : allures plausibles entre 2'30" et 10'00" au km


# --- Formules de Daniels-Gilbert ---------------------------------------------------------------

def vo2_at_speed(v_m_min: float) -> float:
    """Coût en oxygène (ml/kg/min) pour courir à v mètres par minute."""
    return -4.60 + 0.182258 * v_m_min + 0.000104 * v_m_min**2


def fraction_sustainable(t_min: float) -> float:
    """Fraction de la VO2max soutenable pendant t minutes."""
    return 0.8 + 0.1894393 * math.exp(-0.012778 * t_min) + 0.2989558 * math.exp(-0.1932605 * t_min)


def vdot(distance_m: float, time_s: float) -> float:
    t_min = time_s / 60
    return vo2_at_speed(distance_m / t_min) / fraction_sustainable(t_min)


def predict_time_s(vdot_value: float, distance_m: float) -> float:
    """Temps (s) qui correspond à ce VDOT sur cette distance (recherche par dichotomie)."""
    low, high = distance_m / 10, distance_m * 1.2  # entre 36 km/h et 3 km/h : large
    for _ in range(100):
        mid = (low + high) / 2
        if vdot(distance_m, mid) > vdot_value:
            low = mid  # trop rapide pour ce VDOT : il faut plus de temps
        else:
            high = mid
    return (low + high) / 2


def pace_at_fraction(vdot_value: float, fraction: float) -> float:
    """Allure (s/km) pour courir à une fraction donnée du VDOT : on inverse vo2_at_speed."""
    a, b, c = 0.000104, 0.182258, -(4.60 + fraction * vdot_value)
    v = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)  # m/min
    return 1000 / v * 60


def riegel_time_s(known_m: float, known_s: float, target_m: float) -> float:
    """Référence naïve classique : T2 = T1 x (D2 / D1)^1.06."""
    return known_s * (target_m / known_m) ** 1.06


# --- Dénivelé ------------------------------------------------------------------------------------

# Équivalence de Scarf (2007) : 1 m de montée coûte autant que 7,92 m de plat.
# Référence : P. Scarf, "Route choice in mountain navigation, Naismith's rule, and the equivalence
# of distance and climb", Journal of Sports Sciences, 2007.
SCARF_EQUIVALENCE = 7.92
MAX_CLIMB_M_PER_KM = 50  # au-delà : trail où l'on marche, la relation effort / vitesse n'est plus comparable


def flat_equivalent_m(distance_m, dplus_m):
    """Distance de plat équivalente à un parcours avec du dénivelé positif."""
    return distance_m + SCARF_EQUIVALENCE * dplus_m


def add_flat_speed(runs: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le D+ et la vitesse équivalente sur le plat de chaque sortie.

    Priorité à la vitesse ajustée à la pente calculée par Garmin (GAP), précise kilomètre par kilomètre ;
    sinon, équivalence de Scarf appliquée au D+ total de la sortie.
    """
    runs = runs.copy()
    runs["dplus_m"] = runs["elevation_gain_m"].fillna(0) if "elevation_gain_m" in runs else 0.0
    moving = runs["moving_duration_s"].fillna(runs["duration_s"]) if "moving_duration_s" in runs else runs["duration_s"]
    scarf = flat_equivalent_m(runs["distance_m"], runs["dplus_m"]) / moving
    gap = runs["avg_gap_speed_ms"] if "avg_gap_speed_ms" in runs else pd.Series(np.nan, index=runs.index)
    runs["flat_speed_ms"] = gap.where(gap > 0, scarf)
    runs["flat_source"] = np.where(gap > 0, "gap_garmin", "scarf")
    runs["climb_m_per_km"] = runs["dplus_m"] / (runs["distance_m"] / 1000)
    return runs


# --- Performances personnelles -------------------------------------------------------------------

HARD_SESSIONS = {"course", "tempo", "fractionne"}  # seuls efforts assez intenses pour refléter la capacité
DECAY_PER_DAY = 0.0003       # une performance perd ~1 % de valeur par mois d'ancienneté
MAX_DECAY = 0.10             # au plus -10 %, même pour une performance très ancienne
CALIBRATION_BOUNDS = (0.75, 1.10)


def session_types(labels: pd.DataFrame | None) -> dict:
    """Type de chaque sortie : ton étiquette si elle existe, sinon la suggestion des règles."""
    if labels is None or labels.empty:
        return {}
    label = labels["label"].where(labels["label"].notna() & (labels["label"].str.strip() != ""), labels["suggestion"])
    return dict(zip(labels["activity_id"], label))


def race_effort(run: pd.Series) -> tuple[float, float]:
    """Chrono d'une course : le meilleur temps sur la distance officielle calculé par Garmin dans l'activité.

    La durée totale de l'enregistrement peut inclure l'échauffement, la marche après l'arrivée
    ou une montre arrêtée en retard ; le meilleur 10 km (ou 5 km) de l'activité ne les contient pas.
    """
    for distance, col in ((10000.0, "fastest_10k_s"), (5000.0, "fastest_5k_s")):
        if run["distance_m"] >= distance * 0.97 and pd.notna(run.get(col)):
            return distance, float(run[col])
    moving = run.get("moving_duration_s")
    return float(run["distance_m"]), float(moving if pd.notna(moving) else run["duration_s"])


def collect_performances(activities: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Performances utilisables : TOUTES les courses, et les meilleurs 5/10 km des seances dures uniquement."""
    runs = activities[activities["sport"] == "running"].copy()
    runs["date"] = pd.to_datetime(runs["start_time"]).dt.normalize()
    runs["kind"] = runs["activity_id"].map(session_types(labels))
    rows = []

    for _, r in runs[runs["kind"] == "course"].iterrows():
        distance, time_s = race_effort(r)
        dplus = (r.get("elevation_gain_m") or 0) * distance / r["distance_m"]  # D+ au prorata de la portion
        rows.append({"date": r["date"], "distance_m": flat_equivalent_m(distance, dplus), "time_s": time_s,
                     "source": "course", "dplus_m": round(dplus)})

    hard = runs[runs["kind"].isin(HARD_SESSIONS - {"course"})]
    for col, dist in (("fastest_5k_s", 5000.0), ("fastest_10k_s", 10000.0)):
        if col in hard:
            for _, r in hard[hard[col].notna()].iterrows():
                source = f"meilleur {int(dist / 1000)} km ({r['kind']})"
                dplus = (r.get("elevation_gain_m") or 0) * dist / r["distance_m"]
                rows.append({"date": r["date"], "distance_m": flat_equivalent_m(dist, dplus), "time_s": r[col],
                             "source": source, "dplus_m": round(dplus)})

    # distance_m est une distance ÉQUIVALENTE SUR LE PLAT (D+ converti par l'équivalence de Scarf)
    perf = pd.DataFrame(rows, columns=["date", "distance_m", "time_s", "source", "dplus_m"])
    if perf.empty:
        return perf.assign(vdot=pd.Series(dtype=float))
    pace = perf["time_s"] / (perf["distance_m"] / 1000)
    perf = perf[pace.between(MIN_PACE_S, MAX_PACE_S)].copy()  # écarte les valeurs aberrantes
    perf["vdot"] = [vdot(d, t) for d, t in zip(perf["distance_m"], perf["time_s"])]
    return perf.sort_values("date").reset_index(drop=True)


def vo2max_history(activities: pd.DataFrame) -> pd.DataFrame:
    """Historique des VO2 max estimées par la montre (sorties de course en extérieur)."""
    v = activities.loc[activities["vo2max"].notna(), ["start_time", "vo2max"]].copy()
    v["date"] = pd.to_datetime(v["start_time"]).dt.normalize()
    return v[["date", "vo2max"]].sort_values("date").reset_index(drop=True)


def hr_speed_vo2max(activities: pd.DataFrame, labels: pd.DataFrame | None, hr_rest: float, hr_max: float) -> pd.DataFrame:
    """VO2 max estimée sur CHAQUE sortie régulière, à partir de la relation fréquence cardiaque / vitesse.

    Principe : le % de réserve cardiaque est proche du % de réserve de VO2 (relation de Swain).
    Le coût en O2 de la vitesse moyenne (formule de Daniels) rapporté à ce % donne une VO2 max.
    Le dénivelé est converti en vitesse équivalente sur le plat (GAP Garmin, sinon équivalence de Scarf).
    Exclusions : fractionné (moyennes trompeuses), sorties trop courtes ou trop faciles, trail très raide.
    """
    usable = (activities["sport"] == "running") & activities["avg_hr"].notna() & activities["avg_speed_ms"].notna()
    runs = activities[usable].copy()
    runs["kind"] = runs["activity_id"].map(session_types(labels))
    runs = runs[(runs["duration_s"] >= 15 * 60) & (runs["distance_m"] >= 2000) & (runs["kind"] != "fractionne")]
    runs = add_flat_speed(runs)
    runs = runs[runs["climb_m_per_km"] <= MAX_CLIMB_M_PER_KM]  # le dénivelé est corrigé, sauf trail très raide
    reserve = (runs["avg_hr"] - hr_rest) / (hr_max - hr_rest)
    runs = runs[reserve.between(0.5, 0.97)].assign(reserve=reserve)
    vo2 = runs["flat_speed_ms"].mul(60).map(vo2_at_speed)  # vitesse équivalente sur le plat
    runs["vo2max"] = 3.5 + (vo2 - 3.5) / runs["reserve"]  # 3,5 ml/kg/min : consommation au repos
    runs["date"] = pd.to_datetime(runs["start_time"]).dt.normalize()
    return runs[["date", "vo2max", "reserve", "kind"]].sort_values("date").reset_index(drop=True)


PHYSIO_SOURCES = {
    # nom : (fenêtre récente en jours, nombre minimal de mesures)
    "vo2max_montre": (120, 1),
    "relation_fc_vitesse": (56, 3),
}


def calibration_ratio(races: pd.DataFrame, series: pd.DataFrame) -> tuple[float, int]:
    """Rapport médian entre le VDOT réel des courses et l'estimation physiologique de l'époque."""
    if races.empty or series.empty:
        return 1.0, 0
    smooth = series.set_index("date")["vo2max"].rolling("28D").median().reset_index()
    matched = pd.merge_asof(races.sort_values("date"), smooth.sort_values("date"), on="date",
                            direction="backward", tolerance=pd.Timedelta(days=45))
    matched = matched[matched["vo2max"].notna()]
    if matched.empty:
        return 1.0, 0
    return float(np.clip(np.median(matched["vdot"] / matched["vo2max"]), *CALIBRATION_BOUNDS)), len(matched)


def estimate_vdot(perf: pd.DataFrame, physio: dict[str, pd.DataFrame], today: date) -> dict | None:
    """VDOT actuel : moyenne de toutes les estimations disponibles, chacune expliquée.

    - sources physiologiques (VO2 max de la montre, relation FC/vitesse sur toutes les sorties),
      chacune corrigée par un calibrage sur les courses réelles (économie de course personnelle) ;
    - meilleure performance, dépréciée selon son ancienneté.
    """
    today_ts = pd.Timestamp(today)
    races = perf[perf["source"] == "course"]
    components, estimates = {}, []

    for name, series in physio.items():
        window, min_n = PHYSIO_SOURCES[name]
        recent = series[series["date"] >= today_ts - pd.Timedelta(days=window)]
        if len(recent) < min_n:
            continue
        ratio, n_races = calibration_ratio(races, series)
        raw = float(recent["vo2max"].median())
        value = raw * ratio
        estimates.append(value)
        components[name] = {
            "vo2max_estimee": round(raw, 1),
            "mesures_utilisees": len(recent),
            "periode": f"{recent['date'].min().date()} au {recent['date'].max().date()}",
            "calibrage": round(ratio, 3),
            "courses_pour_calibrer": n_races,
            "vdot_estime": round(value, 1),
        }

    if len(perf):
        age = (today_ts - perf["date"]).dt.days.clip(lower=0)
        decayed = perf["vdot"] * (1 - (age * DECAY_PER_DAY).clip(upper=MAX_DECAY))
        best = perf.loc[decayed.idxmax()]
        estimates.append(float(decayed.max()))
        components["performances"] = {
            "date": best["date"].date().isoformat(),
            "source": best["source"],
            "distance_km": round(float(best["distance_m"]) / 1000, 2),
            "temps": format_time(best["time_s"]),
            "vdot_brut": round(float(best["vdot"]), 1),
            "vdot_deprecie": round(float(decayed.max()), 1),
            "nombre_performances": len(perf),
        }

    if not estimates:
        return None
    return {"vdot": float(np.mean(estimates)), "composantes": components, "courses_etiquetees": len(races)}


# --- Allures personnelles observées ---------------------------------------------------------------

SESSION_TYPES = ["ef", "tempo", "fractionne", "course"]
# Cibles de FC (% de la FC max) utilisées avec le modèle FC -> allure quand les séances observées manquent
HR_TARGETS = {"ef": (0.65, 0.75), "tempo": (0.84, 0.89), "fractionne": (0.90, 0.95)}
VDOT_EQUIVALENT = {"ef": "ef", "tempo": "seuil", "fractionne": "fractionne"}
MIN_OBSERVED = {"ef": 3, "tempo": 2, "fractionne": 2, "course": 1}


def recent_runs(activities: pd.DataFrame, labels: pd.DataFrame | None, today: date, window_days: int) -> pd.DataFrame:
    runs = activities[activities["sport"] == "running"].copy()
    runs["date"] = pd.to_datetime(runs["start_time"]).dt.normalize()
    runs = runs[runs["date"] >= pd.Timestamp(today) - pd.Timedelta(days=window_days)]
    runs["kind"] = runs["activity_id"].map(session_types(labels))
    moving = runs["moving_duration_s"].fillna(runs["duration_s"]) if "moving_duration_s" in runs else runs["duration_s"]
    runs["raw_pace_s"] = moving / (runs["distance_m"] / 1000)  # allure réelle, dénivelé compris
    runs = add_flat_speed(runs)
    runs["pace_s"] = 1000 / runs["flat_speed_ms"]  # allure équivalente sur le plat : comparable d'une sortie à l'autre
    return runs[runs["pace_s"].between(MIN_PACE_S, MAX_PACE_S)]


def observed_paces(runs: pd.DataFrame, hr_max: float) -> dict:
    """Allures et FC réellement pratiquées dans chaque type de séance (quartiles : la moitié centrale)."""
    out = {}
    for kind in SESSION_TYPES:
        g = runs[runs["kind"] == kind]
        # Fractionné : l'allure moyenne mélange efforts et récupérations -> on prend le meilleur km de la séance
        paces = g["fastest_1k_s"].dropna() if kind == "fractionne" else g["pace_s"]
        if paces.empty:
            continue
        hr = g["avg_hr"].dropna()
        out[kind] = {
            "seances": int(len(paces)),
            "allure_rapide": format_time(paces.quantile(0.25)) + "/km",
            "allure_lente": format_time(paces.quantile(0.75)) + "/km",
            "allure_rapide_s": round(float(paces.quantile(0.25))),
            "allure_lente_s": round(float(paces.quantile(0.75))),
            "allure_mediane": format_time(paces.median()) + "/km",
            "allure_mediane_s": round(float(paces.median())),
            "allure_reelle_mediane": format_time(g["raw_pace_s"].median()) + "/km",
            "dplus_median_m": round(float(g["dplus_m"].median())),
            "fc_moyenne": None if hr.empty else round(float(hr.median())),
            "fc_fourchette": None if hr.empty else [round(float(hr.quantile(0.25))), round(float(hr.quantile(0.75)))],
            "fc_pct_max": None if hr.empty else round(float(hr.median()) / hr_max * 100),
        }
    return out


def hr_pace_model(runs: pd.DataFrame) -> dict | None:
    """Relation personnelle FC -> vitesse, apprise sur les sorties régulières (hors fractionné)."""
    steady = runs[runs["kind"].isin(["ef", "tempo", "course"]) & runs["avg_hr"].notna()]
    if len(steady) < 5 or steady["avg_hr"].nunique() < 3:
        return None
    speed = 1000 / steady["pace_s"]  # m/s
    slope, intercept = np.polyfit(steady["avg_hr"], speed, 1)
    if slope <= 0:
        return None  # relation incohérente (chaleur, dénivelé...) : on ne s'en sert pas
    predicted = intercept + slope * steady["avg_hr"]
    r2 = 1 - ((speed - predicted) ** 2).sum() / ((speed - speed.mean()) ** 2).sum()
    return {"pente": float(slope), "ordonnee": float(intercept), "r2": float(r2), "sorties": int(len(steady))}


def pace_at_hr(model: dict, hr: float) -> float:
    return 1000 / (model["ordonnee"] + model["pente"] * hr)


def personal_training_paces(runs: pd.DataFrame, hr_max: float, vdot_value: float | None) -> dict:
    """Allures recommandées : observées si assez de séances, sinon modèle FC -> allure, sinon théorie VDOT."""
    observed = observed_paces(runs, hr_max)
    model = hr_pace_model(runs)
    zones = {}
    for kind in ["ef", "tempo", "fractionne"]:
        entry = {"observe": observed.get(kind)}
        low_pct, high_pct = HR_TARGETS[kind]
        if model:
            fast_s, slow_s = pace_at_hr(model, high_pct * hr_max), pace_at_hr(model, low_pct * hr_max)
            entry["modele_fc"] = {
                "fc_cible": [round(low_pct * hr_max), round(high_pct * hr_max)],
                "allure_rapide": format_time(fast_s) + "/km",
                "allure_lente": format_time(slow_s) + "/km",
                "allure_rapide_s": round(fast_s),
                "allure_lente_s": round(slow_s),
            }
        if vdot_value:
            low, high, _ = TRAINING_ZONES[VDOT_EQUIVALENT[kind]]
            fast_s, slow_s = pace_at_fraction(vdot_value, high), pace_at_fraction(vdot_value, low)
            entry["theorique_vdot"] = {
                "allure_rapide": format_time(fast_s) + "/km",
                "allure_lente": format_time(slow_s) + "/km",
                "allure_rapide_s": round(fast_s),
                "allure_lente_s": round(slow_s),
            }
        if entry["observe"] and entry["observe"]["seances"] >= MIN_OBSERVED[kind]:
            source = "observe"
        elif "modele_fc" in entry and kind != "fractionne":  # la FC réagit trop lentement sur des répétitions courtes
            source = "modele_fc"
        elif "theorique_vdot" in entry:
            source = "theorique_vdot"
        else:
            source = "observe" if entry["observe"] else None
        chosen = entry.get(source) or {}
        entry["recommandation"] = {
            "source": source,
            "allure_rapide": chosen.get("allure_rapide"),
            "allure_lente": chosen.get("allure_lente"),
            "allure_rapide_s": chosen.get("allure_rapide_s"),
            "allure_lente_s": chosen.get("allure_lente_s"),
            "fc_cible": chosen.get("fc_fourchette") or chosen.get("fc_cible")
            or (entry["modele_fc"]["fc_cible"] if "modele_fc" in entry else None),
        }
        zones[kind] = entry
    best_1k = runs["fastest_1k_s"].dropna()
    return {
        "allure_max": None if best_1k.empty else {
            "meilleur_1km": format_time(best_1k.min()) + "/km",
            "date": runs.loc[best_1k.idxmin(), "date"].date().isoformat(),
        },
        "modele_fc_allure": None if model is None else {"r2": round(model["r2"], 2), "sorties": model["sorties"]},
        "courses_observees": observed.get("course"),
        "zones": zones,
    }


def garmin_predictions(raw) -> dict:
    """Extrait les temps prédits par la montre (en secondes), quelle que soit la forme exacte du JSON."""
    if isinstance(raw, list):
        raw = raw[-1] if raw else {}
    if not isinstance(raw, dict):
        return {}
    keys = {"5k": ("time5k", "5k"), "10k": ("time10k", "10k"), "semi": ("timehalfmarathon", "half"),
            "marathon": ("timemarathon", "marathon")}
    out = {}
    for name, candidates in keys.items():
        for key, value in raw.items():
            k = key.lower()
            if isinstance(value, int | float) and k.startswith("time") and any(c in k for c in candidates):
                if name == "marathon" and "half" in k:
                    continue
                out[name] = float(value)
                break
    return out


# --- Forme du jour -------------------------------------------------------------------------------

def day_state(gold: pd.DataFrame) -> dict:
    """État du dernier jour connu : VFC et sommeil par rapport à la normale, fatigue accumulée."""
    g = gold.sort_values("date").reset_index(drop=True)
    g["hrv_28d"] = g["hrv_last_night"].rolling(28, min_periods=7).mean()
    g["sleep_28d"] = g["sleep_h"].rolling(28, min_periods=7).mean()
    last = g[g["hrv_last_night"].notna()].iloc[-1] if g["hrv_last_night"].notna().any() else g.iloc[-1]

    def num(x):
        return None if pd.isna(x) else float(x)

    hrv_rel = num((last["hrv_last_night"] - last["hrv_28d"]) / last["hrv_28d"]) if num(last["hrv_28d"]) else None
    tsb_rel = num(last["tsb"] / last["ctl"]) if num(last["ctl"]) else None
    return {
        "date": pd.Timestamp(last["date"]).date().isoformat(),
        "hrv": num(last["hrv_last_night"]),
        "hrv_normale_28j": num(last["hrv_28d"]),
        "hrv_ecart_pct": None if hrv_rel is None else round(hrv_rel * 100, 1),
        "sommeil_h": num(last["sleep_h"]),
        "sommeil_normal_h": num(last["sleep_28d"]),
        "fc_repos": num(last["resting_hr"]),
        "charge_aigue_atl": num(last["atl"]),
        "charge_chronique_ctl": num(last["ctl"]),
        "fraicheur_tsb": num(last["tsb"]),
        "fraicheur_relative": None if tsb_rel is None else round(tsb_rel, 2),
    }


def day_adjustment(state: dict) -> tuple[float, list[str]]:
    """Pénalité (ou bonus) en % du temps, avec ses raisons. Heuristique prudente, bornée à [-1 %, +3 %]."""
    adj, reasons = 0.0, []
    hrv = state.get("hrv_ecart_pct")
    if hrv is not None and hrv <= -20:
        adj += 0.02
        reasons.append(f"VFC {hrv:.0f} % sous ta normale : récupération nettement incomplète (+2 %)")
    elif hrv is not None and hrv <= -10:
        adj += 0.01
        reasons.append(f"VFC {hrv:.0f} % sous ta normale : récupération incomplète (+1 %)")
    sleep = state.get("sommeil_h")
    if sleep is not None and sleep < 6:
        adj += 0.005
        reasons.append(f"Nuit courte ({sleep:.1f} h) (+0,5 %)")
    tsb = state.get("fraicheur_relative")
    if tsb is not None and tsb < -0.3:
        adj += 0.01
        reasons.append("Fatigue accumulée élevée (fraîcheur très négative) (+1 %)")
    elif tsb is not None and tsb > 0.1 and (hrv is None or hrv >= 0):
        adj -= 0.005
        reasons.append("Bien reposé : fraîcheur positive et VFC normale (-0,5 %)")
    adj = float(np.clip(adj, -0.01, 0.03))
    if not reasons:
        reasons.append("Aucun signal particulier : forme du jour normale")
    return adj, reasons


def format_time(seconds: float) -> str:
    seconds = round(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}'{s:02d}\"" if h else f"{m}'{s:02d}\""
