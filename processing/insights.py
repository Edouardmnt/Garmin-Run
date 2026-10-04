"""Analyses rédigées : transforme les séries de la couche gold en phrases précises sur la personne.

Utilisé par l'API (/analyse) : le tableau de bord et le futur coach affichent exactement les mêmes analyses.
Les repères chiffrés (zones de l'ACWR, seuils de VFC) sont des repères usuels, pas des règles absolues.
"""

import pandas as pd

SPORT_NAMES = {"running": "la course", "tennis": "le tennis", "strength": "la musculation",
               "walking": "la marche", "hiking": "la randonnée", "other": "les autres activités"}


def _pct(new, old):
    if old is None or pd.isna(old) or old == 0 or new is None or pd.isna(new):
        return None
    return (new - old) / old * 100


def _signed(value, unit=" %", digits=0):
    return f"{value:+.{digits}f}{unit}".replace(".", ",")


def prepare(gold: pd.DataFrame) -> pd.DataFrame:
    df = gold.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["hrv_7j"] = df["hrv_last_night"].rolling(7, min_periods=3).mean()
    df["hrv_28j"] = df["hrv_last_night"].rolling(28, min_periods=7).mean()
    df["rhr_7j"] = df["resting_hr"].rolling(7, min_periods=3).mean()
    df["rhr_28j"] = df["resting_hr"].rolling(28, min_periods=7).mean()
    df["sleep_7j"] = df["sleep_h"].rolling(7, min_periods=3).mean()
    df["sleep_28j"] = df["sleep_h"].rolling(28, min_periods=7).mean()
    for col in ("bedtime_h", "avg_stress", "steps", "body_battery_max", "deep_sleep_s", "rem_sleep_s"):
        if col in df:
            df[f"{col}_28j"] = df[col].rolling(28, min_periods=7).median()
    return df


# --- Verdict du jour -----------------------------------------------------------------------------

def readiness(state: dict) -> dict:
    """Verdict du jour à partir de la VFC, du sommeil et de la fraîcheur. Score de 0 à 100."""
    score, reasons = 70, []
    hrv = state.get("hrv_ecart_pct")
    if hrv is not None:
        if hrv <= -20:
            score -= 35
            reasons.append(f"ta VFC est {abs(hrv):.0f} % sous ta normale : ton organisme n'a pas fini de récupérer")
        elif hrv <= -10:
            score -= 20
            reasons.append(f"ta VFC est {abs(hrv):.0f} % sous ta normale")
        elif hrv >= 5:
            score += 10
            reasons.append(f"ta VFC est {hrv:.0f} % au-dessus de ta normale")
        else:
            reasons.append("ta VFC est dans ta normale")
    sleep = state.get("sommeil_h")
    if sleep is not None:
        if sleep < 6:
            score -= 15
            reasons.append(f"ta nuit a été courte ({sleep:.1f} h)".replace(".", ","))
        elif sleep >= 7.5:
            score += 5
            reasons.append(f"tu as bien dormi ({sleep:.1f} h)".replace(".", ","))
    stress = state.get("stress_veille")
    if stress is not None and stress >= 50:
        score -= 10
        reasons.append(f"ta journée d'hier a été stressante (stress moyen {stress:.0f})")
    battery = state.get("body_battery_max")
    if battery is not None and battery < 50:
        score -= 10
        reasons.append(f"ta Body Battery n'est remontée qu'à {battery:.0f} cette nuit")
    if state.get("douleur_recente"):
        score = min(score, 60)
        reasons.append("tu as signalé une douleur après une sortie récente")
    tsb = state.get("fraicheur_relative")
    if tsb is not None:
        if tsb < -0.3:
            score -= 20
            reasons.append("ta fatigue accumulée est élevée")
        elif tsb > 0.1:
            score += 10
            reasons.append("tu es reposé par rapport à ton entraînement habituel")
    score = max(0, min(100, score))
    if score >= 70:
        level, title = "vert", "Feu vert"
        advice = "Bonne journée pour une séance de qualité : tempo, fractionné ou sortie longue."
    elif score >= 45:
        level, title, advice = "ambre", "Séance modérée", "Privilégie une séance en endurance fondamentale, sans forcer."
    else:
        level, title, advice = "rouge", "Récupération", "Repos ou footing très léger : ton corps a besoin de récupérer."
    sentence = ", ".join(reasons) if reasons else "pas assez de données pour juger ta forme du jour"
    return {"score": score, "niveau": level, "titre": title, "conseil": advice,
            "explication": sentence[0].upper() + sentence[1:] + "."}


# --- Analyses par graphique ----------------------------------------------------------------------

def load_insight(df: pd.DataFrame, sports: list[str]) -> dict:
    weekly = df[[f"load_{s}" for s in sports] + ["load_total"]].resample("W").sum()
    if len(weekly) < 2:
        return {"resume": "Pas encore assez de semaines pour analyser ta charge.", "points": []}
    last, previous = weekly.iloc[-1], weekly.iloc[-5:-1]
    change = _pct(last["load_total"], previous["load_total"].mean())
    recent = weekly.iloc[-4:]
    shares = {s: recent[f"load_{s}"].sum() / recent["load_total"].sum() * 100 for s in sports if recent["load_total"].sum()}
    main = max(shares, key=shares.get) if shares else None
    points = []
    if change is not None:
        trend = "plus" if change > 0 else "moins"
        points.append(f"Cette semaine, ta charge est {_signed(change)} par rapport à la moyenne des 4 semaines "
                      f"précédentes : tu t'entraînes {trend} que d'habitude.")
    if main:
        detail = ", ".join(f"{SPORT_NAMES.get(s, s)} {v:.0f} %" for s, v in sorted(shares.items(), key=lambda x: -x[1]) if v >= 5)
        points.append(f"Sur 4 semaines, {SPORT_NAMES.get(main, main)} représente la plus grande part de ta charge ({detail}).")
    acwr = df["acwr"].dropna().iloc[-1] if "acwr" in df and df["acwr"].notna().any() else None
    if acwr is not None:
        if acwr > 1.5:
            zone = "au-dessus de 1,5 : hausse brutale de la charge, le risque de blessure augmente"
        elif acwr > 1.3:
            zone = "entre 1,3 et 1,5 : progression rapide, à surveiller"
        elif acwr >= 0.8:
            zone = "entre 0,8 et 1,3 : la zone de progression habituellement recommandée"
        else:
            zone = "sous 0,8 : ta charge récente est basse par rapport à ton habitude"
        points.append(f"Ton ratio charge aiguë / chronique (ACWR) est de {acwr:.2f}".replace(".", ",") + f", {zone}.")
    resume = points[0] if points else "Charge stable."
    return {"resume": resume, "points": points}


def fitness_insight(df: pd.DataFrame) -> dict:
    ctl = df["ctl"].dropna()
    if len(ctl) < 28:
        return {"resume": "Il faut au moins 4 semaines de données pour suivre ta forme de fond.", "points": []}
    change = _pct(ctl.iloc[-1], ctl.iloc[-29])
    tsb, last_ctl = df["tsb"].iloc[-1], ctl.iloc[-1]
    rel = tsb / last_ctl if last_ctl else 0
    points = []
    if change is not None:
        verb = "progresse" if change > 3 else "baisse" if change < -3 else "est stable"
        points.append(f"Ta forme de fond (CTL) {verb} : {_signed(change)} en 4 semaines.")
    if rel > 0.1:
        state = "tu es frais : ta fatigue récente est inférieure à ta charge habituelle, idéal avant une course"
    elif rel >= -0.1:
        state = "tu es à l'équilibre entre fatigue récente et charge habituelle"
    elif rel >= -0.3:
        state = "tu es en phase de construction : un peu de fatigue, normale quand on progresse"
    else:
        state = "ta fatigue récente est nettement supérieure à ton habitude : prévois de la récupération"
    points.append(f"Ta fraîcheur (TSB) est de {tsb:.0f} : {state}.")
    return {"resume": points[0], "points": points}


def recovery_insight(df: pd.DataFrame) -> dict:
    last = df.dropna(subset=["hrv_7j"]).iloc[-1] if df["hrv_7j"].notna().any() else None
    if last is None or pd.isna(last["hrv_28j"]):
        return {"resume": "Pas assez de nuits suivies pour analyser ta récupération.", "points": []}
    points = []
    hrv_change = _pct(last["hrv_7j"], last["hrv_28j"])
    week = df["hrv_last_night"].iloc[-7:]
    low_nights = int((week < 0.9 * last["hrv_28j"]).sum())
    if hrv_change is not None:
        meaning = ("plutôt bon signe" if hrv_change > 3 else "un signe de fatigue ou de stress" if hrv_change < -5
                   else "dans ta normale")
        points.append(f"Ta VFC moyenne sur 7 jours est à {_signed(hrv_change)} de ta référence sur 28 jours : {meaning}.")
    points.append(f"{low_nights} nuit(s) sur les 7 dernières avec une VFC nettement sous ta normale (plus de 10 % en dessous).")
    rhr_change = None if pd.isna(last["rhr_28j"]) else last["rhr_7j"] - last["rhr_28j"]
    if rhr_change is not None:
        points.append(f"Ta FC de repos est à {_signed(rhr_change, ' bpm', 1)} de ta moyenne : "
                      + ("une hausse de plus de 3 bpm signale souvent une fatigue ou un début de maladie."
                         if rhr_change > 3 else "rien d'inhabituel."))
    return {"resume": points[0], "points": points}


def sleep_insight(df: pd.DataFrame) -> dict:
    week = df["sleep_h"].iloc[-7:].dropna()
    if week.empty:
        return {"resume": "Aucune nuit suivie ces 7 derniers jours.", "points": []}
    normal = df["sleep_28j"].dropna().iloc[-1] if df["sleep_28j"].notna().any() else None
    short = int((week < 6).sum())
    points = [f"Tu as dormi en moyenne {week.mean():.1f} h sur tes {len(week)} dernières nuits suivies".replace(".", ",")
              + ("" if normal is None else f" (ta normale : {normal:.1f} h)".replace(".", ",")) + "."]
    points.append(f"{short} nuit(s) de moins de 6 h cette semaine." if short else "Aucune nuit de moins de 6 h cette semaine.")
    return {"resume": points[0], "points": points}


def _fr(value: float, digits: int = 1) -> str:
    """Nombre au format français : virgule décimale."""
    return f"{value:.{digits}f}".replace(".", ",")


def _hhmm(hour: float) -> str:
    hour = hour % 24
    return f"{int(hour)}h{round((hour % 1) * 60):02d}"


def night_insight(df: pd.DataFrame) -> dict:
    """Analyse de la dernière nuit suivie : durée, phases, réveils, heure de coucher, VFC."""
    nights = df.dropna(subset=["sleep_h"])
    if nights.empty:
        return {"resume": "Aucune nuit suivie récemment.", "points": [], "nuit": None}
    n = nights.iloc[-1]
    total = n["sleep_s"] or 1
    phases = {k: (n.get(f"{k}_sleep_s") or 0) / total * 100 for k in ("deep", "light", "rem")}
    awake_min = (n.get("awake_s") or 0) / 60
    points = [f"Tu as dormi {_fr(n['sleep_h'])} h"
              + (f" (score de sommeil {n['sleep_score']:.0f}/100)" if pd.notna(n.get("sleep_score")) else "") + "."]
    deep_comment = ("c'est la phase de récupération physique, une bonne part" if phases["deep"] >= 15
                    else "un peu bas : c'est pourtant la phase de récupération physique")
    points.append(f"Sommeil profond {phases['deep']:.0f} % ({deep_comment}), paradoxal {phases['rem']:.0f} %, "
                  f"léger {phases['light']:.0f} %, éveillé {awake_min:.0f} min.")
    if pd.notna(n.get("bedtime_h")) and pd.notna(n.get("bedtime_h_28j")):
        shift = (n["bedtime_h"] - n["bedtime_h_28j"]) * 60
        when = "plus tard" if shift > 0 else "plus tôt"
        usual = _hhmm(n["bedtime_h_28j"])
        points.append(f"Coucher à {_hhmm(n['bedtime_h'])}, {abs(shift):.0f} min {when} que d'habitude ({usual})."
                      + (" Un coucher tardif réduit souvent le sommeil profond." if shift > 60 else ""))
    if pd.notna(n.get("hrv_last_night")) and pd.notna(n.get("hrv_28j")):
        ecart = _pct(n["hrv_last_night"], n["hrv_28j"])
        points.append(f"VFC de la nuit : {n['hrv_last_night']:.0f} ms, {_signed(ecart)} par rapport à ta normale.")
    night = {"date": nights.index[-1].date().isoformat(), "sommeil_h": round(float(n["sleep_h"]), 2),
             "profond_pct": round(phases["deep"]), "paradoxal_pct": round(phases["rem"]),
             "leger_pct": round(phases["light"]), "eveil_min": round(awake_min),
             "coucher": None if pd.isna(n.get("bedtime_h")) else _hhmm(n["bedtime_h"]),
             "lever": None if pd.isna(n.get("wake_h")) else _hhmm(n["wake_h"])}
    return {"resume": points[0], "points": points, "nuit": night}


def day_insight(df: pd.DataFrame) -> dict:
    """Analyse de la dernière journée : pas, stress, Body Battery."""
    day = df.iloc[-1]
    points = []
    if pd.notna(day.get("steps")):
        normal = day.get("steps_28j")
        compare = "" if pd.isna(normal) else f" ({_signed(_pct(day['steps'], normal))} par rapport à ta journée type)"
        points.append(f"{day['steps']:,.0f} pas".replace(",", " ") + compare + ".")
    if pd.notna(day.get("body_battery_max")):
        comment = ("Une recharge sous 50 indique une nuit peu réparatrice." if day["body_battery_max"] < 50
                   else "Bonne recharge nocturne." if day["body_battery_max"] >= 75 else "")
        points.append((f"Body Battery : rechargée jusqu'à {day['body_battery_max']:.0f} pendant la nuit, "
                       f"descendue à {day['body_battery_min']:.0f}. {comment}").strip())
    return {"resume": points[0] if points else "Pas de données pour la dernière journée.", "points": points}


def stress_insight(df: pd.DataFrame) -> dict:
    """Stress mesuré par la montre (0-25 repos, 26-50 faible, 51-75 moyen, 76-100 élevé)."""
    week = df["avg_stress"].iloc[-7:].dropna() if "avg_stress" in df else pd.Series(dtype=float)
    if week.empty:
        return {"resume": "Pas de mesure de stress récente.", "points": []}
    normal = df["avg_stress_28j"].dropna().iloc[-1] if df["avg_stress_28j"].notna().any() else None
    level = "repos" if week.mean() <= 25 else "faible" if week.mean() <= 50 else "moyen" if week.mean() <= 75 else "élevé"
    points = [f"Stress moyen de {week.mean():.0f} sur les 7 derniers jours (niveau {level})"
              + ("" if normal is None else f", contre {normal:.0f} habituellement") + "."]
    stressful = int((week > 50).sum())
    points.append(f"{stressful} journée(s) stressante(s) cette semaine (stress moyen au-dessus de 50)."
                  if stressful else "Aucune journée stressante cette semaine.")
    if "high_stress_min" in df and df["high_stress_min"].iloc[-7:].notna().any():
        points.append(f"En moyenne {df['high_stress_min'].iloc[-7:].mean():.0f} min par jour en stress élevé.")
    hrv = df["hrv_last_night"].shift(-1)
    pair = pd.concat([df["avg_stress"], hrv], axis=1).dropna().iloc[-60:]
    if len(pair) >= 20:
        corr = pair.corr().iloc[0, 1]
        if corr <= -0.2:
            points.append("Chez toi, les journées stressantes sont souvent suivies d'une VFC plus basse la nuit suivante "
                          f"(corrélation {_fr(corr, 2)}).")
    return {"resume": points[0], "points": points}


def activities_insight(activities: pd.DataFrame | None, today) -> dict:
    """Activités des 7 derniers jours, tous sports confondus."""
    if activities is None or activities.empty:
        return {"resume": "Aucune activité enregistrée.", "points": [], "semaine": []}
    acts = activities.copy()
    acts["date"] = pd.to_datetime(acts["start_time"]).dt.normalize()
    week = acts[acts["date"] > pd.Timestamp(today) - pd.Timedelta(days=7)]
    if week.empty:
        return {"resume": "Aucune activité ces 7 derniers jours : une semaine de repos.", "points": [], "semaine": []}
    by_sport = week.groupby("sport").agg(seances=("activity_id", "size"), minutes=("duration_min", "sum"),
                                         km=("distance_km", "sum"))
    parts = [f"{SPORT_NAMES.get(sp, sp)} {int(r.seances)} fois ({r.minutes:.0f} min)" for sp, r in by_sport.iterrows()]
    points = [f"Cette semaine : {len(week)} activités, {_fr(week['duration_min'].sum() / 60)} h au total"
              + " : " + ", ".join(parts) + "."]
    hardest = week.loc[week["aerobic_te"].fillna(0).idxmax()] if week["aerobic_te"].notna().any() else None
    if hardest is not None:
        sport = SPORT_NAMES.get(hardest["sport"], hardest["sport"])
        points.append(f"Séance la plus exigeante : {sport} du {hardest['date']:%d/%m}, "
                      f"effet d'entraînement aérobie {_fr(hardest['aerobic_te'])}/5.")
    running = by_sport.loc["running", "km"] if "running" in by_sport.index else 0
    points.append(f"{_fr(running)} km de course à pied sur la semaine.")
    semaine = [{"sport": sp, "seances": int(r.seances), "minutes": round(float(r.minutes))} for sp, r in by_sport.iterrows()]
    return {"resume": points[0], "points": points, "semaine": semaine}


def build_analysis(gold: pd.DataFrame, sports: list[str], state: dict, activities: pd.DataFrame | None = None) -> dict:
    df = prepare(gold)
    return {
        "verdict": readiness(state),
        "charge": load_insight(df, sports),
        "forme": fitness_insight(df),
        "recuperation": recovery_insight(df),
        "sommeil": sleep_insight(df),
        "nuit": night_insight(df),
        "journee": day_insight(df),
        "stress": stress_insight(df),
        "activites": activities_insight(activities, df.index.max()),
    }
