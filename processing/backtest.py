"""Backtest des prédictions de temps : l'application aurait-elle vu juste ?

Pour chaque performance réelle (course, meilleur 5 ou 10 km d'une séance dure), on se replace à la veille :
le niveau est estimé avec les SEULES données antérieures (VO2 max, relation FC/vitesse, performances,
charge, questionnaires), puis on prédit le temps sur la distance réellement courue (D+ inclus) et on le
compare au chrono. Aucune information du jour de la performance n'est utilisée : c'est la condition pour
que la mesure soit honnête.

Méthodes comparées :
- application    : la prédiction affichée (moyenne des estimations + correction des questionnaires) ;
- sans_correction: la même, sans la correction issue des questionnaires (mesure son apport) ;
- une colonne par source d'estimation (VO2 max montre, relation FC/vitesse, performances) ;
- riegel_derniere: référence naïve, la formule de Riegel appliquée à la dernière performance connue.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from processing.feedback import prediction_bias
from processing.performance import estimate_vdot, predict_time_s, riegel_time_s

MIN_DAYS_OF_HISTORY = 14  # sans au moins deux semaines de données avant, l'estimation n'a pas de sens


def estimate_before(day: pd.Timestamp, perf: pd.DataFrame, physio: dict, load: pd.Series | None) -> dict | None:
    """Niveau estimé la veille de `day`, avec les seules données strictement antérieures."""
    limit = day - pd.Timedelta(days=1)
    past = {name: s[s["date"] <= limit] for name, s in physio.items()}
    past_load = None if load is None else load[load.index <= limit]
    return estimate_vdot(perf[perf["date"] <= limit], past, limit.date(), past_load)


def backtest(perf: pd.DataFrame, physio: dict, load: pd.Series | None = None,
             feedback: list[dict] | None = None) -> pd.DataFrame:
    """Une ligne par performance testée : temps réel, temps prédit par chaque méthode, erreurs en %."""
    feedback = feedback or []
    if perf.empty:  # aucune performance (pas encore de course ni de séance dure étiquetée) : rien à tester
        return pd.DataFrame()
    first_day = min([s["date"].min() for s in physio.values() if len(s)] + [perf["date"].min()])
    rows = []
    for p in perf.sort_values("date").itertuples():
        if p.date < first_day + pd.Timedelta(days=MIN_DAYS_OF_HISTORY):
            continue
        estimate = estimate_before(p.date, perf, physio, load)
        if estimate is None:
            continue
        before = (p.date - timedelta(days=1)).date().isoformat()
        bias, _ = prediction_bias([f for f in feedback if f["date_sortie"] <= before])
        raw = predict_time_s(estimate["vdot"], p.distance_m)  # distance_m : équivalent plat (D+ inclus)
        row = {"date": p.date, "source": p.source, "distance_m": p.distance_m, "temps_reel_s": p.time_s,
               "vdot_estime": estimate["vdot"], "correction_pct": bias,
               "application": raw * (1 + bias / 100), "sans_correction": raw}
        for name, comp in estimate["composantes"].items():
            vdot = comp.get("vdot_estime", comp.get("vdot_deprecie"))
            row[name] = predict_time_s(vdot, p.distance_m) if vdot else np.nan
        previous = perf[perf["date"] < p.date].sort_values("date")
        row["riegel_derniere"] = (riegel_time_s(previous.iloc[-1].distance_m, previous.iloc[-1].time_s, p.distance_m)
                                  if len(previous) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


METHODS = ["application", "sans_correction", "vo2max_montre", "relation_fc_vitesse", "performances", "riegel_derniere"]


def metrics(results: pd.DataFrame) -> dict:
    """Erreur de chaque méthode : MAPE (erreur moyenne en %), biais (signe : + = trop pessimiste), part à ±3 %."""
    out = {}
    for method in METHODS:
        if method not in results or results[method].isna().all():
            continue
        valid = results[results[method].notna()]
        error = (valid[method] - valid["temps_reel_s"]) / valid["temps_reel_s"] * 100
        out[method] = {"n": int(len(valid)), "mape_pct": round(float(error.abs().mean()), 2),
                       "biais_pct": round(float(error.mean()), 2),
                       "erreur_mediane_s": round(float((valid[method] - valid["temps_reel_s"]).abs().median())),
                       "part_a_3pct": round(float((error.abs() <= 3).mean() * 100))}
    return out


def by_source(results: pd.DataFrame, method: str = "application") -> dict:
    """Erreur de l'application selon le type de performance (course, meilleur 5 km...)."""
    out = {}
    for source, group in results.groupby("source"):
        error = (group[method] - group["temps_reel_s"]) / group["temps_reel_s"] * 100
        out[source] = {"n": int(len(group)), "mape_pct": round(float(error.abs().mean()), 2),
                       "biais_pct": round(float(error.mean()), 2)}
    return out


def questionnaire_agreement(feedback: list[dict]) -> dict:
    """Ce que disent les questionnaires après course sur la justesse des prédictions."""
    answers = [f["reponses"].get("prediction") for f in feedback]
    answers = [a for a in answers if a in ("Trop optimiste", "Juste", "Trop pessimiste")]
    if not answers:
        return {"reponses": 0}
    return {"reponses": len(answers), "juste_pct": round(answers.count("Juste") / len(answers) * 100),
            "trop_optimiste": answers.count("Trop optimiste"), "trop_pessimiste": answers.count("Trop pessimiste")}
