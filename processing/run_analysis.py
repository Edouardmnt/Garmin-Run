"""Analyse d'une sortie de course : allure et fréquence cardiaque, kilomètre par kilomètre.

Indicateurs (repères usuels en entraînement, pas des règles absolues) :
- régularité : variation de l'allure entre les kilomètres, et comparaison 1re moitié / 2de moitié
  (négative split = 2de moitié plus rapide) ;
- dérive cardiaque (découplage allure / FC) : baisse de l'efficacité (vitesse / FC) entre la 1re et la
  2de moitié. Sous 5 % : endurance solide à cette allure ; 5 à 10 % : début de fatigue ; au-delà :
  allure trop élevée pour la durée, chaleur, déshydratation ou fatigue ;
- zones cardiaques : répartition du temps, comparée à ce qu'on attend du type de séance ;
- efficacité : mètres parcourus par battement cardiaque (allure ramenée sur le plat), comparée aux
  sorties récentes du même type : si elle monte, tu cours plus vite pour le même effort.

Dénivelé : chaque kilomètre est aussi exprimé en allure équivalente sur le plat (équivalence de Scarf :
1 m de montée = 7,92 m de plat). Régularité et dérive cardiaque se calculent sur cette allure, pour qu'une
côte ne passe pas pour un coup de fatigue. La descente n'est pas créditée (choix prudent).
"""

import numpy as np
import pandas as pd

from processing.performance import flat_equivalent_m

ZONE_NAMES = {1: "Zone 1 (récupération)", 2: "Zone 2 (endurance)", 3: "Zone 3 (tempo)", 4: "Zone 4 (seuil)",
              5: "Zone 5 (VMA)"}
# Zones où doit se trouver l'essentiel du temps, selon le type de séance
EXPECTED_ZONES = {"ef": {1, 2}, "longue": {1, 2}, "tempo": {3, 4}, "fractionne": {4, 5}, "course": {3, 4, 5}}
TYPE_NAMES = {"ef": "endurance fondamentale", "tempo": "tempo", "fractionne": "fractionné", "course": "course"}


def fmt_pace(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    m, s = divmod(round(seconds), 60)
    return f"{m}'{s:02d}\"/km"


def _fr(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _de(name: str) -> str:
    """« de tempo », « d'endurance fondamentale »."""
    return ("d'" if name[:1] in "aeiouéè" else "de ") + name


def _nth(n) -> str:
    n = int(n)
    return "1er" if n == 1 else f"{n}e"


def efficiency(distance_m, duration_s, avg_hr, dplus_m=0):
    """Mètres (équivalent plat) par battement cardiaque : plus c'est haut, plus tu es économe."""
    if not distance_m or not duration_s or not avg_hr:
        return None
    return flat_equivalent_m(distance_m, dplus_m or 0) / (duration_s / 60) / avg_hr


def laps_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Tours exploitables (au moins 300 m), avec allure et cumul de distance."""
    laps = laps.sort_values("lap").copy()
    laps = laps[(laps["distance_m"] >= 300) & (laps["duration_s"] > 0)]
    laps["pace_s"] = laps["duration_s"] / (laps["distance_m"] / 1000)
    gain = laps["elevation_gain_m"].astype(float).fillna(0) if "elevation_gain_m" in laps else 0.0
    laps["dplus"] = gain
    laps["flat_m"] = flat_equivalent_m(laps["distance_m"].astype(float), gain)
    laps["flat_pace_s"] = laps["duration_s"] / (laps["flat_m"] / 1000)  # allure équivalente sur le plat
    laps["km"] = laps["distance_m"].cumsum() / 1000
    return laps.reset_index(drop=True)


def halves(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = laps["distance_m"].sum() / 2
    first = laps[laps["distance_m"].cumsum() <= cut]
    return first, laps.drop(first.index)


def analyse_run(run: pd.Series, laps: pd.DataFrame, kind: str | None, history: pd.DataFrame,
                hr_max: float, reco: dict | None = None) -> dict:
    """Analyse complète d'une sortie. `history` : sorties récentes (avec colonnes kind, efficacite, date)."""
    duration = run.get("moving_duration_s") if pd.notna(run.get("moving_duration_s")) else run["duration_s"]
    pace = duration / (run["distance_m"] / 1000)
    dplus = run.get("elevation_gain_m") if pd.notna(run.get("elevation_gain_m")) else 0
    eff = efficiency(run["distance_m"], duration, run.get("avg_hr"), dplus)
    sections = {"allure": [], "zones": [], "efficacite": []}
    out = {}
    kind_name = TYPE_NAMES.get(kind, "sortie")

    out["resume"] = {
        "date": pd.Timestamp(run["start_time"]).strftime("%Y-%m-%d %H:%M"),
        "type": kind, "type_libelle": kind_name,
        "distance_km": round(run["distance_m"] / 1000, 2), "duree_min": round(duration / 60),
        "allure": fmt_pace(pace), "allure_s": round(pace),
        "fc_moyenne": None if pd.isna(run.get("avg_hr")) else int(run["avg_hr"]),
        "fc_max": None if pd.isna(run.get("max_hr")) else int(run["max_hr"]),
        "fc_pct_max": None if pd.isna(run.get("avg_hr")) else round(run["avg_hr"] / hr_max * 100),
        "denivele_m": round(dplus), "denivele_par_km": round(dplus / (run["distance_m"] / 1000), 1),
        "allure_plat": fmt_pace(duration / (flat_equivalent_m(run["distance_m"], dplus) / 1000)),
        "efficacite": None if eff is None else round(eff, 3),
    }

    # --- Kilomètre par kilomètre
    laps = laps_table(laps) if laps is not None and len(laps) else pd.DataFrame()
    if len(laps) and laps["dplus"].sum() > dplus:  # le D+ des tours est parfois plus complet que celui de la sortie
        out["resume"]["denivele_m"] = round(float(laps["dplus"].sum()))
    out["tours"] = [{"km": round(r.km, 2), "allure": fmt_pace(r.pace_s), "allure_s": round(r.pace_s),
                     "fc": None if pd.isna(r.avg_hr) else int(r.avg_hr),
                     "allure_plat": fmt_pace(r.flat_pace_s), "allure_plat_s": round(r.flat_pace_s),
                     "denivele_m": round(r.dplus)}
                    for r in laps.itertuples()]
    out["regularite"] = out["derive"] = None
    if len(laps) >= 4:
        main = laps[laps["distance_m"] >= 900] if (laps["distance_m"] >= 900).sum() >= 3 else laps
        cv = main["flat_pace_s"].std() / main["flat_pace_s"].mean() * 100
        first, second = halves(laps)
        p1 = first["duration_s"].sum() / first["flat_m"].sum() * 1000  # allures équivalentes sur le plat
        p2 = second["duration_s"].sum() / second["flat_m"].sum() * 1000
        fast, slow = main.loc[main["flat_pace_s"].idxmin()], main.loc[main["flat_pace_s"].idxmax()]
        out["regularite"] = {"variation_pct": round(cv, 1), "allure_1re_moitie": fmt_pace(p1),
                             "allure_2de_moitie": fmt_pace(p2), "ecart_s": round(p2 - p1),
                             "km_le_plus_rapide": int(fast.lap), "km_le_plus_lent": int(slow.lap)}
        steady = "très régulière" if cv < 3 else "régulière" if cv < 6 else "irrégulière"
        climb_total = max(dplus, float(laps["dplus"].sum()))  # D+ des tours si plus complet que celui de la sortie
        hilly = climb_total / (run["distance_m"] / 1000) >= 8
        basis = " en équivalent plat" if hilly else ""
        if p2 < p1 - 3:
            split = f"tu as fini plus vite ({fmt_pace(p2)} contre {fmt_pace(p1)}) : un négative split, signe de bonne gestion"
        elif p2 > p1 + 8:
            split = (f"tu as ralenti en 2de moitié ({fmt_pace(p2)} contre {fmt_pace(p1)}) : "
                     + ("départ sans doute trop rapide" if kind in ("course", "tempo") else "fatigue en fin de sortie"))
        else:
            split = f"tes deux moitiés sont proches ({fmt_pace(p1)} puis {fmt_pace(p2)})"
        extremes = ""
        if cv >= 1.5:  # sans écart notable, inutile de citer un kilomètre plus rapide qu'un autre
            extremes = (f" Kilomètre le plus rapide : le {_nth(fast.lap)} en {fmt_pace(fast.flat_pace_s)} ; "
                        f"le plus lent : le {_nth(slow.lap)} en {fmt_pace(slow.flat_pace_s)}.")
        sections["allure"].append(f"Allure {steady}{basis} (variation de {_fr(cv)} % d'un kilomètre à l'autre) ; "
                                  f"{split}.{extremes}")
        if hilly:
            climb = laps.loc[laps["dplus"].idxmax()]
            sections["allure"].insert(0, (
                f"Parcours vallonné : {round(climb_total)} m de D+ ({_fr(climb_total / (run['distance_m'] / 1000), 0)} m/km), "
                f"soit une allure de {fmt_pace(pace)} qui vaut "
                f"{fmt_pace(duration / (flat_equivalent_m(run['distance_m'], climb_total) / 1000))} sur le plat. Le kilomètre le "
                f"plus raide, le {_nth(climb.lap)} (+{round(climb.dplus)} m), couru en {fmt_pace(climb.pace_s)}, "
                f"équivaut à {fmt_pace(climb.flat_pace_s)} sur le plat. Les allures ci-dessous en tiennent compte."))

        # Dérive cardiaque : efficacité (vitesse / FC) 1re moitié contre 2de moitié
        if first["avg_hr"].notna().all() and second["avg_hr"].notna().all() and kind != "fractionne":
            def ef(part):
                hr = (part["avg_hr"] * part["duration_s"]).sum() / part["duration_s"].sum()
                return part["flat_m"].sum() / part["duration_s"].sum() / hr
            drift = (ef(first) - ef(second)) / ef(first) * 100
            hr1 = (first["avg_hr"] * first["duration_s"]).sum() / first["duration_s"].sum()
            hr2 = (second["avg_hr"] * second["duration_s"]).sum() / second["duration_s"].sum()
            level = "faible" if drift < 5 else "modérée" if drift < 10 else "forte"
            out["derive"] = {"decouplage_pct": round(drift, 1), "niveau": level,
                             "fc_1re_moitie": round(hr1), "fc_2de_moitie": round(hr2)}
            meaning = {"faible": "ton endurance tient bien à cette allure",
                       "modérée": "la fatigue s'installe en fin de sortie : normal sur une sortie longue ou chaude",
                       "forte": "allure trop élevée pour cette durée, ou chaleur, déshydratation, fatigue"}[level]
            sections["allure"].append(f"Dérive cardiaque {level} ({_fr(drift)} %) : ta FC est passée de {round(hr1)} à "
                          f"{round(hr2)} bpm entre les deux moitiés ; {meaning}.")

    # --- Zones cardiaques
    zones = {z: run.get(f"hr_z{z}_s") for z in range(1, 6)}
    total = sum(v for v in zones.values() if v and pd.notna(v))
    out["zones"] = None
    if total:
        share = {z: (zones[z] or 0) / total * 100 for z in zones}
        out["zones"] = [{"zone": z, "nom": ZONE_NAMES[z], "pct": round(p), "minutes": round((zones[z] or 0) / 60)}
                        for z, p in share.items()]
        main_zone = max(share, key=share.get)
        expected = EXPECTED_ZONES.get(kind)
        txt = f"Tu as passé {round(share[main_zone])} % du temps en {ZONE_NAMES[main_zone].lower()}"
        if expected:
            inside = sum(share[z] for z in expected)
            if kind in ("ef", "longue") and share[3] + share[4] + share[5] > 30:
                txt += (f", mais {round(share[3] + share[4] + share[5])} % au-dessus de la zone 2 : pour une "
                        "endurance fondamentale, c'est trop intense. Ralentis pour vraiment récupérer et construire ta base")
            elif inside >= 60:
                txt += f" : {round(inside)} % du temps dans les zones attendues pour une séance {_de(kind_name)}"
            else:
                txt += f" : seulement {round(inside)} % dans les zones attendues pour une séance {_de(kind_name)}"
        sections["zones"].append(txt + ".")

    # --- Conformité à l'allure conseillée
    out["conformite"] = None
    if reco and reco.get("allure_rapide_s") and reco.get("allure_lente_s") and kind in ("ef", "tempo"):
        fast, slow = reco["allure_rapide_s"], reco["allure_lente_s"]
        # Les allures conseillées sont en équivalent plat : on compare l'allure de la sortie ramenée sur le plat
        flat_pace = duration / (flat_equivalent_m(run["distance_m"], dplus) / 1000)
        verdict = ("dans ta fourchette" if fast - 5 <= flat_pace <= slow + 5
                   else "plus rapide" if flat_pace < fast else "plus lente")
        out["conformite"] = {"fourchette": f"{fmt_pace(fast)} à {fmt_pace(slow)}", "verdict": verdict,
                             "rapide_s": round(fast), "lente_s": round(slow)}
        sections["allure"].append(f"Allure {verdict} que tes allures {_de(kind_name)} ({fmt_pace(fast)} à {fmt_pace(slow)})"
                      if verdict != "dans ta fourchette"
                      else f"Allure dans ta fourchette {_de(kind_name)} ({fmt_pace(fast)} à {fmt_pace(slow)}).")
        if not sections["allure"][-1].endswith("."):
            sections["allure"][-1] += "."

    # --- Efficacité comparée aux sorties récentes du même type
    out["comparaison"] = None
    same = history[(history["kind"] == kind) & (history["activity_id"] != run["activity_id"])
                   & (history["date"] < pd.Timestamp(run["start_time"]))].tail(8) if kind else history.iloc[0:0]
    if eff and len(same) >= 3 and same["efficacite"].notna().sum() >= 3:
        ref = same["efficacite"].median()
        change = (eff - ref) / ref * 100
        out["comparaison"] = {"sorties": int(len(same)), "variation_pct": round(change, 1)}
        if abs(change) < 2:
            txt = "dans la moyenne de tes"
        elif change > 0:
            txt = "meilleure que la moyenne de tes"
        else:
            txt = "moins bonne que la moyenne de tes"
        sections["efficacite"].append(f"Efficacité (distance parcourue par battement, sur le plat) {txt} {len(same)} dernières "
                      f"sorties {_de(kind_name)} ({'+' if change >= 0 else ''}{_fr(change)} %)"
                      + (" : tu cours plus vite pour le même effort cardiaque." if change >= 2 else
                         " : fatigue, chaleur ou terrain peuvent l'expliquer." if change <= -2 else "."))
    out["sections"] = sections
    out["points"] = [p for part in sections.values() for p in part]
    return out


def efficiency_history(runs: pd.DataFrame) -> pd.DataFrame:
    """Efficacité de chaque sortie, pour suivre sa tendance (sorties de 20 min et plus, hors fractionné)."""
    r = runs.copy()
    duration = r["moving_duration_s"].fillna(r["duration_s"]) if "moving_duration_s" in r else r["duration_s"]
    dplus = r["elevation_gain_m"].fillna(0) if "elevation_gain_m" in r else 0
    r["efficacite"] = [efficiency(d, t, h, e) for d, t, h, e in
                       zip(r["distance_m"], duration, r["avg_hr"], np.broadcast_to(dplus, len(r)))]
    r.loc[(duration < 20 * 60) | (r["kind"] == "fractionne"), "efficacite"] = None
    return r
