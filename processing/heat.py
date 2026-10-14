"""Effet de la chaleur sur la course : indice WBGT, pénalité de performance, sensibilité personnelle apprise.

1. Indice de chaleur. Le WBGT (« wet bulb globe temperature ») combine température et humidité : c'est l'indice
   utilisé par World Athletics. Sans mesure du rayonnement, on utilise l'approximation du Bureau of Meteorology
   australien : WBGT ≈ 0,567 × T + 0,393 × e + 3,94, avec e la pression de vapeur d'eau (hPa).

2. Pénalité de référence (littérature). Ely et al. (2007, Medicine & Science in Sports & Exercise) : sur 140
   marathons, les meilleurs coureurs perdent de 1,7 % (WBGT 5-10 °C) à 4,5 % (20-25 °C) par rapport au record,
   soit environ 0,19 % par degré au-dessus de 10 °C, et les coureurs moins rapides perdent davantage.
   Une analyse de 1 258 courses (2022) situe les meilleures performances entre 7,5 et 15 °C de WBGT.
   Référence retenue pour un coureur amateur : 0,3 % par degré de WBGT au-dessus de 10 °C.

3. Sensibilité personnelle. Sur tes sorties régulières, la VO2 max estimée par la relation FC/vitesse baisse
   quand il fait chaud (à allure égale, la FC monte). L'écart de chaque sortie à ton niveau du moment, rapporté
   au WBGT, donne ta sensibilité. Ta mesure et la référence sont combinées selon leur précision (moyenne
   pondérée par l'inverse des variances) : une mesure bruitée ou fondée sur peu de sorties chaudes compte peu.
"""

import math

import numpy as np
import pandas as pd

REFERENCE_WBGT_C = 10.0       # en dessous, pas de pénalité : conditions idéales
PRIOR_PCT_PER_C = 0.3         # référence pour un coureur amateur (élites : ~0,19 %/°C, Ely 2007)
PRIOR_SD = 0.2                # incertitude sur la référence (%/°C) : sert à pondérer ta propre mesure
MIN_HEAT_C = 3.0              # une sortie « chaude » : WBGT au moins 3 °C au-dessus de la référence
MAX_PCT_PER_C = 1.5           # garde-fou contre une estimation aberrante
TREND_WINDOW_DAYS = 42        # niveau du moment : médiane des estimations sur ±3 semaines
DEFAULT_RACE_WEATHER = {"temperature_c": 15.0, "humidite_pct": 65.0}  # sans prévision : conditions moyennes


def vapor_pressure_hpa(temperature_c: float, humidity_pct: float) -> float:
    return humidity_pct / 100 * 6.105 * math.exp(17.27 * temperature_c / (237.7 + temperature_c))


def wbgt_simplified(temperature_c: float, humidity_pct: float) -> float:
    """WBGT approché à l'ombre (Bureau of Meteorology australien)."""
    return 0.567 * temperature_c + 0.393 * vapor_pressure_hpa(temperature_c, humidity_pct) + 3.94


def heat_penalty_pct(wbgt_c: float | None, pct_per_c: float = PRIOR_PCT_PER_C) -> float:
    """Temps allongé de ce pourcentage par rapport à des conditions idéales."""
    if wbgt_c is None or (isinstance(wbgt_c, float) and math.isnan(wbgt_c)):
        return 0.0
    return max(0.0, wbgt_c - REFERENCE_WBGT_C) * pct_per_c


def heat_level(wbgt_c: float) -> str:
    """Catégories inspirées de World Athletics, en mots simples."""
    if wbgt_c < 10:
        return "frais, conditions idéales"
    if wbgt_c < 18:
        return "doux"
    if wbgt_c < 23:
        return "chaud"
    if wbgt_c < 28:
        return "très chaud"
    return "chaleur extrême"


def learn_sensitivity(steady: pd.DataFrame) -> dict:
    """Sensibilité à la chaleur (% de performance par °C de WBGT), apprise sur les sorties régulières.

    `steady` : une ligne par sortie, colonnes date, vo2max (relation FC/vitesse, non corrigée) et wbgt_c.
    """
    column = "vo2max_mesuree" if "vo2max_mesuree" in steady else "vo2max"  # toujours la valeur non corrigée
    if "wbgt_c" not in steady:
        steady = steady.assign(wbgt_c=np.nan)
    runs = steady.dropna(subset=[column, "wbgt_c"]).sort_values("date").assign(vo2max=lambda d: d[column])
    result = {"pct_par_degre": PRIOR_PCT_PER_C, "sorties": int(len(runs)), "sorties_chaudes": 0,
              "mesure_brute": None, "reference": PRIOR_PCT_PER_C, "origine": "reference"}
    if len(runs) < 6:
        return result
    dated = runs.set_index("date")
    trend = dated["vo2max"].rolling(f"{TREND_WINDOW_DAYS}D", center=True, min_periods=3).median()
    deviation = (dated["vo2max"] / trend - 1) * 100            # % au-dessus ou en dessous du niveau du moment
    heat = (dated["wbgt_c"] - REFERENCE_WBGT_C).clip(lower=0)
    valid = deviation.notna()
    x, y = heat[valid].to_numpy(float), deviation[valid].to_numpy(float)
    n_hot = int((x >= MIN_HEAT_C).sum())
    result["sorties_chaudes"] = n_hot
    if n_hot < 2 or np.ptp(x) < MIN_HEAT_C:
        return result
    slope, intercept = np.polyfit(x, y, 1)                     # négatif : la VO2 estimée baisse avec la chaleur
    residuals = y - (slope * x + intercept)
    se = float(np.sqrt(residuals.var(ddof=2) / ((x - x.mean()) ** 2).sum()))  # erreur type de la pente
    measured = float(np.clip(-slope, -MAX_PCT_PER_C, MAX_PCT_PER_C))
    w_own, w_prior = 1 / se ** 2, 1 / PRIOR_SD ** 2
    combined = (w_own * measured + w_prior * PRIOR_PCT_PER_C) / (w_own + w_prior)
    result.update({"pct_par_degre": round(float(np.clip(combined, 0.0, MAX_PCT_PER_C)), 3),
                   "mesure_brute": round(measured, 3), "erreur_type": round(se, 3),
                   "poids_mesure_pct": round(w_own / (w_own + w_prior) * 100), "origine": "personnelle"})
    return result


def attach_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le WBGT de chaque sortie (colonne wbgt_c, vide sans météo) et garde la valeur mesurée à part."""
    out = df.drop(columns=["wbgt_c"], errors="ignore").merge(
        weather[["activity_id", "wbgt_c"]].drop_duplicates("activity_id"), on="activity_id", how="left")
    if "vo2max" in out and "vo2max_mesuree" not in out:
        out["vo2max_mesuree"] = out["vo2max"]
    if "time_s" in out and "time_s_mesure" not in out:
        out["time_s_mesure"] = out["time_s"]
    return out


def normalize(perf: pd.DataFrame, steady: pd.DataFrame, pct_per_c: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ramène performances et VO2 max estimées à des conditions idéales : la chaleur les avait dégradées.

    Une performance courue à 24 °C de WBGT vaut mieux que son chrono : on la compare à ce qu'elle aurait
    donné au frais. Le temps de course est ensuite allongé selon la météo prévue le jour J.
    """
    from processing.performance import vdot

    def penalty(df):
        return df["wbgt_c"].map(lambda w: heat_penalty_pct(w, pct_per_c)) if "wbgt_c" in df else 0.0

    perf, steady = perf.copy(), steady.copy()
    if len(perf) and "time_s_mesure" in perf:
        perf["time_s"] = perf["time_s_mesure"] / (1 + penalty(perf) / 100)
        perf["vdot"] = [vdot(d, t) for d, t in zip(perf["distance_m"], perf["time_s"])]
    if len(steady) and "vo2max_mesuree" in steady:
        steady["vo2max"] = steady["vo2max_mesuree"] * (1 + penalty(steady) / 100)
    return perf, steady


def raw(perf: pd.DataFrame, steady: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Les mêmes données sans correction de chaleur (pour comparer dans le backtest)."""
    return normalize(perf, steady, 0.0)


def race_weather_penalty(temperature_c: float | None, humidity_pct: float | None, pct_per_c: float) -> dict:
    """Effet de la météo prévue sur un temps de course."""
    weather = {"temperature_c": temperature_c if temperature_c is not None else DEFAULT_RACE_WEATHER["temperature_c"],
               "humidite_pct": humidity_pct if humidity_pct is not None else DEFAULT_RACE_WEATHER["humidite_pct"]}
    wbgt = wbgt_simplified(weather["temperature_c"], weather["humidite_pct"])
    return {**weather, "wbgt_c": round(wbgt, 1), "niveau": heat_level(wbgt),
            "penalite_pct": round(heat_penalty_pct(wbgt, pct_per_c), 1),
            "par_defaut": temperature_c is None}
