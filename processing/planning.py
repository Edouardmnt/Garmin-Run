"""Planning d'entraînement jusqu'à une course, construit à partir des données de la personne.

Principes (périodisation classique, volontairement simple et explicable) :
- phases : développement, spécifique (4 semaines), affûtage, semaine de course ;
- une sortie longue, une à deux séances de qualité, le reste en endurance fondamentale ;
- jamais de séance dure la veille ou le lendemain d'un tennis, ni deux séances dures d'affilée ;
- volume de départ = volume réel des 4 dernières semaines, progression d'environ 7 % par semaine ;
- allures = allures personnelles (observées, modèle FC, ou VDOT), allure course = temps prédit ;
- séance du jour adaptée au verdict de forme (feu vert, séance modérée, récupération).
"""

import math
from datetime import date, timedelta

DAY_NAMES = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
DISTANCE_LABELS = {"5k": "5 km", "10k": "10 km", "semi": "semi-marathon", "marathon": "marathon"}
MIN_WEEKLY_KM = {"5k": 15, "10k": 20, "semi": 25, "marathon": 35}
MAX_WEEKLY_KM = {"5k": 45, "10k": 55, "semi": 65, "marathon": 85}
LONG_RUN_CAP_KM = {"5k": 12, "10k": 16, "semi": 21, "marathon": 32}
LONG_RUN_MIN_KM = {"5k": 8, "10k": 10, "semi": 14, "marathon": 20}  # puis +1 km par semaine jusqu'au plafond
HARD = {"tempo", "fractionne", "specifique", "longue", "course"}

# Séances de qualité selon la distance visée : (répétitions de départ, distance d'une répétition en m, récupération)
INTERVALS = {"5k": (6, 400, "1 min 30 de trot"), "10k": (5, 1000, "2 min de trot"),
             "semi": (4, 1600, "2 min de trot"), "marathon": (3, 2000, "3 min de trot")}
SPECIFIC = {"5k": (3, 1500), "10k": (3, 2000), "semi": (2, 5000), "marathon": (2, 8000)}


def fmt_pace(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(round(seconds), 60)
    return f"{m}'{s:02d}\"/km"


def category_for(km: float) -> str:
    """Famille d'entraînement d'une distance libre : la structure du plan suit la distance classique la plus proche."""
    if km <= 7.5:
        return "5k"
    if km <= 15:
        return "10k"
    if km <= 30:
        return "semi"
    return "marathon"


def phase_for(weeks_left: int | None) -> str:
    if weeks_left is None:
        return "Développement"
    if weeks_left == 0:
        return "Semaine de course"
    if weeks_left == 1:
        return "Affûtage"
    if weeks_left <= 4:
        return "Spécifique"
    return "Développement"


def weekly_volume(base_km: float, distance: str, week_index: int, phase: str) -> float:
    start = max(base_km, MIN_WEEKLY_KM[distance])
    if phase == "Développement":
        km = start * 1.07 ** week_index
    else:  # le volume atteint en fin de développement est maintenu, puis réduit pour l'affûtage
        km = start * 1.07 ** max(0, week_index - 1)
    km = min(km, start * 1.3, MAX_WEEKLY_KM[distance])
    return round(km * {"Affûtage": 0.7, "Semaine de course": 0.5}.get(phase, 1.0), 1)


def place_days(available: list[int], long_day: int, n_hard: int, n_easy: int, tennis: set[int]) -> dict[int, str]:
    """Répartit les séances dans la semaine : dures loin du tennis, de la sortie longue et entre elles."""
    plan = {long_day: "longue"}
    free = [d for d in available if d != long_day]

    def penalty(day: int, hard_days: set[int]) -> int:
        neighbours = {(day - 1) % 7, (day + 1) % 7}
        long_run = 1 if long_day in neighbours else 0  # la veille ou le lendemain de la sortie longue
        return 3 * len(neighbours & tennis) + 2 * len(neighbours & (hard_days - {long_day})) + 3 * long_run

    def hard_days() -> set[int]:
        return {d for d, t in plan.items() if t in HARD | {"qualite"}}

    for i in range(n_hard):
        if not free:
            break
        best = min(free, key=lambda d: (penalty(d, hard_days()), -abs(d - long_day)))
        if i > 0 and penalty(best, hard_days()) >= 5:
            n_easy += 1  # une 2e séance dure enchaînerait avec le tennis ou une autre séance dure : footing à la place
            break
        plan[best] = "qualite"
        free.remove(best)
    for day in sorted(free, key=lambda d: penalty(d, hard_days()))[:n_easy]:
        plan[day] = "ef"
    return plan


WARMUP_MIN = 25  # échauffement 15 min + retour au calme 10 min, en endurance fondamentale


def quality_volume(work_km: float, work_pace_s: float | None, n_recoveries: int, recovery_min: float,
                   ef_pace_s: float) -> tuple[float, float]:
    """Distance (km) et durée (min) totales d'une séance : échauffement, travail, récupérations, retour au calme."""
    work_pace_s = work_pace_s or ef_pace_s
    minutes = WARMUP_MIN + work_km * work_pace_s / 60 + n_recoveries * recovery_min
    km = work_km + (WARMUP_MIN + n_recoveries * recovery_min) * 60 / (ef_pace_s * 1.1)  # récup un peu plus lente
    return round(km, 1), round(minutes)


def pace_range(p: dict) -> str:
    return f"{fmt_pace(p.get('rapide_s'))} – {fmt_pace(p.get('lente_s'))}"


# --- Étapes structurées : exploitables par la montre ---------------------------------------------

def step(kind: str, duree_s: float | None = None, distance_m: float | None = None, allure: dict | None = None) -> dict:
    """Une étape : "echauffement", "effort", "recuperation" ou "retour_au_calme", avec sa cible d'allure (s/km)."""
    target = None
    if allure and allure.get("rapide_s") and allure.get("lente_s"):
        target = {"rapide_s": round(allure["rapide_s"]), "lente_s": round(allure["lente_s"])}
    return {"type": kind, "duree_s": None if duree_s is None else round(duree_s),
            "distance_m": None if distance_m is None else round(distance_m), "allure": target}


def repeat(times: int, steps: list[dict]) -> dict:
    return {"type": "repetition", "repetitions": times, "etapes": steps}


def around(pace_s: float, margin_s: int = 3) -> dict:
    """Fourchette serrée autour d'une allure précise (allure de course)."""
    return {"rapide_s": pace_s - margin_s, "lente_s": pace_s + margin_s}


WARMUP_S, COOLDOWN_S = 15 * 60, 10 * 60


def session(day: date, kind: str, title: str, description: str, km: float, pace_s: float | None,
            pace_txt: str, hr: list | None, goal: str, minutes: float | None = None) -> dict:
    if minutes is None and pace_s is not None:
        minutes = km * pace_s / 60
    return {
        "date": day.isoformat(), "jour": DAY_NAMES[day.weekday()], "type": kind, "titre": title,
        "description": description, "distance_km": round(km, 1),
        "duree_min": None if minutes is None else round(minutes),
        "allure": pace_txt, "fc_cible": None if not hr else f"{hr[0]}–{hr[1]} bpm", "objectif": goal,
    }


def build_plan(today: date, distance: str, race_date: date | None, sessions_per_week: int, tennis_days: list[int],
               long_day: int, base_weekly_km: float, paces: dict, race_pace_s: float, verdict: str = "vert",
               volume_factor: float = 1.0, ef_shift_s: int = 0, race_km: float | None = None,
               race_label: str | None = None) -> dict:
    """paces : {"ef"|"tempo"|"fractionne": {"rapide_s", "lente_s", "fc_cible"}}.

    volume_factor : réduction du volume décidée à partir des signaux récents (douleur, sommeil, stress) ;
    ef_shift_s    : secondes ajoutées aux allures d'endurance fondamentale (footings ressentis trop durs) ;
    race_km, race_label : distance réelle de la course (distance libre) ; `distance` donne la famille d'entraînement.
    """
    tennis = set(tennis_days)
    available = [d for d in range(7) if d not in tennis]
    if long_day in tennis:
        long_day = max(available) if available else 6
    week_start = today - timedelta(days=today.weekday())
    n_weeks = 2 if race_date is None else max(1, min(20, math.ceil((race_date - week_start).days / 7 + 0.01)))

    ef, tempo, frac = (paces.get(k, {}) for k in ("ef", "tempo", "fractionne"))
    if ef_shift_s and ef.get("rapide_s") is not None:
        ef = {**ef, "rapide_s": ef["rapide_s"] + ef_shift_s, "lente_s": ef["lente_s"] + ef_shift_s}
    ef_mid = (ef.get("rapide_s", 360) + ef.get("lente_s", 390)) / 2
    label = race_label or DISTANCE_LABELS[distance]
    race_distance_km = race_km or {"5k": 5, "10k": 10, "semi": 21.1, "marathon": 42.2}[distance]
    weeks, notes = [], []

    for w in range(n_weeks):
        weeks_left = None if race_date is None else n_weeks - 1 - w
        phase = phase_for(weeks_left)
        volume = weekly_volume(base_weekly_km, distance, w, phase) * volume_factor
        monday = week_start + timedelta(weeks=w)
        sessions = []

        if phase == "Semaine de course":
            race_day = race_date
            sharpen = race_day - timedelta(days=2)
            for d in sorted({sharpen, race_day} | ({race_day - timedelta(days=4)} if sessions_per_week >= 3 else set())):
                if d == race_day:
                    advice = (f"Vise une allure régulière de {fmt_pace(race_pace_s)}, "
                              "en partant légèrement plus lentement sur le premier kilomètre.")
                    sessions.append(session(d, "course", f"Course : {label}", advice,
                                            race_distance_km, race_pace_s,
                                            fmt_pace(race_pace_s), None, "Le jour J."))
                    sessions[-1]["etapes"] = [step("effort", distance_m=sessions[-1]["distance_km"] * 1000,
                                                   allure=around(race_pace_s, 5))]
                elif d == sharpen:
                    sessions.append(session(d, "ef", "Footing d'activation", "20 min en endurance fondamentale, puis 4 lignes "
                                            "droites de 80 m en accélérant progressivement.", 4, ef_mid,
                                            pace_range(ef),
                                            ef.get("fc_cible"), "Garder des jambes vives sans se fatiguer."))
                    sessions[-1]["etapes"] = [step("effort", duree_s=20 * 60, allure=ef),
                                              repeat(4, [step("effort", distance_m=80), step("recuperation", duree_s=60)])]
                else:
                    sessions.append(session(d, "ef", "Footing facile", "30 min en endurance fondamentale.", 5, ef_mid,
                                            pace_range(ef),
                                            ef.get("fc_cible"), "Entretenir sans fatigue avant la course."))
                    sessions[-1]["etapes"] = [step("effort", duree_s=30 * 60, allure=ef)]
            weeks.append({"numero": w + 1, "debut": monday.isoformat(), "phase": phase,
                          "volume_km": round(sum(x["distance_km"] for x in sessions), 1), "seances": sessions})
            continue

        n_runs = min(sessions_per_week, len(available))
        n_hard = 0 if n_runs <= 1 else 1 + (n_runs >= 4 and phase == "Spécifique")
        layout = place_days(available, long_day, n_hard, n_runs - 1 - n_hard, tennis)
        long_km = min(max(volume * 0.32, LONG_RUN_MIN_KM[distance] + w), LONG_RUN_CAP_KM[distance])
        if phase == "Affûtage":
            long_km *= 0.7
        quality_km = volume * 0.2  # part de volume réservée à chaque séance de qualité
        easy_km = max(5.0, (volume - long_km - quality_km * n_hard) / max(1, n_runs - 1 - n_hard))
        quality_kinds = (["specifique", "fractionne" if distance in ("5k", "10k") else "tempo"]
                         if phase == "Spécifique" else ["specifique"] if phase == "Affûtage"
                         else ["fractionne" if w % 2 == 0 else "tempo"])

        for offset, kind in sorted(layout.items()):
            day = monday + timedelta(days=offset)
            if kind == "longue":
                sessions.append(session(day, "longue", "Sortie longue", f"{long_km:.0f} km en endurance fondamentale, "
                                        "allure régulière et confortable.", long_km, ef.get("lente_s", ef_mid),
                                        pace_range(ef), ef.get("fc_cible"),
                                        "Développer l'endurance et l'économie de course."))
                sessions[-1]["etapes"] = [step("effort", distance_m=long_km * 1000, allure=ef)]
            elif kind == "ef":
                sessions.append(session(day, "ef", "Footing en endurance fondamentale", f"{easy_km:.0f} km tranquilles : "
                                        "tu dois pouvoir parler en courant.", easy_km, ef_mid,
                                        pace_range(ef), ef.get("fc_cible"),
                                        "Récupérer activement et construire le volume."))
                sessions[-1]["etapes"] = [step("effort", distance_m=easy_km * 1000, allure=ef)]
            else:
                q = quality_kinds.pop(0) if quality_kinds else "tempo"
                if q == "fractionne":
                    reps, rep_m, rest = INTERVALS[distance]
                    reps += w // 2
                    km, minutes = quality_volume(reps * rep_m / 1000, frac.get("rapide_s"), reps - 1,
                                                 float(rest.split()[0]) + (0.5 if "30" in rest else 0), ef_mid)
                    sessions.append(session(day, "fractionne", "Fractionné", f"Échauffement 15 min, puis {reps} × {rep_m} m "
                                            f"à allure fractionné, récupération {rest}. Retour au calme 10 min.", km,
                                            frac.get("rapide_s"), pace_range(frac),
                                            frac.get("fc_cible"), "Améliorer la VMA et la vitesse.", minutes))
                    rest_s = (float(rest.split()[0]) + (0.5 if "30" in rest else 0)) * 60
                    sessions[-1]["etapes"] = [step("echauffement", duree_s=WARMUP_S, allure=ef),
                                              repeat(reps, [step("effort", distance_m=rep_m, allure=frac),
                                                            step("recuperation", duree_s=rest_s)]),
                                              step("retour_au_calme", duree_s=COOLDOWN_S, allure=ef)]
                elif q == "tempo":
                    tempo_min = min(40, 15 + 5 * w)
                    tempo_pace = tempo.get("lente_s") or ef_mid
                    km, minutes = quality_volume(tempo_min * 60 / tempo_pace, tempo_pace, 0, 0, ef_mid)
                    sessions.append(session(day, "tempo", "Tempo", f"Échauffement 15 min, puis {tempo_min} min continues à "
                                            "allure tempo, retour au calme 10 min.", km, tempo_pace, pace_range(tempo),
                                            tempo.get("fc_cible"), "Repousser le seuil : tenir vite plus longtemps.", minutes))
                    sessions[-1]["etapes"] = [step("echauffement", duree_s=WARMUP_S, allure=ef),
                                              step("effort", duree_s=tempo_min * 60, allure=tempo),
                                              step("retour_au_calme", duree_s=COOLDOWN_S, allure=ef)]
                else:
                    reps, rep_m = SPECIFIC[distance]
                    if phase == "Affûtage":
                        reps = max(1, reps - 1)
                    km, minutes = quality_volume(reps * rep_m / 1000, race_pace_s, reps - 1, 2, ef_mid)
                    sessions.append(session(day, "specifique", f"Allure {label}", f"Échauffement 15 min, puis {reps} × "
                                            f"{rep_m / 1000:g} km à allure course, récupération 2 min. Retour au calme 10 min.",
                                            km, race_pace_s, fmt_pace(race_pace_s), None,
                                            "Automatiser l'allure de course.", minutes))
                    sessions[-1]["etapes"] = [step("echauffement", duree_s=WARMUP_S, allure=ef),
                                              repeat(reps, [step("effort", distance_m=rep_m, allure=around(race_pace_s)),
                                                            step("recuperation", duree_s=120)]),
                                              step("retour_au_calme", duree_s=COOLDOWN_S, allure=ef)]
        weeks.append({"numero": w + 1, "debut": monday.isoformat(), "phase": phase,
                      "volume_km": round(sum(x["distance_km"] for x in sessions), 1), "seances": sessions})

    # Les séances déjà passées de la semaine en cours restent visibles, marquées comme passées
    for week in weeks:
        for s in week["seances"]:
            s["passee"] = s["date"] < today.isoformat()

    # Séance du jour adaptée à la forme
    for week in weeks:
        for s in week["seances"]:
            if s["date"] == today.isoformat() and s["type"] in HARD - {"course"} and verdict != "vert":
                original = s["titre"]
                s.update({"type": "ef", "titre": "Footing facile (séance adaptée)",
                          "description": "Ta forme du jour ne permet pas une séance dure : 30 à 40 min très faciles, "
                          f"ou repos complet. La séance prévue ({original.lower()}) peut être décalée de 48 h.",
                          "distance_km": 6.0, "allure": f"{fmt_pace(ef.get('lente_s'))} ou plus lent",
                          "etapes": [step("effort", duree_s=35 * 60, allure={"rapide_s": ef.get("lente_s"),
                                                                              "lente_s": (ef.get("lente_s") or 0) + 30})]})
                notes.append(f"La séance du jour ({original.lower()}) a été allégée à cause de ta forme du jour.")

    if race_date is None:
        notes.append("Aucune date de course : voici deux semaines de développement général.")
    if tennis:
        notes.append("Jours de tennis : " + ", ".join(DAY_NAMES[d].lower() for d in sorted(tennis))
                     + ". Aucune séance dure n'est placée la veille ou le lendemain quand c'est possible.")
    return {"semaines": weeks, "notes": notes}
