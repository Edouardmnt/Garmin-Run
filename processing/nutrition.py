"""Conseils de nutrition et d'hydratation pour une course, adaptés à la durée prévue et à la température.

Repères issus des recommandations de nutrition sportive (prise de position conjointe ACSM / Academy of
Nutrition and Dietetics / Dietitians of Canada, 2016) et de la littérature sur l'apport glucidique à l'effort :
- moins de 1 h  : pas de glucides nécessaires ; un rinçage de bouche avec une boisson sucrée peut aider ;
- 1 h à 2 h 30  : 30 à 60 g de glucides par heure ;
- plus de 2 h 30 : jusqu'à 60 à 90 g par heure, avec un mélange glucose + fructose et un intestin entraîné ;
- boisson : à adapter à la soif, à la chaleur et à la sudation, souvent 400 à 800 ml par heure ;
- sodium : utile pour les efforts longs ou chauds, souvent 300 à 600 mg par heure.

Ce sont des repères généraux : chaque stratégie doit être testée à l'entraînement.
"""

GEL_CARBS_G = 25  # glucides d'un gel classique


def carbs_per_hour(duration_min: float) -> tuple[int, int]:
    if duration_min < 60:
        return 0, 0
    if duration_min <= 150:
        return 30, 60
    return 60, 90


def fluids_per_hour(duration_min: float, temperature_c: float) -> tuple[int, int]:
    if duration_min < 45:
        return 0, 0
    if temperature_c >= 25:
        return 600, 900
    if temperature_c >= 18:
        return 500, 750
    if temperature_c >= 10:
        return 400, 600
    return 300, 500


def sodium_needed(duration_min: float, temperature_c: float) -> bool:
    return duration_min > 120 or (temperature_c >= 25 and duration_min > 60)


def race_nutrition(distance_label: str, duration_min: float, pace_s_km: float, temperature_c: float) -> dict:
    carbs, fluids = carbs_per_hour(duration_min), fluids_per_hour(duration_min, temperature_c)
    long_race = duration_min > 90

    before = ["La veille : un dîner riche en glucides (pâtes, riz, pommes de terre) et rien de nouveau ou d'épicé."]
    if duration_min > 150:
        before.insert(0, "Les 2 à 3 jours avant : augmente la part de glucides à chaque repas (charge glucidique).")
    before += [
        "Le matin : petit-déjeuner riche en glucides, pauvre en fibres et en graisses, 2 à 3 h avant le départ "
        "(pain, confiture, banane, riz), avec 300 à 500 ml d'eau.",
        "Dans l'heure avant : de petites gorgées d'eau." + (" Un gel ou une compote 15 min avant le départ est possible."
                                                            if long_race else ""),
    ]

    during, timeline = [], []
    if carbs == (0, 0):
        during.append(f"Pour un {distance_label} d'environ {duration_min:.0f} min, aucun apport de glucides n'est nécessaire.")
    else:
        gels = (round(carbs[0] / GEL_CARBS_G, 1), round(carbs[1] / GEL_CARBS_G, 1))
        low, high = (f"{g:g}".replace(".", ",") for g in gels)
        during.append(f"Glucides : {carbs[0]} à {carbs[1]} g par heure, soit environ {low} à {high} gels "
                      f"de {GEL_CARBS_G} g par heure, en commençant vers la 30e-40e minute.")
        if carbs[1] >= 90:
            during.append("Au-delà de 60 g par heure, privilégie des produits mélangeant glucose et fructose, "
                          "mieux tolérés par l'intestin.")
    if fluids == (0, 0):
        during.append("Boire pendant la course n'est généralement pas nécessaire sur une durée aussi courte.")
    else:
        during.append(f"Boisson : environ {fluids[0]} à {fluids[1]} ml par heure à {temperature_c:.0f} °C, "
                      "par petites gorgées toutes les 15 à 20 minutes, en te fiant aussi à ta soif.")
    if sodium_needed(duration_min, temperature_c):
        during.append("Sodium : 300 à 600 mg par heure (boisson d'effort ou pastilles de sel), "
                      "pour compenser les pertes de sueur.")
    if temperature_c >= 25:
        during.append("Chaleur : asperge-toi aux ravitaillements et pars un peu moins vite que ton allure cible.")

    # Repères au fil de la course, en minutes et en kilomètres
    if fluids != (0, 0):
        for minute in range(20, int(duration_min), 20):
            timeline.append({"minute": minute, "km": round(minute * 60 / pace_s_km, 1), "action": "Quelques gorgées d'eau"})
    if carbs != (0, 0):
        interval = 60 / (((carbs[0] + carbs[1]) / 2) / GEL_CARBS_G)
        minute = 35.0
        while minute < duration_min - 10:
            timeline.append({"minute": round(minute), "km": round(minute * 60 / pace_s_km, 1),
                             "action": f"Un gel ({GEL_CARBS_G} g de glucides) avec de l'eau"})
            minute += interval
    timeline.sort(key=lambda t: t["minute"])

    after = ["Dans les 30 à 60 min : un en-cas avec glucides et protéines (par exemple un laitage, une banane, du pain).",
             "Réhydrate-toi progressivement pendant les heures suivantes, sans tout boire d'un coup."]
    return {
        "duree_prevue_min": round(duration_min),
        "temperature_c": temperature_c,
        "glucides_g_par_heure": list(carbs),
        "boisson_ml_par_heure": list(fluids),
        "sodium": sodium_needed(duration_min, temperature_c),
        "avant": before,
        "pendant": during,
        "reperes": timeline,
        "apres": after,
        "avertissement": "Repères généraux de nutrition sportive : teste toujours ta stratégie à l'entraînement "
                         "et demande l'avis d'un professionnel de santé en cas de pathologie ou de régime particulier.",
    }
