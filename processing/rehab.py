"""Douleurs fréquentes du coureur : exercices de renforcement et de mobilité souvent proposés en kinésithérapie.

Ce n'est PAS un diagnostic. Le coach s'appuie sur cette base fixe et relue, plutôt que sur ce que le modèle
« croit savoir » : les exercices et les signaux d'alerte affichés viennent d'ici, pas de la génération.
Chaque zone indique quand consulter rapidement et quel professionnel voir.

Règle commune (modèle de suivi de la douleur, couramment utilisé en rééducation des tendinopathies) :
une douleur jusqu'à 3/10 pendant l'exercice est acceptable si elle ne s'aggrave pas le lendemain matin.
"""

import re

GENERAL_RULES = [
    "Une gêne jusqu'à 3 sur 10 pendant un exercice est acceptable si elle n'est pas plus forte le lendemain matin. "
    "Au-delà, ou si elle augmente, arrête l'exercice.",
    "Commence par la version la plus facile, 3 à 4 fois par semaine, et progresse sur 2 à 3 semaines.",
    "Ces exercices ne remplacent pas un bilan : si la douleur dure plus de 1 à 2 semaines malgré le repos relatif, "
    "consulte un médecin du sport ou un kinésithérapeute.",
]

RED_FLAGS = [
    "douleur très localisée sur un os, qui fait mal quand tu appuies dessus (risque de fracture de fatigue)",
    "gonflement, chaleur ou rougeur",
    "douleur la nuit ou au repos",
    "boiterie ou impossibilité de prendre appui",
    "douleur qui augmente pendant la course au lieu de s'estomper",
    "craquement ou douleur vive soudaine (claquage, rupture)",
    "fourmillements, perte de force ou de sensibilité",
]

ZONES = {
    "genou_avant": {
        "nom": "Genou, face avant (autour ou sous la rotule)",
        "mots": ("rotule", "devant du genou", "avant du genou", "sous la rotule", "genou"),
        "specialiste": "kinésithérapeute ou médecin du sport",
        "exercices": [
            {"nom": "Chaise contre un mur (isométrique)", "dosage": "5 × 30 à 45 s, genoux à 60° environ",
             "consigne": "Dos plaqué au mur, genoux au-dessus des chevilles. Souvent bien toléré même quand le genou "
                         "est sensible."},
            {"nom": "Pont fessier", "dosage": "3 × 12 à 15", "consigne": "Allongé, pousse dans les talons et monte le "
             "bassin sans cambrer. Progression : sur une jambe."},
            {"nom": "Abduction de hanche couché sur le côté", "dosage": "3 × 15 par côté",
             "consigne": "Jambe du dessus tendue, pointe du pied vers l'avant, monte sans basculer le bassin."},
            {"nom": "Descente de marche contrôlée", "dosage": "3 × 10 par jambe, lentement",
             "consigne": "Genou dans l'axe du pied, sans qu'il rentre vers l'intérieur. Seulement si sans douleur."},
        ],
        "course": "Réduis le volume et le dénivelé (surtout les descentes) ; privilégie le plat et les allures faciles.",
    },
    "genou_exterieur": {
        "nom": "Genou, face extérieure",
        "mots": ("exterieur du genou", "cote du genou", "bandelette", "essuie-glace", "tfl"),
        "specialiste": "kinésithérapeute ou médecin du sport",
        "exercices": [
            {"nom": "Coquillage (clamshell)", "dosage": "3 × 15 par côté",
             "consigne": "Couché sur le côté, genoux fléchis, ouvre le genou du dessus sans rouler le bassin. "
                         "Progression : avec un élastique."},
            {"nom": "Gainage latéral", "dosage": "3 × 20 à 40 s par côté", "consigne": "Corps aligné, sur l'avant-bras."},
            {"nom": "Pont fessier sur une jambe", "dosage": "3 × 10 par jambe", "consigne": "Bassin bien horizontal."},
            {"nom": "Rouleau de massage sur la cuisse (côté et avant)", "dosage": "1 à 2 min par zone",
             "consigne": "Sur les muscles de la cuisse et de la hanche, pas directement sur le genou."},
        ],
        "course": "Évite les descentes et les pistes en dévers ; raccourcis un peu ta foulée.",
    },
    "achille": {
        "nom": "Tendon d'Achille",
        "mots": ("achille", "tendon", "arriere du talon", "derriere le talon"),
        "specialiste": "kinésithérapeute ou médecin du sport",
        "exercices": [
            {"nom": "Montées sur pointes lentes, jambe tendue", "dosage": "3 × 15, 3 s de montée et 3 s de descente",
             "consigne": "Sur les deux pieds au début, puis sur un seul. Gêne acceptable jusqu'à 3/10."},
            {"nom": "Montées sur pointes, genou fléchi", "dosage": "3 × 15",
             "consigne": "Même mouvement, genou légèrement plié : travaille le soléaire."},
            {"nom": "Isométrique sur pointe", "dosage": "5 × 30 à 45 s",
             "consigne": "Tiens la position haute sur pointes : souvent calme la douleur les jours sensibles."},
        ],
        "course": "Évite côtes, fractionné et chaussures très basses le temps que ça se calme ; ne t'étire pas "
                  "fort sur un tendon douloureux.",
    },
    "mollet": {
        "nom": "Mollet",
        "mots": ("mollet", "soleaire", "gastro"),
        "specialiste": "médecin du sport en cas de douleur vive soudaine, sinon kinésithérapeute",
        "exercices": [
            {"nom": "Montées sur pointes progressives", "dosage": "3 × 12 à 15",
             "consigne": "D'abord sur deux pieds, puis un seul, seulement si c'est indolore."},
            {"nom": "Montées sur pointes genou fléchi", "dosage": "3 × 12", "consigne": "Travaille le soléaire."},
            {"nom": "Mobilité de cheville genou au mur", "dosage": "2 × 10 par jambe",
             "consigne": "Talon au sol, avance le genou vers le mur sans forcer."},
        ],
        "course": "Après une douleur vive soudaine (claquage possible), pas de course avant avis médical.",
    },
    "tibia": {
        "nom": "Tibia (face interne ou avant de la jambe)",
        "mots": ("tibia", "periostite", "devant de la jambe", "shin"),
        "specialiste": "médecin du sport (une imagerie peut être nécessaire)",
        "exercices": [
            {"nom": "Montées sur pointes", "dosage": "3 × 15", "consigne": "Renforce mollets et soléaire, qui amortissent."},
            {"nom": "Marche sur les talons", "dosage": "3 × 30 s", "consigne": "Pointes de pieds relevées."},
            {"nom": "Équilibre sur une jambe", "dosage": "3 × 30 s par jambe", "consigne": "Progression : yeux fermés."},
        ],
        "course": "Baisse nettement le volume, évite le bitume et la vitesse. Une douleur précise sur l'os au toucher "
                  "doit faire penser à une fracture de fatigue : consultation sans attendre.",
    },
    "pied": {
        "nom": "Voûte plantaire et dessous du talon",
        "mots": ("voute", "plantaire", "dessous du talon", "aponevrose", "talon", "pied"),
        "specialiste": "médecin du sport, kinésithérapeute ou podologue",
        "exercices": [
            {"nom": "Montées sur pointes avec une serviette roulée sous les orteils", "dosage": "3 × 12, lentement",
             "consigne": "Orteils relevés sur la serviette : met la voûte en tension de façon progressive."},
            {"nom": "Rouler une balle sous le pied", "dosage": "2 min", "consigne": "Pression modérée, pas douloureuse."},
            {"nom": "Pied court (contracter la voûte)", "dosage": "3 × 10, 5 s", "consigne": "Rapproche la base du gros "
             "orteil du talon sans recroqueviller les orteils."},
            {"nom": "Étirement doux du mollet", "dosage": "3 × 30 s", "consigne": "Contre un mur, talon au sol."},
        ],
        "course": "La douleur des premiers pas du matin est typique : réduis le volume et vérifie l'usure de tes chaussures.",
    },
    "hanche_fessier": {
        "nom": "Hanche et fessier",
        "mots": ("hanche", "fessier", "fesse", "pyramidal", "aine"),
        "specialiste": "kinésithérapeute ou médecin du sport (médecin rapidement pour une douleur de l'aine)",
        "exercices": [
            {"nom": "Pont fessier", "dosage": "3 × 15", "consigne": "Progression : sur une jambe."},
            {"nom": "Coquillage (clamshell)", "dosage": "3 × 15 par côté", "consigne": "Avec élastique quand c'est facile."},
            {"nom": "Gainage latéral", "dosage": "3 × 30 s par côté", "consigne": "Hanches hautes, corps aligné."},
        ],
        "course": "Une douleur de l'aine qui persiste chez un coureur doit être vue par un médecin (fracture de fatigue "
                  "du col du fémur, rare mais sérieuse).",
    },
    "ischio": {
        "nom": "Arrière de la cuisse (ischio-jambiers)",
        "mots": ("ischio", "arriere de la cuisse", "derriere la cuisse"),
        "specialiste": "kinésithérapeute ; médecin du sport après une douleur vive soudaine",
        "exercices": [
            {"nom": "Pont fessier, talons éloignés", "dosage": "3 × 12", "consigne": "Talons loin des fesses : sollicite "
             "les ischio-jambiers."},
            {"nom": "Isométrique talon contre le sol", "dosage": "5 × 30 s",
             "consigne": "Allongé, genou fléchi, appuie le talon dans le sol sans bouger."},
            {"nom": "Soulevé de terre sur une jambe, sans charge", "dosage": "3 × 8 par jambe",
             "consigne": "Dos droit, mouvement lent. Seulement s'il est indolore."},
        ],
        "course": "Pas de sprint ni de fractionné court tant que la douleur est là.",
    },
    "dos": {
        "nom": "Bas du dos",
        "mots": ("dos", "lombaire", "lombaires", "reins"),
        "specialiste": "médecin traitant ou kinésithérapeute",
        "exercices": [
            {"nom": "Chat-vache", "dosage": "2 × 10", "consigne": "À quatre pattes, enroule puis creuse doucement le dos."},
            {"nom": "Bird-dog", "dosage": "3 × 8 par côté", "consigne": "Bras et jambe opposés tendus, sans cambrer."},
            {"nom": "Gainage sur les avant-bras", "dosage": "3 × 20 à 40 s", "consigne": "Ventre rentré, dos plat."},
        ],
        "course": "Une douleur qui descend dans la jambe, avec fourmillements ou perte de force, doit être vue par un "
                  "médecin.",
    },
}

# Débuts de mots : « douloureux », « tendinite », « blessé »... (« normal » ne compte pas : début de mot exigé)
PAIN_WORDS = ("mal ", "douleur", "douloureu", "blessure", "blesse", "gene ", "tendin", "contracture", "entorse",
              "claquage", "periostite", "courbature")


def _normalize(text: str) -> str:
    table = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
    text = " " + text.lower().translate(table).replace("’", "'") + " "
    return re.sub(r"(course|courir|marche|aller) a pied", " ", text)  # « course à pied » n'est pas une douleur au pied


def _has(text: str, word: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text) is not None


def mentions_pain(question: str) -> bool:
    """La question parle-t-elle d'une douleur ou d'une partie du corps ?"""
    text = _normalize(question)
    return any(re.search(r"(?<![a-z])" + re.escape(w), text) for w in PAIN_WORDS) or bool(detect_zones(question))


def detect_zones(question: str) -> list[str]:
    """Zones évoquées, de la plus précise à la plus générale (« extérieur du genou » avant « genou »)."""
    text = _normalize(question)
    found = [key for key, zone in ZONES.items() if any(_has(text, m) for m in zone["mots"])]
    if "genou_exterieur" in found and "genou_avant" in found and not any(
            m in text for m in ("rotule", "avant du genou", "devant du genou")):
        found.remove("genou_avant")
    if "achille" in found and "pied" in found and "dessous" not in text and "voute" not in text:
        found.remove("pied")
    return found[:2]


def pain_data(question: str) -> dict:
    """Ce que le coach reçoit (et ce que l'application affiche) quand on lui parle d'une douleur."""
    zones = detect_zones(question)
    return {
        "zones": [{"id": z, **{k: v for k, v in ZONES[z].items() if k != "mots"}} for z in zones],
        "zone_inconnue": not zones,
        "zones_disponibles": None if zones else [z["nom"] for z in ZONES.values()],
        "signaux_alerte": RED_FLAGS,
        "regles": GENERAL_RULES,
    }
