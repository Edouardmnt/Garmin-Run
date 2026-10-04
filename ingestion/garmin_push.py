"""Envoi automatique de la séance du jour sur la montre, chaque matin après la synchronisation.

Lancement, depuis la racine du dépôt : python -m ingestion.garmin_push
Calcule le planning de l'objectif actif (séance du jour déjà adaptée à la forme), puis place la séance
dans le calendrier Garmin. Sans séance prévue ce jour-là, il n'envoie rien.
"""

from api.main import DATA_DIR, todays_session
from ingestion.garmin_export import connect
from processing.watch import send_session


def main() -> None:
    session = todays_session()
    if session is None:
        print("Pas de séance de course prévue aujourd'hui : rien à envoyer.")
        return
    print(f"Séance du jour : {session['titre']} ({session['distance_km']} km)")
    result = send_session(connect(), session, DATA_DIR)
    print(f"Montre : {result['statut']} (entraînement {result['workout_id']} du {result['date']})")


if __name__ == "__main__":
    main()
