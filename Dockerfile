# Image du pipeline Garmin-Run : ingestion + transformations (bronze -> silver -> gold)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNLAB_DATA_DIR=/data

WORKDIR /app

# 1. Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Utilisateur sans privilèges (bonne pratique de sécurité) et dossier de données
RUN useradd --create-home app && mkdir -p /data && chown app /data

# 3. Code de l'application
COPY ingestion/ ingestion/
COPY processing/ processing/
COPY scripts/ scripts/
COPY ml/ ml/
COPY api/ api/

USER app
VOLUME ["/data"]

# Mode par défaut : la démo, sans compte Garmin
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["demo"]
