# Démo en ligne de Foulée sur Hugging Face Spaces (SDK Docker) : données synthétiques, aucun compte Garmin.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNLAB_DATA_DIR=/home/user/data

# Hugging Face exécute le conteneur avec l'utilisateur 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY requirements-demo.txt .
RUN pip install -r requirements-demo.txt

COPY --chown=user . .
USER user

# Les données synthétiques sont générées au démarrage (quelques secondes), puis le tableau de bord démarre
EXPOSE 7860
CMD ["python", "scripts/demo.py", "--port", "7860", "--public", "--regenerer"]
