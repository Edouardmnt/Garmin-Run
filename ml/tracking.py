"""Configuration commune du suivi MLflow (stocké dans le dossier de données, jamais publié)."""

import os
from pathlib import Path

# Types autorisés au rechargement des modèles (format sécurisé skops). Ils viennent de nos propres scripts.
TRUSTED_TYPES = [
    "numpy.dtype",
    "ml.models.ResidualModel",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def get_mlflow(data_dir: Path, experiment: str):
    """Renvoie le module mlflow configuré, ou None si MLflow est absent ou désactivé (RUNLAB_MLFLOW=0)."""
    if os.getenv("RUNLAB_MLFLOW", "1") == "0":
        return None
    try:
        import mlflow
        import mlflow.sklearn  # noqa: F401
    except ImportError:
        print("MLflow n'est pas installé : suivi ignoré (pip install -r requirements-dev.txt).")
        return None

    mlflow_dir = data_dir / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(mlflow_dir / "artifacts").resolve().as_uri())
    mlflow.set_experiment(experiment)
    return mlflow
