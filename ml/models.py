"""Modèles maison, dans un module importable : MLflow doit pouvoir les recharger ailleurs."""

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone


class ResidualModel(RegressorMixin, BaseEstimator):
    """Prédit l'écart à la moyenne des 7 derniers jours, puis ajoute cette moyenne.

    Le modèle ne part pas de zéro : il apprend seulement une correction de la référence naïve.
    Sans signal, il prédit une correction proche de 0 et retombe sur la référence.
    """

    def __init__(self, estimator=None, baseline_col: str = "hrv_7d_mean", fallback_col: str = "hrv_last_night"):
        self.estimator = estimator
        self.baseline_col = baseline_col
        self.fallback_col = fallback_col

    def _baseline(self, X: pd.DataFrame) -> pd.Series:
        return X[self.baseline_col].fillna(X[self.fallback_col])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.estimator_ = clone(self.estimator).fit(X, y - self._baseline(X))
        return self

    def predict(self, X: pd.DataFrame):
        return self._baseline(X).to_numpy() + self.estimator_.predict(X)
