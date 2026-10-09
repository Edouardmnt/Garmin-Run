"""Backtest des prédictions de temps, suivi dans MLflow (expérience « prediction-backtest »).

Lancement, depuis la racine du dépôt : python -m ml.backtest_predictions
Chaque exécution enregistre l'erreur de chaque méthode : on suit ainsi la qualité au fil des versions du code.
"""

import json
import subprocess

from api.main import DATA_DIR, chronic_load, estimation_inputs
from ml.tracking import get_mlflow
from processing.backtest import backtest, by_source, metrics, questionnaire_agreement, recalibration
from processing.feedback import load_feedback

NAMES = {"application": "Application", "sans_correction": "Sans correction", "recalibree": "Recalibrage seul",
         "questionnaires": "Questionnaires seuls",
         "vo2max_montre": "VO2 max de la montre", "relation_fc_vitesse": "Relation FC / vitesse",
         "performances": "Performances seules", "riegel_derniere": "Riegel (dernière perf.)"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "inconnu"


def main() -> None:
    gold, perf, physio = estimation_inputs()
    feedback = load_feedback(DATA_DIR)
    results = backtest(perf, physio, chronic_load(gold), feedback)
    if results.empty:
        print("Pas assez de performances pour un backtest.")
        return
    scores = metrics(results)
    print(f"Backtest sur {len(results)} performances, du {results['date'].min():%d/%m/%Y} au {results['date'].max():%d/%m/%Y}\n")
    print(f"{'Méthode':38} {'n':>3} {'Erreur moy.':>12} {'Biais':>8} {'À ±3 %':>8}")
    for method, s in sorted(scores.items(), key=lambda x: x[1]["mape_pct"]):
        print(f"{NAMES[method]:38} {s['n']:>3} {s['mape_pct']:>10.1f} % {s['biais_pct']:>+6.1f} % {s['part_a_3pct']:>6} %")
    print("\n(biais positif : prédictions trop lentes, pessimistes ; négatif : trop rapides, optimistes)")
    print("\nApplication, par type de performance :")
    for source, s in by_source(results).items():
        print(f"  {source:30} n={s['n']:>2}   erreur {s['mape_pct']:>5.1f} %   biais {s['biais_pct']:>+6.1f} %")
    races = results[results["source"] == "course"]
    if len(races):
        today = recalibration(races["temps_reel_s"], races["sans_correction"])
        print(f"\nCorrection appliquée aujourd'hui aux temps affichés : {today:+.1f} % "
              f"(apprise sur {len(races)} course(s))")

    out = DATA_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "backtest_predictions.csv", index=False)
    summary = {"methodes": scores, "par_source": by_source(results), "questionnaires": questionnaire_agreement(feedback),
               "commit": git_commit()}
    (out / "backtest_predictions.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow = get_mlflow(DATA_DIR, "prediction-backtest")
    if mlflow:
        with mlflow.start_run(run_name=f"backtest-{summary['commit']}"):
            mlflow.log_params({"performances": len(results), "commit": summary["commit"]})
            for method, s in scores.items():
                mlflow.log_metrics({f"{method}_mape_pct": s["mape_pct"], f"{method}_biais_pct": s["biais_pct"],
                                    f"{method}_part_a_3pct": s["part_a_3pct"]})
            mlflow.log_artifact(str(out / "backtest_predictions.csv"))
        print("\nRésultats enregistrés dans MLflow (expérience « prediction-backtest »).")


if __name__ == "__main__":
    main()
