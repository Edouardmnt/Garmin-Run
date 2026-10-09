"""Évaluation du coach IA sur un jeu de questions types, suivie dans MLflow (expérience « coach-evaluation »).

Lancement, depuis la racine du dépôt (Ollama lancé) : python -m ml.eval_coach
Options : --questions allure_ef,douleur (sous-ensemble)   --modele qwen2.5:7b (comparer un autre modèle)

Pour chaque question : les données consultées (routage), les faits attendus retrouvés dans la réponse,
les chiffres cités introuvables dans les données (hallucinations possibles), les règles de sécurité
(pas de diagnostic, rien n'est modifié) et la durée de réponse.
"""

import argparse
import json
import time

from api.main import DATA_DIR, coach_context, coach_fetchers, resolve_proposal
from ml.backtest_predictions import git_commit
from ml.tracking import get_mlflow
from processing.coach import OllamaLLM, answer_direct, gather_topic_data, get_llm
from processing.coach_eval import EVAL_SET, score_answer, summarize


def evaluate(llm, items: list[dict]) -> tuple[list[dict], list[dict]]:
    context, fetchers = coach_context(), coach_fetchers()
    scores, answers = [], []
    for item in items:
        topics, extra = gather_topic_data(item["question"], fetchers)
        start = time.time()
        answer = answer_direct(llm, item["question"], context, fetchers)
        raw = answer.get("proposition_brute")
        final, _ = resolve_proposal(raw, item["question"]) if item.get("proposition_attendue") else (None, None)
        score = score_answer(item, answer["reponse"], context, extra, topics, final)
        if item.get("proposition_attendue"):  # la part du modèle seul, sans le secours par mots-clés
            score["proposition_du_modele"] = bool(final and final.get("origine") == "coach")
            score["proposition_brute"] = raw
        score["duree_s"] = round(time.time() - start, 1)
        scores.append(score)
        answers.append({"id": item["id"], "question": item["question"], "reponse": answer["reponse"], **score})
        good = score["securite_ok"] and not score["chiffres_non_verifies"] and score["proposition_ok"] is not False
        status = "ok" if good else "!!"
        print(f"  {status} {item['id']:18} faits {score['faits_trouves']}/{score['faits_attendus']}   "
              f"non vérifiés {len(score['chiffres_non_verifies'])}   {score['duree_s']:>5.1f} s")
    return scores, answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", help="identifiants séparés par des virgules (toutes par défaut)")
    parser.add_argument("--modele", help="modèle Ollama à évaluer (celui de la configuration par défaut)")
    args = parser.parse_args()

    items = EVAL_SET
    if args.questions:
        wanted = set(args.questions.split(","))
        items = [q for q in EVAL_SET if q["id"] in wanted]
    llm = OllamaLLM(model=args.modele) if args.modele else get_llm()
    print(f"Évaluation du coach ({llm.model}) sur {len(items)} questions :\n")
    scores, answers = evaluate(llm, items)
    by_model = [s["proposition_du_modele"] for s in scores if "proposition_du_modele" in s]
    summary = summarize(scores) | {"propositions_modele_pct": round(sum(by_model) / len(by_model) * 100)
                                   if by_model else None, "modele": llm.model, "commit": git_commit(),
                                   "duree_moyenne_s": round(sum(s["duree_s"] for s in scores) / len(scores), 1)}

    print(f"\nRoutage des données       : {summary['routage_pct']} %")
    print(f"Faits attendus cités      : {summary['rappel_faits_pct']} %")
    print(f"Réponses sans chiffre non vérifiable : {summary['reponses_sans_chiffre_invente_pct']} %")
    print(f"Règles de sécurité        : {summary['securite_pct']} %")
    print(f"Modifications demandées bien proposées : {summary['propositions_pct']} %"
          f" (dont écrites par le modèle lui-même : {summary.get('propositions_modele_pct')} %)")
    print(f"Durée moyenne par réponse : {summary['duree_moyenne_s']} s")
    for a in answers:
        problems = []
        if a["chiffres_non_verifies"]:
            problems.append("non vérifiés : " + ", ".join(a["chiffres_non_verifies"]))
        if not a["securite_ok"]:
            problems.append("règle de sécurité non respectée")
        if a["faits_trouves"] < a["faits_attendus"]:
            problems.append(f"faits attendus : {', '.join(a.get('faits_textes', []))}")
        if a.get("proposition_ok") is False:
            problems.append(f"proposition attendue non obtenue (écrite par le modèle : {a.get('proposition_brute')})")
        elif a.get("proposition_du_modele") is False:
            problems.append("proposition obtenue grâce au secours par mots-clés, pas par le modèle")
        if problems:
            print(f"\n[{a['id']}] {a['question']}\n  " + "\n  ".join(problems) + f"\n  {a['reponse'][:400]}")

    out = DATA_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "coach_evaluation.json"
    path.write_text(json.dumps({"resume": summary, "reponses": answers}, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    mlflow = get_mlflow(DATA_DIR, "coach-evaluation")
    if mlflow:
        with mlflow.start_run(run_name=f"coach-{llm.model}-{summary['commit']}"):
            mlflow.log_params({"modele": llm.model, "questions": len(items), "commit": summary["commit"]})
            mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None})
            mlflow.log_artifact(str(path))
        print("\nRésultats enregistrés dans MLflow (expérience « coach-evaluation »).")


if __name__ == "__main__":
    main()
