"""Publie la démo sur un Space Hugging Face (SDK Docker).

    $env:HF_TOKEN = "hf_..."          # jeton « write » : https://huggingface.co/settings/tokens
    python deploy/huggingface/push_space.py --space Edouardmnt/foulee

Copie uniquement le code utile à la démo (jamais data/), place le Dockerfile et la fiche du Space à la racine,
puis envoie le tout. Le workflow GitHub .github/workflows/space.yml fait la même chose à chaque push sur main.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INCLUDE = ["api", "dashboard", "ingestion", "ml", "processing", "scripts", ".streamlit", "requirements-demo.txt"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def stage(target: Path) -> None:
    for name in INCLUDE:
        src = ROOT / name
        if not src.exists():  # ex. .streamlit/ absent d'un clone : la démo garde le thème par défaut
            continue
        if src.is_dir():
            shutil.copytree(src, target / name, ignore=IGNORE)
        else:
            shutil.copy2(src, target / name)
    shutil.copy2(HERE / "Dockerfile", target / "Dockerfile")
    shutil.copy2(HERE / "README.md", target / "README.md")  # fiche du Space (en-tête YAML)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", required=True, help="identifiant du Space, ex. Edouardmnt/foulee")
    parser.add_argument("--dry-run", action="store_true", help="prépare le dossier sans rien envoyer")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        stage(target)
        files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
        print(f"{len(files)} fichiers prêts pour {args.space}")
        if args.dry_run:
            return
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(target), repo_id=args.space, repo_type="space",
                          commit_message="Mise à jour de la démo depuis GitHub")
        print(f"Démo publiée : https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
