"""Écritures atomiques : un lecteur voit l'ancien fichier complet ou le nouveau, jamais un fichier à moitié écrit.

La synchronisation réécrit les tables pendant que l'API et le tableau de bord les lisent. Sans cela, une lecture
au mauvais moment tombe sur un parquet incomplet (« Invalid column metadata (corrupt file?) »).
Principe : écrire dans un fichier temporaire du même dossier, puis le renommer d'un coup (os.replace).
"""

import os
import time
from pathlib import Path


def replace(tmp: Path, path: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows : le fichier est ouvert en lecture à cet instant précis
            time.sleep(0.1 * (attempt + 1))
    os.replace(tmp, path)  # dernière tentative : l'erreur remonte si le fichier reste verrouillé


def write_parquet(df, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    replace(tmp, path)


def write_csv(df, path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, **kwargs)
    replace(tmp, path)
