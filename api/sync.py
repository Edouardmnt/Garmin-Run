"""Synchronisation à la demande : lancée par l'API, en arrière-plan, avec sa progression.

- Les étapes sont celles du pipeline « matin » : Garmin, silver, gold, étiquetage, séance du jour sur la montre.
- Un verrou sur le volume garantit une seule synchronisation à la fois, même avec plusieurs onglets ouverts.
- La date de dernière synchronisation est celle de la table gold, que le CronJob et l'API mettent à jour tous deux.

RUNLAB_SYNC_STEPS=demo remplace l'appel à Garmin par des données synthétiques (tests, démonstration).
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Le verrou est « rafraîchi » toutes les HEARTBEAT_S secondes tant que la synchronisation tourne vraiment.
# Si le Pod est redémarré en pleine synchronisation (mise à jour de l'image, rollout), plus personne ne le
# rafraîchit : au bout de STALE_LOCK_S, il est considéré comme abandonné et une nouvelle synchro peut partir.
HEARTBEAT_S = 5
STALE_LOCK_S = 60
STEP_TIMEOUT_S = 10 * 60  # une étape bloquée (Garmin qui ne répond plus) est interrompue


def steps() -> list[tuple[str, list[str]]]:
    """Étapes, avec le libellé affiché dans la barre de progression."""
    first = (("Génération de données de démonstration", ["scripts/generate_sample_data.py"])
             if os.getenv("RUNLAB_SYNC_STEPS") == "demo"
             else ("Récupération de tes données Garmin", ["ingestion/garmin_export.py"]))
    plan = [first,
            ("Préparation des données", ["processing/build_silver.py"]),
            ("Calcul de tes indicateurs", ["processing/build_gold.py"]),
            ("Mise à jour de tes séances", ["scripts/make_run_labels.py"])]
    if os.getenv("RUNLAB_SYNC_STEPS") != "demo":
        plan.append(("Séance du jour sur ta montre", ["-m", "ingestion.garmin_push"]))
    return plan


class SyncManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dir = data_dir / "sync"
        self.lock = self.dir / "verrou"
        self.status_file = self.dir / "statut.json"
        self._thread: threading.Thread | None = None
        self._started = time.time()

    # --- État -------------------------------------------------------------------------------------
    def last_sync(self) -> datetime | None:
        gold = self.data_dir / "gold" / "daily_features.parquet"
        return datetime.fromtimestamp(gold.stat().st_mtime) if gold.exists() else None

    def running(self) -> bool:
        return self.lock.exists() and time.time() - self.lock.stat().st_mtime < STALE_LOCK_S

    def status(self) -> dict:
        saved = self._read_status()
        last = self.last_sync()
        return {
            "en_cours": self.running(),
            "etape": saved.get("etape") if self.running() else None,
            "progression": saved.get("progression", 0.0) if self.running() else 1.0,
            "depuis_s": self._elapsed_s(saved) if self.running() else None,
            "erreur": saved.get("erreur"),
            "derniere_synchro": None if last is None else last.isoformat(timespec="seconds"),
            "age_min": None if last is None else round((datetime.now() - last).total_seconds() / 60),
        }

    @staticmethod
    def _elapsed_s(saved: dict) -> int | None:
        try:
            return max(0, round(time.time() - float(saved["debut"])))
        except (KeyError, TypeError, ValueError):
            return None

    def _read_status(self) -> dict:
        """Lecture tolérante : un fichier illisible donne un état vide, jamais une erreur."""
        try:
            return json.loads(self.status_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            return {}

    def _save(self, **fields) -> None:
        """Écriture atomique : fichier temporaire, puis renommage instantané.

        write_text vide le fichier avant d'écrire : une lecture à cet instant trouverait un fichier vide.
        Avec os.replace, un lecteur voit l'ancien contenu complet ou le nouveau, jamais un état intermédiaire.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.status_file.with_name(f"statut.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, self.status_file)
                return
            except PermissionError:  # Windows : le fichier est ouvert en lecture à cet instant précis
                time.sleep(0.05 * (attempt + 1))
        tmp.unlink(missing_ok=True)

    # --- Lancement --------------------------------------------------------------------------------
    def start(self, older_than_min: int | None = None) -> dict:
        """Lance une synchronisation si aucune n'est en cours et si les données sont assez anciennes."""
        current = self.status()
        if current["en_cours"]:
            return {**current, "lancee": False, "raison": "deja_en_cours"}
        if older_than_min is not None and current["age_min"] is not None and current["age_min"] < older_than_min:
            return {**current, "lancee": False, "raison": "donnees_recentes"}
        self.dir.mkdir(parents=True, exist_ok=True)
        try:  # création exclusive : deux demandes simultanées ne lancent qu'une synchronisation
            with open(self.lock, "x", encoding="utf-8") as f:
                f.write(datetime.now().isoformat())
        except FileExistsError:
            if self.running():
                return {**self.status(), "lancee": False, "raison": "deja_en_cours"}
            self.lock.write_text(datetime.now().isoformat(), encoding="utf-8")  # verrou périmé : on le reprend
        plan = steps()
        self._started = time.time()
        self._save(etape=plan[0][0], progression=0.0, erreur=None, debut=self._started)
        self._thread = threading.Thread(target=self._run, args=(plan,), daemon=True)
        self._thread.start()
        return {**self.status(), "lancee": True}

    def _run(self, plan: list[tuple[str, list[str]]]) -> None:
        env = {**os.environ, "RUNLAB_DATA_DIR": str(self.data_dir), "PYTHONUTF8": "1"}
        error = None
        try:
            for i, (label, args) in enumerate(plan):
                self._save(etape=label, progression=round(i / len(plan), 2), erreur=None, debut=self._started)
                code, output = self._run_step(args, env)
                if code is None:
                    error = f"{label} : interrompue après {STEP_TIMEOUT_S // 60} min sans réponse."
                    break
                if code != 0:
                    tail = output.strip().splitlines()[-1:] or ["erreur inconnue"]
                    error = f"{label} : {tail[0][:200]}"
                    break
        finally:
            self._save(etape=None, progression=1.0, erreur=error)
            self.lock.unlink(missing_ok=True)

    def _run_step(self, args: list[str], env: dict) -> tuple[int | None, str]:
        """Lance une étape en rafraîchissant le verrou pendant qu'elle tourne ; None si elle a dépassé le délai."""
        log = self.dir / "etape.log"  # sortie de l'étape en cours : utile pour comprendre un blocage
        with open(log, "w", encoding="utf-8", errors="replace") as out:
            proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True)
            start = time.time()
            while proc.poll() is None:
                self.lock.touch()
                if time.time() - start > STEP_TIMEOUT_S:
                    proc.kill()
                    proc.wait()
                    return None, ""
                try:
                    proc.wait(timeout=HEARTBEAT_S)
                except subprocess.TimeoutExpired:
                    pass
        return proc.returncode, log.read_text(encoding="utf-8", errors="replace")

    def wait(self, timeout_s: float = 120) -> None:
        """Attend la fin de la synchronisation lancée par ce processus (utile aux tests)."""
        if self._thread:
            self._thread.join(timeout_s)
