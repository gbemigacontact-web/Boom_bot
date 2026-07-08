"""
main.py
=======
Point d'entrée principal du bot Boom & Crash v2.
"""

import sys
import os
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import asyncio
import logging
import subprocess
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

ROOT_DIR   = _SRC_DIR.parent
DATA_DIR   = ROOT_DIR / "data"
STATE_FILE = DATA_DIR / "market_states.json"

from deriv_client import fetch_all_market_data
from state_machine import StateManager
from scenario_engine import run_scenario_engine
from telegram_notifier import notify_results, notify_diagnoses


def commit_state(state_file: Path) -> None:
    git_name  = os.environ.get("GIT_USER_NAME",  "Boom-Crash-Bot-v2")
    git_email = os.environ.get("GIT_USER_EMAIL", "bot@boom-crash.local")
    try:
        subprocess.run(["git", "config", "user.name",  git_name],  check=True)
        subprocess.run(["git", "config", "user.email", git_email], check=True)
        result = subprocess.run(
            ["git", "diff", "--quiet", str(state_file)],
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info("Aucun changement d'état — pas de commit.")
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(["git", "add", str(state_file)], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"chore: état bot — {now}"],
            check=True,
        )
        subprocess.run(["git", "push"], check=True)
        logger.info("État commité et pushé sur GitHub.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Commit non bloquant: {e}")


async def run() -> None:
    start_time = time.time()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info(f"{'='*55}")
    logger.info(f"  BOT BOOM & CRASH v2 — {now_str}")
    logger.info(f"  src: {_SRC_DIR}")
    logger.info(f"  data: {DATA_DIR}")
    logger.info(f"{'='*55}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    state_manager = StateManager(state_file=str(STATE_FILE))
    logger.info(state_manager.summary())

    logger.info("Connexion à l'API Deriv...")
    try:
        market_data = await fetch_all_market_data()
    except Exception as e:
        logger.error(f"Erreur récupération données: {e}")
        sys.exit(1)

    if not market_data:
        logger.error("Aucune donnée récupérée — arrêt.")
        sys.exit(1)

    logger.info(f"Données: {list(market_data.keys())}")

    results = run_scenario_engine(market_data, state_manager)

    signals = [r for r in results if r.signal]
    alerts  = [r for r in results if r.alert and not r.signal]
    skipped = [r for r in results if r.skipped]
    logger.info(f"Résultats: {len(signals)} signal(s), {len(alerts)} alerte(s), {len(skipped)} ignoré(s)")

    run_time = time.time() - start_time
    notify_results(results, run_time_seconds=run_time, send_summary=True)

    # Envoyer un résumé des diagnostics toutes les heures (quand la minute est 0)
    if datetime.now(timezone.utc).minute == 0:
        diagnoses = [r.diagnosis for r in results if r.diagnosis]
        if diagnoses:
            notify_diagnoses(diagnoses)

    commit_state(STATE_FILE)

    logger.info(f"Run complet en {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(run())
