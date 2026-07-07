"""
main.py
=======
Point d'entrée principal du bot Boom & Crash v2.

Orchestre le cycle complet à chaque run GitHub Actions :
  1. Chargement de l'état FSM persistant
  2. Récupération des données Deriv (WebSocket)
  3. Moteur de scénarios (FSM + analyse technique)
  4. Notifications Telegram (signaux + alertes + résumé)
  5. Sauvegarde de l'état mis à jour
  6. Commit automatique du fichier d'état dans le repo

Le commit automatique permet à chaque run de voir l'état
laissé par le run précédent — c'est la mémoire du bot.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration du logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ── Chemin du fichier d'état ─────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
STATE_FILE = DATA_DIR / "market_states.json"

# Ajouter src/ au path Python pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from deriv_client import fetch_all_market_data
from state_machine import StateManager
from scenario_engine import run_scenario_engine
from telegram_notifier import notify_results


# ─────────────────────────────────────────────────────────────────────────────
# Commit automatique de l'état dans GitHub
# ─────────────────────────────────────────────────────────────────────────────

def commit_state(state_file: Path) -> None:
    """
    Commit et push le fichier d'état market_states.json dans le repo.
    Permet à chaque run GitHub Actions de retrouver l'état du run précédent.

    Nécessite que les variables GIT_USER_NAME et GIT_USER_EMAIL soient
    définies dans les secrets GitHub, ou utilise des valeurs par défaut.
    """
    git_name  = os.environ.get("GIT_USER_NAME",  "Boom-Crash-Bot")
    git_email = os.environ.get("GIT_USER_EMAIL", "bot@boom-crash.local")

    try:
        subprocess.run(["git", "config", "user.name",  git_name],  check=True)
        subprocess.run(["git", "config", "user.email", git_email], check=True)

        # Vérifier s'il y a des changements à committer
        result = subprocess.run(
            ["git", "diff", "--quiet", str(state_file)],
            capture_output=True,
        )

        if result.returncode == 0:
            logger.info("Aucun changement d'état — pas de commit nécessaire.")
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(["git", "add", str(state_file)], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"chore: mise à jour état bot — {now}"],
            check=True,
        )
        subprocess.run(["git", "push"], check=True)
        logger.info("État commité et pushé sur GitHub.")

    except subprocess.CalledProcessError as e:
        # Ne pas faire échouer le run si le commit échoue
        logger.warning(f"Commit d'état échoué (non bloquant): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Cycle principal
# ─────────────────────────────────────────────────────────────────────────────

async def run() -> None:
    start_time = time.time()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info(f"{'='*55}")
    logger.info(f"  BOT BOOM & CRASH v2 — Démarrage — {now_str}")
    logger.info(f"{'='*55}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Étape 1 : Chargement de l'état FSM ───────────────────────────────────
    state_manager = StateManager(state_file=str(STATE_FILE))
    logger.info(state_manager.summary())

    # ── Étape 2 : Récupération des données Deriv ──────────────────────────────
    logger.info("Connexion à l'API Deriv...")
    try:
        market_data = await fetch_all_market_data()
    except Exception as e:
        logger.error(f"Erreur critique lors de la récupération des données: {e}")
        sys.exit(1)

    if not market_data:
        logger.error("Aucune donnée récupérée — arrêt du run.")
        sys.exit(1)

    logger.info(f"Données disponibles: {list(market_data.keys())}")

    # ── Étape 3 : Moteur de scénarios ─────────────────────────────────────────
    logger.info("Lancement du moteur de scénarios...")
    results = run_scenario_engine(market_data, state_manager)

    # Log des résultats
    signals  = [r for r in results if r.signal]
    alerts   = [r for r in results if r.alert and not r.signal]
    skipped  = [r for r in results if r.skipped]

    logger.info(
        f"Run terminé: {len(signals)} signal(s), "
        f"{len(alerts)} alerte(s), "
        f"{len(skipped)} ignoré(s)"
    )

    # ── Étape 4 : Notifications Telegram ──────────────────────────────────────
    run_time = time.time() - start_time
    notify_results(results, run_time_seconds=run_time, send_summary=True)

    # ── Étape 5 : Sauvegarde + commit ─────────────────────────────────────────
    # L'état a déjà été sauvegardé par run_scenario_engine() → commit GitHub
    commit_state(STATE_FILE)

    elapsed = time.time() - start_time
    logger.info(f"{'='*55}")
    logger.info(f"  Run complet en {elapsed:.1f}s")
    logger.info(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run())
