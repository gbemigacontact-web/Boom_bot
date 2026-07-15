"""
main.py
=======
Point d'entrée principal du bot Boom & Crash v2 — Architecture hybride IA.

Orchestre le cycle complet à chaque run GitHub Actions :
  1. Chargement de l'état persistant (FSM fallback + mémoire IA)
  2. Récupération des données Deriv (WebSocket, vue large 500 bougies H4/H1)
  3. Orchestrateur : pour chaque instrument, IA en priorité, bascule
     automatique sur le moteur Python si Gemini est indisponible
  4. Notifications Telegram (signaux + alertes notables + résumé)
  5. Sauvegarde de l'état mis à jour (déjà faite par l'orchestrateur)
  6. Commit automatique du fichier d'état dans le repo

Le commit automatique permet à chaque run de voir l'état laissé par le
run précédent — c'est la mémoire du bot (FSM fallback + continuité du
jugement Gemini).
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
from scenario_orchestrator import run_orchestrator
from ai_analyzer import is_quota_exhausted
from telegram_notifier import notify_results


# ─────────────────────────────────────────────────────────────────────────────
# Vérifications de démarrage
# ─────────────────────────────────────────────────────────────────────────────

def check_environment() -> None:
    """
    Vérifie la présence des variables d'environnement critiques et logue
    des avertissements clairs plutôt que de laisser le bot échouer plus
    tard avec une erreur cryptique. Le bot peut fonctionner sans
    GEMINI_API_KEY (bascule intégrale sur le fallback Python), mais
    l'opérateur doit le savoir explicitement.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning(
            "GEMINI_API_KEY absent — le bot fonctionnera en mode fallback "
            "Python intégral pour ce run (aucune analyse IA)."
        )
    if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
        logger.warning(
            "TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID absent — les notifications "
            "ne pourront pas être envoyées."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Commit automatique de l'état dans GitHub
# ─────────────────────────────────────────────────────────────────────────────

def commit_state(state_file: Path) -> None:
    """
    Commit et push le fichier d'état market_states.json dans le repo.
    Permet à chaque run GitHub Actions de retrouver l'état du run précédent
    (FSM fallback + mémoire IA).

    Nécessite que les variables GIT_USER_NAME et GIT_USER_EMAIL soient
    définies dans les secrets GitHub, ou utilise des valeurs par défaut.
    """
    git_name  = os.environ.get("GIT_USER_NAME",  "Boom-Crash-Bot")
    git_email = os.environ.get("GIT_USER_EMAIL", "bot@boom-crash.local")

    try:
        subprocess.run(["git", "config", "user.name",  git_name],  check=True)
        subprocess.run(["git", "config", "user.email", git_email], check=True)

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
    logger.info(f"  BOT BOOM & CRASH v2 — IA + Fallback — {now_str}")
    logger.info(f"{'='*55}")

    check_environment()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Étape 1 : Chargement de l'état persistant ─────────────────────────────
    state_manager = StateManager(state_file=str(STATE_FILE))
    logger.info(state_manager.summary())

    # ── Étape 2 : Récupération des données Deriv ──────────────────────────────
    logger.info("Connexion à l'API Deriv (vue large H4/H1 500 bougies)...")
    try:
        market_data = await fetch_all_market_data()
    except Exception as e:
        logger.error(f"Erreur critique lors de la récupération des données: {e}")
        sys.exit(1)

    if not market_data:
        logger.error("Aucune donnée récupérée — arrêt du run.")
        sys.exit(1)

    logger.info(f"Données disponibles: {list(market_data.keys())}")

    # ── Étape 3 : Orchestrateur (IA en priorité, fallback automatique) ────────
    logger.info("Lancement de l'orchestrateur...")
    results = run_orchestrator(market_data, state_manager)

    signals        = [r for r in results if r.signal]
    closed         = [r for r in results if r.position_closed]
    skipped        = [r for r in results if r.skipped]
    analyzed_by_ia = [r for r in results if r.source == "IA"]
    fallback_used  = [r for r in results if r.source == "FALLBACK"]
    quota_exhausted = is_quota_exhausted()

    logger.info(
        f"Run terminé: {len(signals)} signal(s), "
        f"{len(closed)} clôture(s), "
        f"{len(skipped)} ignoré(s) | "
        f"IA: {len(analyzed_by_ia)}, Fallback: {len(fallback_used)}"
        + (" | ⚠️ quota IA épuisé" if quota_exhausted else "")
    )

    # ── Étape 4 : Notifications Telegram ──────────────────────────────────────
    run_time = time.time() - start_time
    notify_results(
        results,
        run_time_seconds=run_time,
        send_summary=True,
        quota_exhausted=quota_exhausted,
    )

    # ── Étape 5 : Commit de l'état ────────────────────────────────────────────
    # L'état (FSM fallback + mémoire IA) a déjà été sauvegardé par
    # run_orchestrator() → il ne reste plus qu'à le committer sur GitHub.
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
