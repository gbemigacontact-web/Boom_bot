"""
state_machine.py
================
Machine à états finis (FSM) pour le bot Boom & Crash v2.

Chaque instrument (Boom 500, Crash 1000, etc.) a son propre état
persistant dans un fichier JSON. Le bot lit l'état au démarrage de
chaque run, avance dans la séquence selon ce qu'il détecte, et
écrit le nouvel état avant de terminer.

ÉTATS POSSIBLES
───────────────
NEUTRE
  → Pas de BOS récent. Le bot scanne H4/H1 en attente d'un BOS.

BOS_DETECTE
  → Un BOS vient d'être confirmé. Les zones sont cartographiées :
    OB principal, FVG, swing point de liquidité (invalidation).
    Le bot attend que le prix entre dans l'une des zones.

PULLBACK_ZONE_FVG
  → Le prix est entré dans le FVG (pullback court). Surveillance M15
    activée pour une bougie de rejet ou un mini-CHoCH.

PULLBACK_ZONE_OB
  → Le prix est entré dans l'OB (pullback profond). Même surveillance.

CONFIRMATION_ATTENDUE
  → Le prix est dans une zone valide. On attend la bougie de
    confirmation M15 (mèche > 60% range + volume > moy. 20 bougies).

ENTREE_ACTIVE
  → Signal envoyé. On attend que le TP1/TP2/TP3 ou le SL soit atteint.

RISQUE_CHOCH
  → Le prix a traversé l'OB sans confirmation et approche du swing
    point d'invalidation. Mode alerte — le retournement est possible.

RETOURNEMENT_SURVEILLANCE
  → Le CHoCH est confirmé (prix a clôturé sous/sur le swing point).
    Le bot suit la séquence du Scénario 2 :
    OB inverse → fausse cassure → BOS inverse → entrée.

FIBONACCI_ACTIF
  → Aucun OB ni FVG détectable sur H1 ou M15. Les niveaux Fibonacci
    (0.382, 0.5, 0.618, 0.786) du swing d'impulsion sont les seuls
    aimants. État intermédiaire avant CONFIRMATION_ATTENDUE.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger("state_machine")

# ── Chemin du fichier d'état persistant ──────────────────────────────────────
# Stocké dans le repo GitHub → mis à jour par chaque run → commit automatique
STATE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "market_states.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Définition des états
# ─────────────────────────────────────────────────────────────────────────────

class MarketState(str, Enum):
    NEUTRE                    = "NEUTRE"
    BOS_DETECTE               = "BOS_DETECTE"
    PULLBACK_ZONE_FVG         = "PULLBACK_ZONE_FVG"
    PULLBACK_ZONE_OB          = "PULLBACK_ZONE_OB"
    CONFIRMATION_ATTENDUE     = "CONFIRMATION_ATTENDUE"
    ENTREE_ACTIVE             = "ENTREE_ACTIVE"
    RISQUE_CHOCH              = "RISQUE_CHOCH"
    RETOURNEMENT_SURVEILLANCE        = "RETOURNEMENT_SURVEILLANCE"
    RETOURNEMENT_PULLBACK_ATTENDU    = "RETOURNEMENT_PULLBACK_ATTENDU"
    FIBONACCI_ACTIF                  = "FIBONACCI_ACTIF"


class BosDirection(str, Enum):
    HAUSSIER = "HAUSSIER"   # BOS vers le haut → on cherche un SELL sur Boom
    BAISSIER = "BAISSIER"   # BOS vers le bas  → on cherche un BUY sur Crash


# ─────────────────────────────────────────────────────────────────────────────
# Structure de données de l'état d'un instrument
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstrumentState:
    """Contient tout ce que le bot doit se rappeler entre deux runs."""

    instrument: str
    state: MarketState = MarketState.NEUTRE

    # Direction du BOS en cours
    bos_direction: Optional[str] = None

    # Zones cartographiées après un BOS
    ob_high: Optional[float] = None        # Limite haute de l'Order Block
    ob_low: Optional[float] = None         # Limite basse de l'Order Block
    fvg_high: Optional[float] = None       # Limite haute du Fair Value Gap
    fvg_low: Optional[float] = None        # Limite basse du Fair Value Gap
    swing_point: Optional[float] = None    # Niveau d'invalidation (swing point)
    bos_candle_time: Optional[str] = None  # Horodatage de la bougie BOS

    # Niveaux Fibonacci (Scénario 3)
    fib_swing_high: Optional[float] = None
    fib_swing_low: Optional[float] = None
    fib_382: Optional[float] = None
    fib_500: Optional[float] = None
    fib_618: Optional[float] = None
    fib_786: Optional[float] = None

    # Scénario 2 — retournement
    choch_level: Optional[float] = None       # Niveau du CHoCH cassé
    ob_inverse_high: Optional[float] = None   # OB du scénario de retournement
    ob_inverse_low: Optional[float] = None
    false_breakout_high: Optional[float] = None  # Sommet de la fausse cassure

    # Signal actif
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    signal_direction: Optional[str] = None  # "BUY" ou "SELL"

    # Compteur de runs dans l'état courant (évite de rester bloqué)
    runs_in_state: int = 0

    # Historique des 5 derniers états (pour diagnostic)
    state_history: list = field(default_factory=list)

    # Horodatage de la dernière mise à jour
    last_updated: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Transitions autorisées
# ─────────────────────────────────────────────────────────────────────────────

# Définit depuis quel état on peut aller vers quel autre état.
# Toute transition non listée ici est rejetée avec un log d'erreur.
ALLOWED_TRANSITIONS: dict[MarketState, list[MarketState]] = {
    MarketState.NEUTRE: [
        MarketState.BOS_DETECTE,
    ],
    MarketState.BOS_DETECTE: [
        MarketState.PULLBACK_ZONE_FVG,
        MarketState.PULLBACK_ZONE_OB,
        MarketState.FIBONACCI_ACTIF,
        MarketState.RISQUE_CHOCH,      # swing point cassé avant zone
        MarketState.NEUTRE,
    ],
    MarketState.PULLBACK_ZONE_FVG: [
        MarketState.CONFIRMATION_ATTENDUE,
        MarketState.PULLBACK_ZONE_OB,  # le prix traverse le FVG
        MarketState.RISQUE_CHOCH,      # le prix traverse l'OB aussi
        MarketState.NEUTRE,
    ],
    MarketState.PULLBACK_ZONE_OB: [
        MarketState.CONFIRMATION_ATTENDUE,
        MarketState.RISQUE_CHOCH,      # le prix approche du swing point
        MarketState.NEUTRE,
    ],
    MarketState.CONFIRMATION_ATTENDUE: [
        MarketState.ENTREE_ACTIVE,     # confirmation détectée → signal envoyé
        MarketState.RISQUE_CHOCH,      # pas de confirmation, prix continue
        MarketState.NEUTRE,
    ],
    MarketState.ENTREE_ACTIVE: [
        MarketState.NEUTRE,            # trade fermé (TP ou SL atteint)
        MarketState.RETOURNEMENT_SURVEILLANCE,  # retournement pendant le trade
    ],
    MarketState.RISQUE_CHOCH: [
        MarketState.RETOURNEMENT_SURVEILLANCE,
        MarketState.PULLBACK_ZONE_OB,
        MarketState.NEUTRE,
    ],
    MarketState.RETOURNEMENT_SURVEILLANCE: [
        MarketState.RETOURNEMENT_PULLBACK_ATTENDU,  # BOS inverse détecté → pullback
        MarketState.NEUTRE,                          # structure invalidée
    ],
    MarketState.RETOURNEMENT_PULLBACK_ATTENDU: [
        MarketState.CONFIRMATION_ATTENDUE,  # pullback + confirmation → entrée
        MarketState.NEUTRE,                 # invalidé
    ],
    MarketState.FIBONACCI_ACTIF: [
        MarketState.CONFIRMATION_ATTENDUE,  # prix sur un niveau fib avec rejet
        MarketState.NEUTRE,                 # invalidé
    ],
}

# Nombre maximum de runs autorisés dans un même état avant reset forcé.
# Évite qu'un instrument reste bloqué indéfiniment dans un état obsolète.
MAX_RUNS_PER_STATE: dict[MarketState, int] = {
    MarketState.NEUTRE:                    999,   # pas de limite
    MarketState.BOS_DETECTE:               12,    # ~48h à 6 runs/jour
    MarketState.PULLBACK_ZONE_FVG:         6,
    MarketState.PULLBACK_ZONE_OB:          6,
    MarketState.CONFIRMATION_ATTENDUE:     3,
    MarketState.ENTREE_ACTIVE:             24,    # ~4 jours
    MarketState.RISQUE_CHOCH:              4,
    MarketState.RETOURNEMENT_SURVEILLANCE:     18,   # ~3 jours
    MarketState.RETOURNEMENT_PULLBACK_ATTENDU: 10,   # ~1.5 jours
    MarketState.FIBONACCI_ACTIF:               8,
}


# ─────────────────────────────────────────────────────────────────────────────
# Gestionnaire d'états
# ─────────────────────────────────────────────────────────────────────────────

class StateManager:
    """
    Charge, sauvegarde et fait évoluer les états de tous les instruments.
    Un seul fichier JSON contient l'état de tous les instruments.
    """

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self._ensure_data_dir()
        self._states: dict[str, InstrumentState] = self._load()

    def _ensure_data_dir(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    def _load(self) -> dict[str, InstrumentState]:
        """Charge les états depuis le fichier JSON."""
        if not os.path.exists(self.state_file):
            logger.info("Aucun fichier d'état trouvé — démarrage à zéro.")
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            states = {}
            for instrument, data in raw.items():
                data["state"] = MarketState(data["state"])
                states[instrument] = InstrumentState(**data)
            logger.info(f"{len(states)} état(s) chargé(s) depuis {self.state_file}")
            return states
        except Exception as e:
            logger.error(f"Erreur de chargement des états: {e} — reset complet.")
            return {}

    def save(self):
        """Sauvegarde tous les états dans le fichier JSON."""
        raw = {}
        for instrument, state in self._states.items():
            d = asdict(state)
            d["state"] = state.state.value
            raw[instrument] = d
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        logger.info(f"États sauvegardés → {self.state_file}")

    def get(self, instrument: str) -> InstrumentState:
        """Retourne l'état d'un instrument, NEUTRE si inconnu."""
        if instrument not in self._states:
            self._states[instrument] = InstrumentState(instrument=instrument)
        return self._states[instrument]

    def transition(
        self,
        instrument: str,
        new_state: MarketState,
        **updates,
    ) -> bool:
        """
        Effectue une transition d'état pour un instrument.

        - Vérifie que la transition est autorisée.
        - Met à jour les champs fournis dans `updates`.
        - Incrémente le compteur de runs.
        - Ajoute l'ancien état à l'historique.
        - Retourne True si la transition a eu lieu, False sinon.
        """
        current = self.get(instrument)
        current_state = current.state

        # Vérification de la transition
        allowed = ALLOWED_TRANSITIONS.get(current_state, [])
        if new_state not in allowed and new_state != current_state:
            logger.error(
                f"{instrument}: transition interdite "
                f"{current_state.value} → {new_state.value}"
            )
            return False

        # Historique (garde les 5 derniers états)
        history = current.state_history[-4:] + [current_state.value]

        # Mise à jour
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)
            else:
                logger.warning(f"Champ inconnu dans InstrumentState: {key}")

        if new_state != current_state:
            current.runs_in_state = 1
        else:
            current.runs_in_state += 1

        current.state = new_state
        current.state_history = history
        current.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"{instrument}: {current_state.value} → {new_state.value} "
            f"(run #{current.runs_in_state} dans cet état)"
        )
        return True

    def check_timeouts(self, instrument: str) -> bool:
        """
        Vérifie si l'instrument est resté trop longtemps dans son état.
        Si oui, remet à NEUTRE et retourne True (timeout déclenché).
        """
        current = self.get(instrument)
        max_runs = MAX_RUNS_PER_STATE.get(current.state, 999)
        if current.runs_in_state > max_runs:
            logger.warning(
                f"{instrument}: timeout dans l'état {current.state.value} "
                f"({current.runs_in_state} runs > max {max_runs}) → reset NEUTRE"
            )
            self.transition(instrument, MarketState.NEUTRE)
            return True
        return False

    def increment_run(self, instrument: str):
        """Incrémente le compteur de runs sans changer d'état."""
        current = self.get(instrument)
        current.runs_in_state += 1
        current.last_updated = datetime.now(timezone.utc).isoformat()

    def reset(self, instrument: str):
        """Remet un instrument à NEUTRE en effaçant toutes les zones."""
        self._states[instrument] = InstrumentState(instrument=instrument)
        logger.info(f"{instrument}: reset complet → NEUTRE")

    def summary(self) -> str:
        """Retourne un résumé texte de tous les états actuels."""
        lines = ["=== État actuel des instruments ==="]
        for instrument, state in self._states.items():
            lines.append(
                f"  {instrument:15s} | {state.state.value:30s} | "
                f"run #{state.runs_in_state}"
            )
        return "\n".join(lines)
