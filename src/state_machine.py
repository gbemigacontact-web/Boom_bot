"""
state_machine.py
================
Persistance d'état pour le bot Boom & Crash v2 — Architecture hybride IA.

DEUX RÔLES DISTINCTS DANS CE FICHIER
--------------------------------------

[SECTION A] FSM legacy (MarketState, InstrumentState, ALLOWED_TRANSITIONS,
StateManager.get/transition/check_timeouts/increment_run/reset/summary) :
CONSERVÉE INTÉGRALEMENT, sans aucune modification de comportement. C'est
le moteur d'état utilisé exclusivement par fallback_engine.py — le filet
de sécurité Python pur qui prend le relais quand Gemini est indisponible
(quota, panne réseau, réponse invalide). Ne pas modifier cette section
sans revalider tout le fallback.

[SECTION B] Mémoire IA (AIMemory, ActiveTrade) : NOUVEAU. GitHub Actions
ne garde aucune mémoire entre deux runs — sans cette persistance, Gemini
repartirait de zéro à chaque exécution et perdrait le fil de sa lecture
structurelle. Cette section stocke le dernier verdict de Gemini par
instrument (scénario, biais, zone retenue, confiance...) pour le
réinjecter dans le prompt du run suivant (voir ai_analyzer.py →
`previous_context`), ainsi que les positions actives ouvertes par l'IA.

Les deux sections partagent le MÊME fichier JSON (un seul commit Git par
run, comme avant) mais restent des espaces de données indépendants.

DÉCOUPLAGE VOLONTAIRE
------------------------
Ce module n'importe RIEN depuis ai_analyzer.py (pas de dépendance à
google-genai ni pydantic). Les méthodes de mise à jour de la mémoire IA
acceptent des valeurs simples (str, bool, float), jamais l'objet
GeminiVerdict directement — c'est scenario_orchestrator.py qui fera le
pont. Ça garde ce fichier léger, testable seul, et utilisable même dans
un contexte qui n'a pas les dépendances Gemini installées.
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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — FSM legacy (inchangée, réservée à fallback_engine.py)
# ═══════════════════════════════════════════════════════════════════════════

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
    HAUSSIER = "HAUSSIER"
    BAISSIER = "BAISSIER"


@dataclass
class InstrumentState:
    """Contient tout ce que le fallback Python doit se rappeler entre deux runs."""

    instrument: str
    state: MarketState = MarketState.NEUTRE

    bos_direction: Optional[str] = None

    ob_high: Optional[float] = None
    ob_low: Optional[float] = None
    fvg_high: Optional[float] = None
    fvg_low: Optional[float] = None
    swing_point: Optional[float] = None
    bos_candle_time: Optional[str] = None

    fib_swing_high: Optional[float] = None
    fib_swing_low: Optional[float] = None
    fib_382: Optional[float] = None
    fib_500: Optional[float] = None
    fib_618: Optional[float] = None
    fib_786: Optional[float] = None

    choch_level: Optional[float] = None
    ob_inverse_high: Optional[float] = None
    ob_inverse_low: Optional[float] = None
    false_breakout_high: Optional[float] = None

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    signal_direction: Optional[str] = None

    runs_in_state: int = 0
    state_history: list = field(default_factory=list)
    last_updated: Optional[str] = None


ALLOWED_TRANSITIONS: dict[MarketState, list[MarketState]] = {
    MarketState.NEUTRE: [
        MarketState.BOS_DETECTE,
    ],
    MarketState.BOS_DETECTE: [
        MarketState.PULLBACK_ZONE_FVG,
        MarketState.PULLBACK_ZONE_OB,
        MarketState.FIBONACCI_ACTIF,
        MarketState.RISQUE_CHOCH,
        MarketState.NEUTRE,
    ],
    MarketState.PULLBACK_ZONE_FVG: [
        MarketState.CONFIRMATION_ATTENDUE,
        MarketState.PULLBACK_ZONE_OB,
        MarketState.RISQUE_CHOCH,
        MarketState.NEUTRE,
    ],
    MarketState.PULLBACK_ZONE_OB: [
        MarketState.CONFIRMATION_ATTENDUE,
        MarketState.RISQUE_CHOCH,
        MarketState.NEUTRE,
    ],
    MarketState.CONFIRMATION_ATTENDUE: [
        MarketState.ENTREE_ACTIVE,
        MarketState.RISQUE_CHOCH,
        MarketState.NEUTRE,
    ],
    MarketState.ENTREE_ACTIVE: [
        MarketState.NEUTRE,
        MarketState.RETOURNEMENT_SURVEILLANCE,
    ],
    MarketState.RISQUE_CHOCH: [
        MarketState.RETOURNEMENT_SURVEILLANCE,
        MarketState.PULLBACK_ZONE_OB,
        MarketState.NEUTRE,
    ],
    MarketState.RETOURNEMENT_SURVEILLANCE: [
        MarketState.RETOURNEMENT_PULLBACK_ATTENDU,
        MarketState.NEUTRE,
    ],
    MarketState.RETOURNEMENT_PULLBACK_ATTENDU: [
        MarketState.CONFIRMATION_ATTENDUE,
        MarketState.NEUTRE,
    ],
    MarketState.FIBONACCI_ACTIF: [
        MarketState.CONFIRMATION_ATTENDUE,
        MarketState.NEUTRE,
    ],
}

MAX_RUNS_PER_STATE: dict[MarketState, int] = {
    MarketState.NEUTRE:                    999,
    MarketState.BOS_DETECTE:               12,
    MarketState.PULLBACK_ZONE_FVG:         6,
    MarketState.PULLBACK_ZONE_OB:          6,
    MarketState.CONFIRMATION_ATTENDUE:     3,
    MarketState.ENTREE_ACTIVE:             24,
    MarketState.RISQUE_CHOCH:              4,
    MarketState.RETOURNEMENT_SURVEILLANCE:     18,
    MarketState.RETOURNEMENT_PULLBACK_ATTENDU: 10,
    MarketState.FIBONACCI_ACTIF:               8,
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Mémoire IA (nouveau — continuité du jugement Gemini)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ActiveTrade:
    """
    Position ouverte, quelle que soit son origine (IA ou fallback
    Python). Une fois créée, son suivi (TP/SL) est une simple
    comparaison numérique — aucun appel IA n'est nécessaire tant que ni
    le SL ni un TP n'est touché.
    """
    direction: str              # "HAUSSIER" | "BAISSIER"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    scenario: str = ""
    confirmation_kind: str = ""
    source: str = "IA"          # "IA" | "FALLBACK"
    opened_at: Optional[str] = None


@dataclass
class AIMemory:
    """
    Dernier verdict structurel de Gemini pour un instrument, conservé
    pour donner de la continuité au run suivant (voir
    ai_analyzer.py::_build_context_text → previous_context).
    """
    instrument: str
    last_scenario: Optional[str] = None
    last_biais: Optional[str] = None
    last_tendance_h4: Optional[str] = None
    last_tendance_h1: Optional[str] = None
    last_zone_label: Optional[str] = None
    last_confirmation_m15: Optional[bool] = None
    last_type_confirmation: Optional[str] = None
    last_action: Optional[str] = None
    last_confiance: Optional[int] = None
    last_commentaire: Optional[str] = None
    last_source: Optional[str] = None       # "IA" | "FALLBACK"
    # Nombre de runs consécutifs où la lecture (scénario + action) n'a
    # pas changé — utile pour repérer une lecture "figée" trop longtemps.
    consecutive_watch_runs: int = 0
    active_trade: Optional[ActiveTrade] = None
    last_updated: Optional[str] = None

    def to_previous_context(self) -> Optional[dict]:
        """Formate la mémoire en dict compact pour le prompt Gemini."""
        ctx: dict = {}
        if self.last_scenario:
            ctx.update({
                "scenario_precedent": self.last_scenario,
                "biais_precedent": self.last_biais,
                "zone_precedente": self.last_zone_label,
                "confirmation_precedente": self.last_confirmation_m15,
                "action_precedente": self.last_action,
                "confiance_precedente": self.last_confiance,
                "runs_consecutifs_meme_lecture": self.consecutive_watch_runs,
                "derniere_source": self.last_source,
            })
        if self.active_trade:
            t = self.active_trade
            ctx["position_active"] = (
                f"{t.direction} depuis {t.entry_price:.4f} "
                f"(SL={t.stop_loss:.4f}, scénario={t.scenario})"
            )
        return ctx or None


# ═══════════════════════════════════════════════════════════════════════════
# Gestionnaire unifié
# ═══════════════════════════════════════════════════════════════════════════

class StateManager:
    """
    Charge, met à jour et sauvegarde à la fois la FSM legacy (section A)
    et la mémoire IA (section B) dans un seul fichier JSON.

    L'API legacy (get/transition/check_timeouts/increment_run/reset/
    summary) garde exactement les mêmes signatures et le même
    comportement qu'avant — fallback_engine.py fonctionne sans aucune
    modification.
    """

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self._fsm_states: dict[str, InstrumentState] = {}
        self._ai_memory: dict[str, AIMemory] = {}
        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    # ── Chargement / migration ───────────────────────────────────────────────

    @staticmethod
    def _split_entry(entry: dict) -> tuple[Optional[dict], Optional[dict]]:
        """
        Sépare une entrée JSON en (données FSM, données mémoire IA).
        Gère la migration automatique depuis l'ancien format (où
        l'entrée était directement un InstrumentState à plat, sans les
        clés "fsm"/"ai_memory") — aucun fichier existant ne peut faire
        planter le chargement.
        """
        if "fsm" in entry or "ai_memory" in entry:
            return entry.get("fsm"), entry.get("ai_memory")
        return entry, None  # ancien format : tout le dict est un InstrumentState

    def _load(self) -> None:
        if not os.path.exists(self.state_file):
            logger.info("Aucun fichier d'état trouvé — démarrage à zéro.")
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                raw = json.load(f)

            for instrument, entry in raw.items():
                fsm_data, ai_data = self._split_entry(entry)

                if fsm_data:
                    try:
                        fsm_data = dict(fsm_data)
                        fsm_data["state"] = MarketState(fsm_data["state"])
                        self._fsm_states[instrument] = InstrumentState(**fsm_data)
                    except Exception as e:
                        logger.warning(
                            f"{instrument}: état FSM illisible ({e}) — ignoré"
                        )

                if ai_data:
                    try:
                        ai_data = dict(ai_data)
                        trade_data = ai_data.pop("active_trade", None)
                        mem = AIMemory(instrument=instrument, **ai_data)
                        if trade_data:
                            mem.active_trade = ActiveTrade(**trade_data)
                        self._ai_memory[instrument] = mem
                    except Exception as e:
                        logger.warning(
                            f"{instrument}: mémoire IA illisible ({e}) — ignorée"
                        )

            logger.info(
                f"{len(self._fsm_states)} état(s) FSM + "
                f"{len(self._ai_memory)} mémoire(s) IA chargé(s) depuis "
                f"{self.state_file}"
            )
        except Exception as e:
            logger.error(f"Erreur de chargement des états: {e} — reset complet.")
            self._fsm_states = {}
            self._ai_memory = {}

    def save(self) -> None:
        raw: dict[str, dict] = {}
        instruments = set(self._fsm_states) | set(self._ai_memory)
        for instrument in instruments:
            entry: dict = {}
            if instrument in self._fsm_states:
                d = asdict(self._fsm_states[instrument])
                d["state"] = self._fsm_states[instrument].state.value
                entry["fsm"] = d
            if instrument in self._ai_memory:
                entry["ai_memory"] = asdict(self._ai_memory[instrument])
            raw[instrument] = entry

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        logger.info(f"États sauvegardés → {self.state_file}")

    # ═════════════════════════════════════════════════════════════════════
    # API legacy — INCHANGÉE, réservée à fallback_engine.py
    # ═════════════════════════════════════════════════════════════════════

    def get(self, instrument: str) -> InstrumentState:
        """Retourne l'état FSM d'un instrument, NEUTRE si inconnu."""
        if instrument not in self._fsm_states:
            self._fsm_states[instrument] = InstrumentState(instrument=instrument)
        return self._fsm_states[instrument]

    def transition(
        self,
        instrument: str,
        new_state: MarketState,
        **updates,
    ) -> bool:
        current = self.get(instrument)
        current_state = current.state

        allowed = ALLOWED_TRANSITIONS.get(current_state, [])
        if new_state not in allowed and new_state != current_state:
            logger.error(
                f"{instrument}: transition interdite "
                f"{current_state.value} → {new_state.value}"
            )
            return False

        history = current.state_history[-4:] + [current_state.value]

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
        current = self.get(instrument)
        current.runs_in_state += 1
        current.last_updated = datetime.now(timezone.utc).isoformat()

    def reset(self, instrument: str):
        self._fsm_states[instrument] = InstrumentState(instrument=instrument)
        logger.info(f"{instrument}: reset FSM complet → NEUTRE")

    def summary(self) -> str:
        lines = ["=== État actuel des instruments (FSM fallback) ==="]
        for instrument, state in self._fsm_states.items():
            lines.append(
                f"  {instrument:15s} | {state.state.value:30s} | "
                f"run #{state.runs_in_state}"
            )
        if self._ai_memory:
            lines.append("=== Mémoire IA ===")
            for instrument, mem in self._ai_memory.items():
                trade_note = " [POSITION ACTIVE]" if mem.active_trade else ""
                lines.append(
                    f"  {instrument:15s} | scénario={mem.last_scenario or '-':20s} | "
                    f"action={mem.last_action or '-':10s}{trade_note}"
                )
        return "\n".join(lines)

    # ═════════════════════════════════════════════════════════════════════
    # API mémoire IA — nouveau
    # ═════════════════════════════════════════════════════════════════════

    def get_ai_memory(self, instrument: str) -> AIMemory:
        if instrument not in self._ai_memory:
            self._ai_memory[instrument] = AIMemory(instrument=instrument)
        return self._ai_memory[instrument]

    def update_ai_memory(
        self,
        instrument: str,
        *,
        scenario: str,
        biais: str,
        tendance_h4: str,
        tendance_h1: str,
        zone_label: Optional[str],
        confirmation_m15: bool,
        type_confirmation: str,
        action: str,
        confiance: int,
        commentaire: str,
        source: str = "IA",
    ) -> AIMemory:
        """
        Met à jour la mémoire IA après un verdict (Gemini ou fallback).
        Volontairement découplé de GeminiVerdict (ai_analyzer.py) — ce
        module reste sans dépendance sur google-genai/pydantic.
        """
        mem = self.get_ai_memory(instrument)

        same_reading = (mem.last_scenario == scenario and mem.last_action == action)
        mem.consecutive_watch_runs = (
            mem.consecutive_watch_runs + 1 if same_reading else 1
        )

        mem.last_scenario = scenario
        mem.last_biais = biais
        mem.last_tendance_h4 = tendance_h4
        mem.last_tendance_h1 = tendance_h1
        mem.last_zone_label = zone_label
        mem.last_confirmation_m15 = confirmation_m15
        mem.last_type_confirmation = type_confirmation
        mem.last_action = action
        mem.last_confiance = confiance
        mem.last_commentaire = commentaire
        mem.last_source = source
        mem.last_updated = datetime.now(timezone.utc).isoformat()
        return mem

    def set_active_trade(self, instrument: str, trade: ActiveTrade) -> None:
        mem = self.get_ai_memory(instrument)
        trade.opened_at = trade.opened_at or datetime.now(timezone.utc).isoformat()
        mem.active_trade = trade
        mem.last_updated = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"{instrument}: position active enregistrée ({trade.source}) "
            f"{trade.direction} @ {trade.entry_price:.4f}"
        )

    def clear_active_trade(self, instrument: str, reason: str = "") -> None:
        mem = self.get_ai_memory(instrument)
        if mem.active_trade:
            logger.info(f"{instrument}: position clôturée — {reason}")
        mem.active_trade = None
        mem.last_updated = datetime.now(timezone.utc).isoformat()

    def get_active_trade(self, instrument: str) -> Optional[ActiveTrade]:
        mem = self._ai_memory.get(instrument)
        return mem.active_trade if mem else None

    def has_active_position(self, instrument: str) -> bool:
        """
        True si une position est active, QUELLE QUE SOIT SA SOURCE
        (IA ou fallback Python). L'orchestrateur interroge cette
        méthode en premier à chaque run : si True, il se contente d'une
        vérification de prix (SL/TP) sans appeler Gemini — évite un
        appel API inutile pour une simple surveillance de trade.
        """
        if self.get_active_trade(instrument) is not None:
            return True
        fsm_state = self._fsm_states.get(instrument)
        return fsm_state is not None and fsm_state.state == MarketState.ENTREE_ACTIVE

    def get_previous_ai_context(self, instrument: str) -> Optional[dict]:
        """Raccourci pour ai_analyzer.py — contexte de continuité du run précédent."""
        return self.get_ai_memory(instrument).to_previous_context()
