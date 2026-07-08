"""
state_machine.py
================
Gestion des états FSM et persistance JSON.
"""

import json
import os
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict
from enum import Enum

logger = logging.getLogger("state_machine")


class MarketState(str, Enum):
    NEUTRE = "NEUTRE"
    BOS_DETECTE = "BOS_DETECTE"
    PULLBACK_ZONE_FVG = "PULLBACK_ZONE_FVG"
    PULLBACK_ZONE_OB = "PULLBACK_ZONE_OB"
    CONFIRMATION_ATTENDUE = "CONFIRMATION_ATTENDUE"
    RISQUE_CHOCH = "RISQUE_CHOCH"
    RETOURNEMENT_SURVEILLANCE = "RETOURNEMENT_SURVEILLANCE"
    RETOURNEMENT_PULLBACK_ATTENDU = "RETOURNEMENT_PULLBACK_ATTENDU"
    FIBONACCI_ACTIF = "FIBONACCI_ACTIF"
    ENTREE_ACTIVE = "ENTREE_ACTIVE"


class BosDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class InstrumentState:
    instrument: str
    state: MarketState = MarketState.NEUTRE
    # Scénario 1
    bos_direction: Optional[str] = None
    ob_high: Optional[float] = None
    ob_low: Optional[float] = None
    fvg_high: Optional[float] = None
    fvg_low: Optional[float] = None
    swing_point: Optional[float] = None
    # Scénario 2
    choch_level: Optional[float] = None
    ob_inverse_high: Optional[float] = None
    ob_inverse_low: Optional[float] = None
    major_high: Optional[float] = None
    major_low: Optional[float] = None
    bos_inverse_level: Optional[float] = None
    false_breakout_high: Optional[float] = None
    false_breakout_low: Optional[float] = None
    # Scénario 3
    fib_swing_high: Optional[float] = None
    fib_swing_low: Optional[float] = None
    fib_382: Optional[float] = None
    fib_500: Optional[float] = None
    fib_618: Optional[float] = None
    fib_786: Optional[float] = None
    # Entrée active
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    signal_direction: Optional[str] = None
    run_count: int = 0


class StateManager:
    def __init__(self, state_file: str):
        self.file = state_file
        self.states: Dict[str, InstrumentState] = {}
        self.load()

    def load(self):
        if os.path.exists(self.file):
            with open(self.file, "r") as f:
                data = json.load(f)
            for name, s in data.items():
                self.states[name] = InstrumentState(**s)
            logger.info("États chargés depuis %s", self.file)
        else:
            logger.info("Aucun fichier d'état trouvé — démarrage à zéro.")
            for instr in ["Boom 500", "Boom 900", "Boom 1000", "Crash 500", "Crash 900", "Crash 1000"]:
                self.states[instr] = InstrumentState(instrument=instr)

    def get(self, instrument: str) -> InstrumentState:
        return self.states[instrument]

    def transition(self, instrument: str, new_state: MarketState, **kwargs):
        s = self.states[instrument]
        s.state = new_state
        for k, v in kwargs.items():
            setattr(s, k, v)
        # Réinitialiser run_count lors d'une transition majeure
        if new_state != MarketState.RETOURNEMENT_SURVEILLANCE:
            s.run_count = 0

    def increment_run(self, instrument: str):
        self.states[instrument].run_count += 1

    def check_timeouts(self, instrument: str):
        s = self.states[instrument]
        if s.state == MarketState.RETOURNEMENT_SURVEILLANCE and s.run_count > 30:
            logger.info(f"Timeout retournement pour {instrument}")
            self.transition(instrument, MarketState.NEUTRE)

    def save(self):
        data = {name: asdict(s) for name, s in self.states.items()}
        with open(self.file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("États sauvegardés → %s", self.file)

    def summary(self) -> str:
        lines = ["=== État actuel des instruments ==="]
        for name, s in self.states.items():
            lines.append(f"{name}: {s.state.value}")
        return "\n".join(lines)
