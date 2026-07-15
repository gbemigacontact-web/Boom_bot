"""
fallback_engine.py
====================
Moteur de secours 100% Python — Architecture hybride IA du bot
Boom & Crash v2.

RÔLE DANS LA NOUVELLE ARCHITECTURE
-------------------------------------
Ce fichier est le scenario_engine.py d'origine, renommé et CONSERVÉ
INTACT dans sa logique de décision. Il ne pilote plus le bot en
temps normal — c'est scenario_orchestrator.py qui appelle Gemini en
priorité. Ce moteur ne prend le relais que lorsque l'IA est
indisponible pour un instrument donné (quota, panne réseau, image
manquante, réponse invalide) — et uniquement pour CET instrument, les
autres continuant normalement sur l'IA.

Interface attendue par scenario_orchestrator.py :
  - classe ScenarioEngine(state_manager) avec méthode
    .process(instrument, data) -> EngineResult
  - constante INSTRUMENT_BIAS: dict[str, Direction]
  - dataclasses EngineResult, SignalResult

Toute la logique SMC (3 scénarios, FSM stricte, confirmations M15,
calcul SL/TP) reste identique à la version validée précédemment.

Modifications strictement cosmétiques par rapport à l'original :
  - Ce docstring d'en-tête.
  - Suppression de l'import mort `detect_mini_choch` depuis
    technical_analysis.py : ce module définit sa PROPRE fonction locale
    du même nom (signature différente, logique différente), qui prenait
    de toute façon le dessus sur l'import — celui-ci n'était donc jamais
    réellement utilisé. Voir le commentaire à sa définition plus bas.
Aucune logique de décision n'a été modifiée.

3 scénarios SMC complets :
  Scénario 1 : Continuation après BOS
  Scénario 2 : Retournement complet (CHoCH → fausse cassure → BOS inverse → pullback)
  Scénario 3 : Fibonacci (aucun OB/FVG visible)

Solidifications conservées de la version précédente :
  [S1] Confirmation enrichie : volume ≥ 1.5× moyenne 20 M15 OU mini-CHoCH
  [S2] Pullback après BOS inverse via RETOURNEMENT_PULLBACK_ATTENDU
       + fausse cassure avec exigence de volume
  [S3] detect_false_breakout() pour la fausse cassure du Scénario 2
  [S4] detect_choch() utilise trend_h1.direction (tendance réelle)
  [S5] Filtre des FVG déjà mitigés
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from state_machine import (
    InstrumentState, MarketState, StateManager,
)
from technical_analysis import (
    Direction, BosSignal, ChochSignal, OrderBlock, FairValueGap,
    FibLevels, BougiConfirmation, TrendState, FalseBreakout,
    detect_trend, detect_bos, detect_choch,
    find_order_blocks, find_ob_nearest_price,
    find_fair_value_gaps, find_fvg_nearest_price,
    compute_fibonacci, price_near_fib_level,
    check_confirmation_candle, price_in_zone,
    detect_false_breakout,
    find_htf_targets, compute_sl_tp, find_swing_points,
    compute_volume_average,
)
from deriv_client import (
    InstrumentData, get_current_price,
    is_spread_abnormal, detect_recent_spike,
    INSTRUMENTS,
)

logger = logging.getLogger("fallback_engine")

INSTRUMENT_BIAS: dict[str, Direction] = {
    "Boom":  Direction.BEARISH,
    "Crash": Direction.BULLISH,
}

# Volume multiplier exigé pour les confirmations (solidification S1)
VOLUME_MULTIPLIER_CONFIRMATION = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Résultats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    instrument: str
    direction: Direction
    scenario: str
    state_before: MarketState
    state_after: MarketState
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confirmation_kind: str
    wick_ratio: float
    body_ratio: float
    volume_ratio: float
    context_lines: list[str] = field(default_factory=list)
    alert_only: bool = False
    alert_message: str = ""


@dataclass
class EngineResult:
    instrument: str
    state_before: MarketState
    state_after: MarketState
    signal: Optional[SignalResult] = None
    alert: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _family(name: str) -> str:
    return "Boom" if name.startswith("Boom") else "Crash"


def _opposite(d: Direction) -> Direction:
    return Direction.BEARISH if d == Direction.BULLISH else Direction.BULLISH


def _build_context(
    trend_h4: TrendState, trend_h1: TrendState, current_price: float,
    bos=None, ob=None, fvg=None, fib=None, confirmation=None,
) -> list[str]:
    lines = [
        f"Tendance H4: {trend_h4.direction.value} | H1: {trend_h1.direction.value}",
        f"Prix actuel: {current_price:.4f}",
    ]
    if bos:
        lines.append(f"BOS {bos.direction.value} @ {bos.broken_level:.4f}")
    if ob:
        lines.append(f"OB [{ob.low:.4f} – {ob.high:.4f}]")
    if fvg:
        lines.append(f"FVG [{fvg.low:.4f} – {fvg.high:.4f}]")
    if fib:
        lines.append(
            f"Fib: 38.2%={fib.fib_382:.4f} | "
            f"50%={fib.fib_500:.4f} | 61.8%={fib.fib_618:.4f}"
        )
    if confirmation:
        lines.append(
            f"Confirmation: {confirmation.kind.value} | "
            f"mèche={confirmation.wick_ratio:.0%} | "
            f"vol={confirmation.volume_ratio:.1f}×"
        )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# [S1] Mini-CHoCH sur M15
#
# NOTE : cette fonction locale porte le même nom qu'une fonction de
# technical_analysis.py, mais avec une signature et une logique
# différentes (celle-ci retourne un booléen simple, l'autre un objet
# BougiConfirmation détaillé avec zone_low/zone_high). C'est CETTE
# version locale qui est utilisée par _check_zone_confirmation ci-dessous
# — l'import depuis technical_analysis a donc été retiré pour éviter
# toute confusion, sans changer le comportement du moteur.
# ─────────────────────────────────────────────────────────────────────────────

def detect_mini_choch(candles: list[dict], bias: Direction) -> bool:
    """
    Détecte un petit changement de structure local sur M15.

    Pour un trade BEARISH (Boom) :
      Le prix fait un micro-sommet, descend, puis remonte et CASSE ce
      micro-sommet → preuve que les vendeurs reprennent le contrôle.

    Pour un trade BULLISH (Crash) :
      Le prix fait un micro-creux, remonte, puis descend et CASSE ce
      micro-creux → preuve que les acheteurs reprennent le contrôle.

    Fenêtre d'analyse : les 15 dernières bougies M15 (≈ 3h75).
    """
    if len(candles) < 6:
        return False

    window = candles[-15:]
    highs, lows = find_swing_points(window, window=2)

    if bias == Direction.BEARISH:
        if len(highs) < 1:
            return False
        last_sh = highs[-1]
        last_close = window[-1]["close"]
        if last_close > last_sh.price:
            idx = last_sh.index
            sub = window[idx:]
            if len(sub) >= 3:
                mid_low = min(c["low"] for c in sub[1:-1])
                if mid_low < last_sh.price:
                    logger.debug("Mini-CHoCH BEARISH détecté sur M15")
                    return True

    elif bias == Direction.BULLISH:
        if len(lows) < 1:
            return False
        last_sl = lows[-1]
        last_close = window[-1]["close"]
        if last_close < last_sl.price:
            idx = last_sl.index
            sub = window[idx:]
            if len(sub) >= 3:
                mid_high = max(c["high"] for c in sub[1:-1])
                if mid_high > last_sl.price:
                    logger.debug("Mini-CHoCH BULLISH détecté sur M15")
                    return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation enrichie (S1 + S2 partagée)
# ─────────────────────────────────────────────────────────────────────────────

def _check_zone_confirmation(
    m15: list[dict],
    bias: Direction,
    volume_multiplier: float = VOLUME_MULTIPLIER_CONFIRMATION,
) -> tuple[bool, str, BougiConfirmation | None]:
    """
    Vérifie la confirmation dans une zone (OB ou FVG).
    Méthode 1 : bougie de rejet avec volume ≥ 1.5× la moyenne 20 M15
    Méthode 2 : mini-CHoCH sur M15

    Retourne (is_confirmed, method_name, bougie_confirmation_or_None).
    """
    conf = check_confirmation_candle(
        m15, bias,
        wick_threshold=0.60,
        body_threshold=0.50,
        volume_multiplier=volume_multiplier,
    )
    if conf.valid:
        return True, "REJECTION_WICK+VOLUME", conf

    if detect_mini_choch(m15, bias):
        last = m15[-1]
        avg_vol = compute_volume_average(m15, 20)
        vol_r = last["volume"] / avg_vol if avg_vol else 0
        rng = last["high"] - last["low"]
        body = abs(last["close"] - last["open"])
        synthetic = BougiConfirmation(
            valid=True,
            kind=conf.kind,
            entry_candle=last,
            entry_price=last["close"],
            wick_ratio=0.0,
            body_ratio=body / rng if rng else 0,
            volume_ratio=vol_r,
        )
        return True, "MINI_CHOCH_M15", synthetic

    return False, "", conf


# ─────────────────────────────────────────────────────────────────────────────
# Moteur principal
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioEngine:

    def __init__(self, state_manager: StateManager):
        self.sm = state_manager

    def process(self, instrument: str, data: InstrumentData) -> EngineResult:
        state = self.sm.get(instrument)
        state_before = state.state
        family = _family(instrument)

        # ── Gardes globaux ────────────────────────────────────────────────────
        if is_spread_abnormal(data):
            self.sm.increment_run(instrument)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=state_before,
                skipped=True, skip_reason="Spread anormal",
            )

        if detect_recent_spike(data.get("M5", [])):
            self.sm.increment_run(instrument)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=state_before,
                skipped=True, skip_reason="Spike récent M5",
            )

        self.sm.check_timeouts(instrument)
        state = self.sm.get(instrument)

        h4  = data.get("H4", [])
        h1  = data.get("H1", [])
        m30 = data.get("M30", [])
        m15 = data.get("M15", [])
        m5  = data.get("M5",  [])

        if not h4 or not h1 or not m15 or not m5:
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=state_before,
                skipped=True, skip_reason="Données manquantes",
            )

        trend_h4 = detect_trend(h4)
        trend_h1 = detect_trend(h1)
        current_price = get_current_price(data)
        bias = INSTRUMENT_BIAS[family]

        s = state.state
        kwargs = dict(
            instrument=instrument, data=data, state=state, bias=bias,
            trend_h4=trend_h4, trend_h1=trend_h1,
            current_price=current_price,
            h4=h4, h1=h1, m30=m30, m15=m15, m5=m5,
            state_before=state_before,
        )

        handlers = {
            MarketState.NEUTRE:                        self._handle_neutre,
            MarketState.BOS_DETECTE:                   self._handle_bos_detecte,
            MarketState.PULLBACK_ZONE_FVG:             self._handle_pullback_zone,
            MarketState.PULLBACK_ZONE_OB:              self._handle_pullback_zone,
            MarketState.CONFIRMATION_ATTENDUE:         self._handle_confirmation,
            MarketState.RISQUE_CHOCH:                  self._handle_risque_choch,
            MarketState.RETOURNEMENT_SURVEILLANCE:     self._handle_retournement,
            MarketState.RETOURNEMENT_PULLBACK_ATTENDU: self._handle_retournement_pullback,
            MarketState.FIBONACCI_ACTIF:               self._handle_fibonacci,
            MarketState.ENTREE_ACTIVE:                 self._handle_entree_active,
        }

        handler = handlers.get(s)
        if handler:
            try:
                return handler(**kwargs)
            except Exception as e:
                logger.error(f"{instrument} handler {s.value}: {e}", exc_info=True)

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=state.state,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # NEUTRE
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_neutre(self, instrument, data, state, bias,
                       trend_h4, trend_h1, current_price,
                       h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        if trend_h4.is_range:
            self.sm.increment_run(instrument)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.NEUTRE,
                alert=f"{instrument}: H4 en range — aucun signal",
            )

        bos = detect_bos(h1)
        if not bos:
            self.sm.increment_run(instrument)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.NEUTRE,
            )

        expected_bos = Direction.BULLISH if bias == Direction.BEARISH else Direction.BEARISH
        if bos.direction != expected_bos:
            self.sm.increment_run(instrument)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.NEUTRE,
            )

        ob  = find_ob_nearest_price(h1, bias, current_price)
        fvg = find_fvg_nearest_price(h1, bias, current_price)
        highs, lows = find_swing_points(h1, window=3)
        swing_pt = (
            lows[-1].price  if bias == Direction.BEARISH and lows  else
            highs[-1].price if bias == Direction.BULLISH and highs else None
        )

        self.sm.transition(
            instrument, MarketState.BOS_DETECTE,
            bos_direction=bos.direction.value,
            ob_high=ob.high if ob else None,
            ob_low=ob.low   if ob else None,
            fvg_high=fvg.high if fvg else None,
            fvg_low=fvg.low   if fvg else None,
            swing_point=swing_pt,
        )

        parts = [f"🔔 BOS {bos.direction.value} détecté — {instrument}"]
        if ob:  parts.append(f"OB: [{ob.low:.4f} – {ob.high:.4f}]")
        if fvg: parts.append(f"FVG: [{fvg.low:.4f} – {fvg.high:.4f}]")
        if swing_pt:
            side = "sous" if bias == Direction.BEARISH else "au-dessus de"
            parts.append(f"Invalidation si clôture {side} {swing_pt:.4f}")
        parts.append("⏳ En attente du pullback.")

        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.BOS_DETECTE,
            alert="\n".join(parts),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # BOS_DETECTE
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_bos_detecte(self, instrument, data, state, bias,
                            trend_h4, trend_h1, current_price,
                            h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        ob_h, ob_l = state.ob_high, state.ob_low
        fvg_h, fvg_l = state.fvg_high, state.fvg_low
        swing_pt = state.swing_point

        if swing_pt:
            casse = (bias == Direction.BEARISH and current_price < swing_pt) or \
                    (bias == Direction.BULLISH and current_price > swing_pt)
            if casse:
                self.sm.transition(instrument, MarketState.RISQUE_CHOCH)
                return EngineResult(
                    instrument=instrument,
                    state_before=state_before, state_after=MarketState.RISQUE_CHOCH,
                    alert=f"⚠️ {instrument}: swing point cassé {swing_pt:.4f} → RISQUE_CHOCH",
                )

        if fvg_h and fvg_l and price_in_zone(current_price, fvg_l, fvg_h):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_FVG)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.PULLBACK_ZONE_FVG,
                alert=(
                    f"📍 {instrument}: prix dans le FVG [{fvg_l:.4f}–{fvg_h:.4f}]\n"
                    f"🔍 Surveillance M15 activée."
                ),
            )

        if ob_h and ob_l and price_in_zone(current_price, ob_l, ob_h):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_OB)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.PULLBACK_ZONE_OB,
                alert=(
                    f"📍 {instrument}: prix dans l'OB [{ob_l:.4f}–{ob_h:.4f}]\n"
                    f"🔍 Pullback profond — surveillance M15."
                ),
            )

        if not ob_h and not fvg_h:
            highs, lows = find_swing_points(h1, window=3)
            if highs and lows:
                sh = max(highs, key=lambda x: x.price)
                sl = min(lows,  key=lambda x: x.price)
                fib = compute_fibonacci(sh.price, sl.price, bias)
                if fib:
                    self.sm.transition(
                        instrument, MarketState.FIBONACCI_ACTIF,
                        fib_swing_high=sh.price, fib_swing_low=sl.price,
                        fib_382=fib.fib_382, fib_500=fib.fib_500,
                        fib_618=fib.fib_618, fib_786=fib.fib_786,
                    )
                    return EngineResult(
                        instrument=instrument,
                        state_before=state_before, state_after=MarketState.FIBONACCI_ACTIF,
                        alert=(
                            f"📐 {instrument}: aucun OB/FVG → Fibonacci\n"
                            f"38.2%={fib.fib_382:.4f} | 50%={fib.fib_500:.4f} | "
                            f"61.8%={fib.fib_618:.4f}"
                        ),
                    )

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.BOS_DETECTE,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PULLBACK_ZONE_FVG / PULLBACK_ZONE_OB
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_pullback_zone(self, instrument, data, state, bias,
                              trend_h4, trend_h1, current_price,
                              h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        confirmed, method, conf_obj = _check_zone_confirmation(m15, bias)

        if confirmed:
            self.sm.transition(
                instrument, MarketState.CONFIRMATION_ATTENDUE,
                entry_price=conf_obj.entry_price if conf_obj else current_price,
                signal_direction=bias.value,
            )
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.CONFIRMATION_ATTENDUE,
                alert=(
                    f"✅ {instrument}: confirmation dans la zone\n"
                    f"Méthode: {method}\n"
                    f"⏳ Calcul SL/TP en cours..."
                ),
            )

        in_ob  = state.ob_high  and price_in_zone(current_price, state.ob_low,  state.ob_high)
        in_fvg = state.fvg_high and price_in_zone(current_price, state.fvg_low, state.fvg_high)

        if not in_ob and not in_fvg:
            sp = state.swing_point
            casse = sp and (
                (bias == Direction.BEARISH and current_price < sp) or
                (bias == Direction.BULLISH and current_price > sp)
            )
            if casse:
                self.sm.transition(instrument, MarketState.RISQUE_CHOCH)
                return EngineResult(
                    instrument=instrument,
                    state_before=state_before, state_after=MarketState.RISQUE_CHOCH,
                    alert=f"⚠️ {instrument}: zone quittée sans confirmation → RISQUE_CHOCH",
                )

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=state.state,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIRMATION_ATTENDUE
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_confirmation(self, instrument, data, state, bias,
                             trend_h4, trend_h1, current_price,
                             h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        entry_price = state.entry_price or current_price
        swing_pt    = state.swing_point or current_price

        h4_targets = find_htf_targets(h4, bias, current_price)
        sl, tp1, tp2, tp3 = compute_sl_tp(bias, entry_price, swing_pt, h4_targets, m15)

        risk = abs(entry_price - sl)
        rr_tp1 = abs(tp1 - entry_price) / risk if risk else 0
        if rr_tp1 < 1.5:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.NEUTRE,
                alert=f"❌ {instrument}: RR insuffisant ({rr_tp1:.1f}) → annulé",
            )

        conf_m5 = check_confirmation_candle(m5, bias, volume_multiplier=1.0)
        conf_kind = conf_m5.kind.value if conf_m5.valid else "ZONE"
        wick_r    = conf_m5.wick_ratio  if conf_m5.valid else 0
        body_r    = conf_m5.body_ratio  if conf_m5.valid else 0
        vol_r     = conf_m5.volume_ratio if conf_m5.valid else 0

        history = state.state_history
        if MarketState.RETOURNEMENT_PULLBACK_ATTENDU.value in history:
            scenario = "Scénario 2 — Retournement"
        elif MarketState.FIBONACCI_ACTIF.value in history:
            scenario = "Scénario 3 — Fibonacci"
        else:
            scenario = "Scénario 1 — Continuation"

        self.sm.transition(
            instrument, MarketState.ENTREE_ACTIVE,
            entry_price=entry_price, stop_loss=sl,
            tp1=tp1, tp2=tp2, tp3=tp3,
            signal_direction=bias.value,
        )

        signal = SignalResult(
            instrument=instrument,
            direction=bias,
            scenario=scenario,
            state_before=state_before,
            state_after=MarketState.ENTREE_ACTIVE,
            entry_price=entry_price,
            stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            confirmation_kind=conf_kind,
            wick_ratio=wick_r, body_ratio=body_r, volume_ratio=vol_r,
            context_lines=_build_context(trend_h4, trend_h1, current_price),
        )

        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.ENTREE_ACTIVE,
            signal=signal,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RISQUE_CHOCH
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_risque_choch(self, instrument, data, state, bias,
                             trend_h4, trend_h1, current_price,
                             h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        choch = detect_choch(h1, bias)
        if choch:
            ob_inv = find_ob_nearest_price(h1, _opposite(bias), current_price)
            self.sm.transition(
                instrument, MarketState.RETOURNEMENT_SURVEILLANCE,
                choch_level=choch.broken_level,
                ob_inverse_high=ob_inv.high if ob_inv else None,
                ob_inverse_low=ob_inv.low   if ob_inv else None,
            )
            msg = (
                f"🔄 {instrument}: CHoCH {choch.direction.value} @ {choch.broken_level:.4f}\n"
                f"Scénario 2 activé."
            )
            if ob_inv:
                msg += f"\nOB inverse: [{ob_inv.low:.4f}–{ob_inv.high:.4f}]"
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.RETOURNEMENT_SURVEILLANCE,
                alert=msg,
            )

        if state.ob_high and price_in_zone(current_price, state.ob_low, state.ob_high):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_OB)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.PULLBACK_ZONE_OB,
                alert=f"↩️ {instrument}: rebond OB avant CHoCH",
            )

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.RISQUE_CHOCH,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RETOURNEMENT_SURVEILLANCE
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_retournement(self, instrument, data, state, bias,
                             trend_h4, trend_h1, current_price,
                             h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        reverse_bias = _opposite(bias)
        ob_inv_h = state.ob_inverse_high
        ob_inv_l = state.ob_inverse_low

        highs, lows = find_swing_points(h1, window=3)
        if highs and bias == Direction.BEARISH:
            last_high = highs[-1].price
            fb = detect_false_breakout(
                h1, last_high,
                direction_of_breakout=Direction.BULLISH,
                volume_multiplier=1.5,
                wick_ratio_threshold=0.40,
            )
            if fb and fb.valid:
                state_obj = self.sm.get(instrument)
                state_obj.false_breakout_high = fb.wick_extreme
                in_ob_inv = (
                    ob_inv_h and ob_inv_l and
                    price_in_zone(fb.close_price, ob_inv_l, ob_inv_h)
                )
                note = " dans OB inverse ✓" if in_ob_inv else ""
                return EngineResult(
                    instrument=instrument,
                    state_before=state_before,
                    state_after=MarketState.RETOURNEMENT_SURVEILLANCE,
                    alert=(
                        f"🎯 {instrument}: fausse cassure @ {fb.wick_extreme:.4f}{note}\n"
                        f"Volume: {fb.volume_ratio:.1f}× la moyenne\n"
                        f"Attente du BOS baissier de confirmation."
                    ),
                )

        bos_inv = detect_bos(h1)
        if bos_inv and bos_inv.direction == reverse_bias:
            ob_bos = find_ob_nearest_price(h1, reverse_bias, current_price)
            fvg_bos = find_fvg_nearest_price(h1, reverse_bias, current_price)
            sp_inv = bos_inv.prior_swing.price

            self.sm.transition(
                instrument, MarketState.RETOURNEMENT_PULLBACK_ATTENDU,
                ob_high=ob_bos.high if ob_bos else None,
                ob_low=ob_bos.low   if ob_bos else None,
                fvg_high=fvg_bos.high if fvg_bos else None,
                fvg_low=fvg_bos.low   if fvg_bos else None,
                swing_point=sp_inv,
                signal_direction=reverse_bias.value,
            )

            msg = (
                f"📉 {instrument}: BOS {reverse_bias.value} confirmé\n"
                f"Retournement validé — attente du pullback sur l'OB inverse."
            )
            if ob_bos:
                msg += f"\nOB du BOS inverse: [{ob_bos.low:.4f}–{ob_bos.high:.4f}]"
            return EngineResult(
                instrument=instrument,
                state_before=state_before,
                state_after=MarketState.RETOURNEMENT_PULLBACK_ATTENDU,
                alert=msg,
            )

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.RETOURNEMENT_SURVEILLANCE,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RETOURNEMENT_PULLBACK_ATTENDU
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_retournement_pullback(
        self, instrument, data, state, bias,
        trend_h4, trend_h1, current_price,
        h4, h1, m30, m15, m5, state_before, **_
    ) -> EngineResult:
        reverse_bias = _opposite(bias)
        ob_h, ob_l   = state.ob_high, state.ob_low
        fvg_h, fvg_l = state.fvg_high, state.fvg_low
        swing_pt     = state.swing_point

        if swing_pt:
            casse = (
                (reverse_bias == Direction.BEARISH and current_price > swing_pt) or
                (reverse_bias == Direction.BULLISH and current_price < swing_pt)
            )
            if casse:
                self.sm.transition(instrument, MarketState.NEUTRE)
                return EngineResult(
                    instrument=instrument,
                    state_before=state_before, state_after=MarketState.NEUTRE,
                    alert=f"❌ {instrument}: structure retournement invalidée → NEUTRE",
                )

        if fvg_h and fvg_l and price_in_zone(current_price, fvg_l, fvg_h):
            confirmed, method, conf_obj = _check_zone_confirmation(m15, reverse_bias)
            if confirmed:
                self.sm.transition(
                    instrument, MarketState.CONFIRMATION_ATTENDUE,
                    entry_price=conf_obj.entry_price if conf_obj else current_price,
                    signal_direction=reverse_bias.value,
                    swing_point=swing_pt,
                )
                return EngineResult(
                    instrument=instrument,
                    state_before=state_before, state_after=MarketState.CONFIRMATION_ATTENDUE,
                    alert=(
                        f"✅ {instrument}: confirmation retournement dans FVG\n"
                        f"Méthode: {method}"
                    ),
                )

        if ob_h and ob_l and price_in_zone(current_price, ob_l, ob_h):
            confirmed, method, conf_obj = _check_zone_confirmation(m15, reverse_bias)
            if confirmed:
                self.sm.transition(
                    instrument, MarketState.CONFIRMATION_ATTENDUE,
                    entry_price=conf_obj.entry_price if conf_obj else current_price,
                    signal_direction=reverse_bias.value,
                    swing_point=swing_pt,
                )
                return EngineResult(
                    instrument=instrument,
                    state_before=state_before, state_after=MarketState.CONFIRMATION_ATTENDUE,
                    alert=(
                        f"✅ {instrument}: confirmation retournement dans OB\n"
                        f"Méthode: {method}"
                    ),
                )

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.RETOURNEMENT_PULLBACK_ATTENDU,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FIBONACCI_ACTIF
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_fibonacci(self, instrument, data, state, bias,
                          trend_h4, trend_h1, current_price,
                          h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        if not state.fib_swing_high or not state.fib_swing_low:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.NEUTRE,
                skipped=True, skip_reason="Fibonacci: swings manquants",
            )

        fib = compute_fibonacci(state.fib_swing_high, state.fib_swing_low, bias)
        if not fib:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.NEUTRE,
                skipped=True, skip_reason="Fibonacci invalide",
            )

        near = price_near_fib_level(current_price, fib)

        if near in ("fib_382", "fib_500", "fib_618"):
            confirmed, method, conf_obj = _check_zone_confirmation(m15, bias)
            if confirmed:
                swing_ref = (
                    state.fib_swing_low  if bias == Direction.BEARISH
                    else state.fib_swing_high
                )
                self.sm.transition(
                    instrument, MarketState.CONFIRMATION_ATTENDUE,
                    entry_price=current_price,
                    swing_point=swing_ref,
                    signal_direction=bias.value,
                )
                return EngineResult(
                    instrument=instrument,
                    state_before=state_before, state_after=MarketState.CONFIRMATION_ATTENDUE,
                    alert=(
                        f"📐 {instrument}: Fibonacci {near} "
                        f"({getattr(fib, near):.4f}) — {method}"
                    ),
                )
            return EngineResult(
                instrument=instrument,
                state_before=state_before, state_after=MarketState.FIBONACCI_ACTIF,
                alert=f"📐 {instrument}: sur {near} ({getattr(fib, near):.4f}) — attente confirmation",
            )

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.FIBONACCI_ACTIF,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ENTREE_ACTIVE
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_entree_active(self, instrument, data, state, bias,
                              trend_h4, trend_h1, current_price,
                              h4, h1, m30, m15, m5, state_before, **_) -> EngineResult:
        sl, tp1 = state.stop_loss, state.tp1
        direction = state.signal_direction

        if sl and tp1 and direction:
            if direction == "BEARISH":
                if current_price >= sl:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(
                        instrument=instrument,
                        state_before=state_before, state_after=MarketState.NEUTRE,
                        alert=f"🛑 {instrument}: SL touché @ {current_price:.4f}",
                    )
                if current_price <= tp1:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(
                        instrument=instrument,
                        state_before=state_before, state_after=MarketState.NEUTRE,
                        alert=f"🎯 {instrument}: TP1 atteint @ {current_price:.4f}",
                    )
            else:
                if current_price <= sl:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(
                        instrument=instrument,
                        state_before=state_before, state_after=MarketState.NEUTRE,
                        alert=f"🛑 {instrument}: SL touché @ {current_price:.4f}",
                    )
                if current_price >= tp1:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(
                        instrument=instrument,
                        state_before=state_before, state_after=MarketState.NEUTRE,
                        alert=f"🎯 {instrument}: TP1 atteint @ {current_price:.4f}",
                    )

        self.sm.increment_run(instrument)
        return EngineResult(
            instrument=instrument,
            state_before=state_before, state_after=MarketState.ENTREE_ACTIVE,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée autonome (optionnel — utile pour tester le fallback seul,
# sans passer par scenario_orchestrator.py / sans clé Gemini). En usage
# normal, c'est scenario_orchestrator.py qui instancie ScenarioEngine et
# appelle .process() instrument par instrument.
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario_engine(
    market_data: dict[str, InstrumentData],
    state_manager: StateManager,
) -> list[EngineResult]:
    engine = ScenarioEngine(state_manager)
    results: list[EngineResult] = []

    for instrument in INSTRUMENTS:
        if instrument not in market_data:
            logger.warning(f"{instrument}: pas de données — ignoré")
            continue
        try:
            result = engine.process(instrument, market_data[instrument])
            results.append(result)
            logger.info(
                f"{instrument}: {result.state_before.value} → "
                f"{result.state_after.value}"
                + (" [SIGNAL]" if result.signal else "")
                + (" [SKIP]"   if result.skipped  else "")
            )
        except Exception as e:
            logger.error(f"{instrument}: erreur moteur — {e}", exc_info=True)

    state_manager.save()
    return results
