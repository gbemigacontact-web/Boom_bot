"""
scenario_engine.py
==================
Moteur de scénarios SMC. Intègre désormais l'analyse continue et l'IA.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from state_machine import (
    InstrumentState, MarketState, StateManager, BosDirection,
)
from technical_analysis import (
    Direction, BosSignal, ChochSignal, OrderBlock, FairValueGap,
    FibLevels, BougiConfirmation, TrendState,
    detect_trend, detect_bos, detect_choch,
    find_order_blocks, find_ob_nearest_price,
    find_fair_value_gaps, find_fvg_nearest_price,
    compute_fibonacci, price_near_fib_level,
    check_confirmation_candle, price_in_zone,
    find_htf_targets, compute_sl_tp, find_swing_points,
    detect_mini_choch, detect_false_breakout,
    find_unmitigated_bullish_targets, find_unmitigated_bearish_targets,
)
from deriv_client import InstrumentData, get_current_price, is_spread_abnormal, detect_recent_spike
from ai_analyzer import analyze_signal_with_gemini
from market_analyzer import analyze_market, MarketDiagnosis

logger = logging.getLogger("scenario_engine")

INSTRUMENT_BIAS = {
    "Boom": Direction.BEARISH,
    "Crash": Direction.BULLISH,
}

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
    ai_analysis: str = ""
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
    diagnosis: Optional[MarketDiagnosis] = None
    ai_diagnosis: str = ""


def _instrument_family(name: str) -> str:
    return "Boom" if name.startswith("Boom") else "Crash"

def _trade_direction_from_bos(bos: BosSignal, family: str) -> Direction:
    return INSTRUMENT_BIAS[family]

def _opposite(direction: Direction) -> Direction:
    if direction == Direction.BULLISH:
        return Direction.BEARISH
    return Direction.BULLISH

def _build_context(trend_h4, trend_h1, current_price, bos=None, ob=None, fvg=None, fib=None, confirmation=None) -> list[str]:
    lines = []
    lines.append(f"Tendance H4: {trend_h4.direction.value} | H1: {trend_h1.direction.value}")
    lines.append(f"Prix actuel: {current_price:.4f}")
    if bos:
        lines.append(f"BOS {bos.direction.value} confirmé @ {bos.broken_level:.4f}")
    if ob:
        lines.append(f"OB [{ob.low:.4f} – {ob.high:.4f}] (mid={ob.mid:.4f})")
    if fvg:
        lines.append(f"FVG [{fvg.low:.4f} – {fvg.high:.4f}] (mid={fvg.mid:.4f})")
    if fib:
        lines.append(f"Fibonacci: 38.2%={fib.fib_382:.4f} | 50%={fib.fib_500:.4f} | 61.8%={fib.fib_618:.4f}")
    if confirmation:
        lines.append(f"Confirmation: {confirmation.kind.value} | mèche={confirmation.wick_ratio:.0%} | corps={confirmation.body_ratio:.0%} | vol={confirmation.volume_ratio:.1f}×")
    return lines


class ScenarioEngine:
    def __init__(self, state_manager: StateManager):
        self.sm = state_manager

    def diagnose(self, instrument: str, data: InstrumentData) -> MarketDiagnosis:
        """Analyse complète, appelée à chaque run, même sans signal."""
        h4 = data.get("H4", [])
        h1 = data.get("H1", [])
        m15 = data.get("M15", [])
        m5 = data.get("M5", [])
        return analyze_market(instrument, h4, h1, m15, m5)

    def process(self, instrument: str, data: InstrumentData) -> EngineResult:
        state = self.sm.get(instrument)
        state_before = state.state
        family = _instrument_family(instrument)

        if is_spread_abnormal(data):
            self.sm.increment_run(instrument)
            return EngineResult(instrument, state_before, state_before, skipped=True, skip_reason="Spread anormal")

        m5 = data.get("M5", [])
        if detect_recent_spike(m5):
            self.sm.increment_run(instrument)
            return EngineResult(instrument, state_before, state_before, skipped=True, skip_reason="Spike récent M5")

        self.sm.check_timeouts(instrument)
        state = self.sm.get(instrument)

        h4 = data.get("H4", [])
        h1 = data.get("H1", [])
        m15 = data.get("M15", [])
        m5 = data.get("M5", [])

        if not h4 or not h1 or not m15 or not m5:
            return EngineResult(instrument, state_before, state_before, skipped=True, skip_reason="Données manquantes")

        trend_h4 = detect_trend(h4)
        trend_h1 = detect_trend(h1)
        current_price = get_current_price(data)
        bias = INSTRUMENT_BIAS[family]

        s = state.state
        if s == MarketState.NEUTRE:
            return self._handle_neutre(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s == MarketState.BOS_DETECTE:
            return self._handle_bos_detecte(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s in (MarketState.PULLBACK_ZONE_FVG, MarketState.PULLBACK_ZONE_OB):
            return self._handle_pullback_zone(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s == MarketState.CONFIRMATION_ATTENDUE:
            return self._handle_confirmation(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s == MarketState.RISQUE_CHOCH:
            return self._handle_risque_choch(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s == MarketState.RETOURNEMENT_SURVEILLANCE:
            return self._handle_retournement(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s == MarketState.RETOURNEMENT_PULLBACK_ATTENDU:
            return self._handle_retournement_pullback(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s == MarketState.FIBONACCI_ACTIF:
            return self._handle_fibonacci(instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before)
        elif s == MarketState.ENTREE_ACTIVE:
            return self._handle_entree_active(instrument, state, current_price, state_before)

        self.sm.increment_run(instrument)
        return EngineResult(instrument, state_before, state.state)

    # ── NEUTRE ──
    def _handle_neutre(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        if trend_h4.is_range:
            self.sm.increment_run(instrument)
            return EngineResult(instrument, state_before, MarketState.NEUTRE, alert=f"{instrument}: H4 en range")

        bos = detect_bos(h1)
        if not bos:
            return EngineResult(instrument, state_before, MarketState.NEUTRE)

        expected_bos = Direction.BULLISH if bias == Direction.BEARISH else Direction.BEARISH
        if bos.direction != expected_bos:
            return EngineResult(instrument, state_before, MarketState.NEUTRE)

        trade_dir = bias
        ob = find_ob_nearest_price(h1, trade_dir, current_price)
        fvg = find_fvg_nearest_price(h1, trade_dir, current_price)
        highs, lows = find_swing_points(h1, window=3)
        swing_pt = (lows[-1].price if trade_dir == Direction.BEARISH and lows else
                    highs[-1].price if trade_dir == Direction.BULLISH and highs else None)

        self.sm.transition(instrument, MarketState.BOS_DETECTE,
                           bos_direction=bos.direction.value,
                           ob_high=ob.high if ob else None, ob_low=ob.low if ob else None,
                           fvg_high=fvg.high if fvg else None, fvg_low=fvg.low if fvg else None,
                           swing_point=swing_pt)
        context = _build_context(trend_h4, trend_h1, current_price, bos=bos, ob=ob, fvg=fvg)
        alert_msg = f"🔔 BOS {bos.direction.value} sur {instrument}\n" + "\n".join(context[1:])
        return EngineResult(instrument, state_before, MarketState.BOS_DETECTE, alert=alert_msg)

    # ── BOS_DETECTE ──
    def _handle_bos_detecte(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        ob_h, ob_l = state.ob_high, state.ob_low
        fvg_h, fvg_l = state.fvg_high, state.fvg_low
        swing_pt = state.swing_point

        if swing_pt is not None:
            if (bias == Direction.BEARISH and current_price < swing_pt) or (bias == Direction.BULLISH and current_price > swing_pt):
                self.sm.transition(instrument, MarketState.RISQUE_CHOCH)
                return EngineResult(instrument, state_before, MarketState.RISQUE_CHOCH, alert=f"⚠️ {instrument}: prix au-delà du swing point → RISQUE_CHOCH")

        if fvg_h and fvg_l and price_in_zone(current_price, fvg_l, fvg_h):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_FVG)
            return EngineResult(instrument, state_before, MarketState.PULLBACK_ZONE_FVG,
                                alert=f"📍 {instrument}: entrée FVG [{fvg_l:.4f} – {fvg_h:.4f}]")
        if ob_h and ob_l and price_in_zone(current_price, ob_l, ob_h):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_OB)
            return EngineResult(instrument, state_before, MarketState.PULLBACK_ZONE_OB,
                                alert=f"📍 {instrument}: entrée OB [{ob_l:.4f} – {ob_h:.4f}]")

        if not ob_h and not fvg_h:
            highs, lows = find_swing_points(h1, window=3)
            if highs and lows:
                sh = max(highs, key=lambda x: x.price)
                sl = min(lows, key=lambda x: x.price)
                fib = compute_fibonacci(sh.price, sl.price, bias)
                if fib:
                    self.sm.transition(instrument, MarketState.FIBONACCI_ACTIF,
                                       fib_swing_high=sh.price, fib_swing_low=sl.price,
                                       fib_382=fib.fib_382, fib_500=fib.fib_500,
                                       fib_618=fib.fib_618, fib_786=fib.fib_786)
                    return EngineResult(instrument, state_before, MarketState.FIBONACCI_ACTIF,
                                        alert=f"📐 {instrument}: Fibonacci activé (pas d'OB/FVG)")
        self.sm.increment_run(instrument)
        return EngineResult(instrument, state_before, MarketState.BOS_DETECTE)

    # ── PULLBACK ZONE ──
    def _handle_pullback_zone(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        conf = check_confirmation_candle(m15, bias, volume_multiplier=1.5)
        mini = detect_mini_choch(m15, bias)
        if conf.valid or (mini and mini.valid):
            chosen = conf if conf.valid else mini
            self.sm.transition(instrument, MarketState.CONFIRMATION_ATTENDUE,
                               entry_price=chosen.entry_price, signal_direction=bias.value)
            return EngineResult(instrument, state_before, MarketState.CONFIRMATION_ATTENDUE,
                                alert=f"✅ {instrument}: confirmation M15 ({chosen.kind.value})")

        ob_h, ob_l = state.ob_high, state.ob_low
        fvg_h, fvg_l = state.fvg_high, state.fvg_low
        in_ob = ob_h and ob_l and price_in_zone(current_price, ob_l, ob_h)
        in_fvg = fvg_h and fvg_l and price_in_zone(current_price, fvg_l, fvg_h)
        if not in_ob and not in_fvg:
            swing_pt = state.swing_point
            if swing_pt and ((bias == Direction.BEARISH and current_price < swing_pt) or (bias == Direction.BULLISH and current_price > swing_pt)):
                self.sm.transition(instrument, MarketState.RISQUE_CHOCH)
                return EngineResult(instrument, state_before, MarketState.RISQUE_CHOCH,
                                    alert=f"⚠️ {instrument}: sortie de zone sans confirmation → RISQUE_CHOCH")
        self.sm.increment_run(instrument)
        return EngineResult(instrument, state_before, state.state)

    # ── CONFIRMATION ATTENDUE ──
    def _handle_confirmation(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        entry_price = state.entry_price or current_price
        swing_pt = state.swing_point or current_price
        h4_targets = find_htf_targets(h4, bias, current_price)
        sl, tp1, tp2, tp3 = compute_sl_tp(bias, entry_price, swing_pt, h4_targets, m15)
        risk = abs(entry_price - sl)
        reward_tp1 = abs(tp1 - entry_price)
        if risk > 0 and reward_tp1 / risk < 1.5:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(instrument, state_before, MarketState.NEUTRE, alert=f"❌ {instrument}: RR insuffisant")

        conf_m5 = check_confirmation_candle(m5, bias, volume_multiplier=1.0)
        conf_kind = conf_m5.kind.value if conf_m5.valid else "ZONE"
        wick_r = conf_m5.wick_ratio if conf_m5.valid else 0
        body_r = conf_m5.body_ratio if conf_m5.valid else 0
        vol_r = conf_m5.volume_ratio if conf_m5.valid else 0

        scenario = "Scénario 1 — Continuation"
        if state_before == MarketState.RETOURNEMENT_PULLBACK_ATTENDU:
            scenario = "Scénario 2 — Retournement"
        elif state_before == MarketState.FIBONACCI_ACTIF:
            scenario = "Scénario 3 — Fibonacci"

        context = _build_context(trend_h4, trend_h1, current_price)
        # Appel IA
        ai_context = {
            "instrument": instrument,
            "trend_h4": trend_h4.direction.value,
            "trend_h1": trend_h1.direction.value,
            "scenario": scenario,
            "direction": bias.value,
            "current_price": current_price,
            "bos": "oui",
            "choch": "non" if "Continuation" in scenario else "oui",
            "ob": f"{state.ob_low}-{state.ob_high}" if state.ob_high else "aucun",
            "fvg": f"{state.fvg_low}-{state.fvg_high}" if state.fvg_high else "aucun",
            "fib": "non",
            "confirmation": conf_kind,
        }
        ai_analysis = analyze_signal_with_gemini(ai_context)

        self.sm.transition(instrument, MarketState.ENTREE_ACTIVE,
                           entry_price=entry_price, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                           signal_direction=bias.value)

        signal = SignalResult(instrument=instrument, direction=bias, scenario=scenario,
                              state_before=state_before, state_after=MarketState.ENTREE_ACTIVE,
                              entry_price=entry_price, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                              confirmation_kind=conf_kind, wick_ratio=wick_r, body_ratio=body_r,
                              volume_ratio=vol_r, context_lines=context, ai_analysis=ai_analysis)
        return EngineResult(instrument, state_before, MarketState.ENTREE_ACTIVE, signal=signal)

    # ── RISQUE CHoCH ──
    def _handle_risque_choch(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        choch = detect_choch(h1, trend_h1.direction)
        if choch:
            reverse_bias = _opposite(bias)
            ob_inv = find_ob_nearest_price(h1, reverse_bias, current_price)
            highs, lows = find_swing_points(h1, window=3)
            major_high = max(h.price for h in highs) if highs else current_price
            major_low = min(l.price for l in lows) if lows else current_price
            self.sm.transition(instrument, MarketState.RETOURNEMENT_SURVEILLANCE,
                               choch_level=choch.broken_level,
                               ob_inverse_high=ob_inv.high if ob_inv else None,
                               ob_inverse_low=ob_inv.low if ob_inv else None,
                               major_high=major_high, major_low=major_low)
            return EngineResult(instrument, state_before, MarketState.RETOURNEMENT_SURVEILLANCE,
                                alert=f"🔄 {instrument}: CHoCH {choch.direction.value} → RETOURNEMENT_SURVEILLANCE")

        ob_h, ob_l = state.ob_high, state.ob_low
        if ob_h and ob_l and price_in_zone(current_price, ob_l, ob_h):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_OB)
            return EngineResult(instrument, state_before, MarketState.PULLBACK_ZONE_OB,
                                alert=f"↩️ {instrument}: rebond OB avant CHoCH")
        self.sm.increment_run(instrument)
        return EngineResult(instrument, state_before, MarketState.RISQUE_CHOCH)

    # ── RETOURNEMENT SURVEILLANCE ──
    def _handle_retournement(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        reverse_bias = _opposite(bias)
        # 1. Cartographier les aimants pour fausse cassure
        if bias == Direction.BEARISH:
            targets = find_unmitigated_bullish_targets(h1, current_price)
            if targets:
                state.false_breakout_target = targets[-1][2]  # high
        else:
            targets = find_unmitigated_bearish_targets(h1, current_price)
            if targets:
                state.false_breakout_target = targets[0][2]   # low

        # 2. BOS inverse ?
        bos_inv = detect_bos(h1)
        if bos_inv and bos_inv.direction == reverse_bias:
            if (bias == Direction.BEARISH and bos_inv.broken_level <= state.bos_inverse_level) or \
               (bias == Direction.BULLISH and bos_inv.broken_level >= state.bos_inverse_level):
                ob_pullback = find_ob_nearest_price(h1, reverse_bias, current_price)
                fvg_pullback = find_fvg_nearest_price(h1, reverse_bias, current_price)
                self.sm.transition(instrument, MarketState.RETOURNEMENT_PULLBACK_ATTENDU,
                                   entry_price=current_price, signal_direction=reverse_bias.value,
                                   ob_high=ob_pullback.high if ob_pullback else None,
                                   ob_low=ob_pullback.low if ob_pullback else None,
                                   fvg_high=fvg_pullback.high if fvg_pullback else None,
                                   fvg_low=fvg_pullback.low if fvg_pullback else None,
                                   swing_point=bos_inv.prior_swing.price)
                return EngineResult(instrument, state_before, MarketState.RETOURNEMENT_PULLBACK_ATTENDU,
                                    alert=f"✅ {instrument}: BOS {reverse_bias.value} confirmé → attente pullback")

        # 3. Fausse cassure (bonus)
        if bias == Direction.BEARISH and state.major_high:
            fb = detect_false_breakout(h1, state.major_high, Direction.BULLISH)
            if fb:
                state.false_breakout_high = fb['high']
        elif bias == Direction.BULLISH and state.major_low:
            fb = detect_false_breakout(h1, state.major_low, Direction.BEARISH)
            if fb:
                state.false_breakout_low = fb['low']

        # 4. Invalidation
        if bias == Direction.BEARISH and current_price > state.major_high:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(instrument, state_before, MarketState.NEUTRE, alert="Retournement annulé, retour haussier")
        if bias == Direction.BULLISH and current_price < state.major_low:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(instrument, state_before, MarketState.NEUTRE, alert="Retournement annulé, retour baissier")

        state.run_count += 1
        if state.run_count > 30:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(instrument, state_before, MarketState.NEUTRE, alert="Timeout retournement")
        return EngineResult(instrument, state_before, MarketState.RETOURNEMENT_SURVEILLANCE)

    # ── RETOURNEMENT PULLBACK ATTENDU ──
    def _handle_retournement_pullback(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        reverse_bias = _opposite(bias)
        ob_h, ob_l = state.ob_high, state.ob_low
        fvg_h, fvg_l = state.fvg_high, state.fvg_low
        swing_pt = state.swing_point

        if swing_pt and ((reverse_bias == Direction.BEARISH and current_price > swing_pt) or (reverse_bias == Direction.BULLISH and current_price < swing_pt)):
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(instrument, state_before, MarketState.NEUTRE, alert="Pullback retournement invalide")

        if fvg_h and fvg_l and price_in_zone(current_price, fvg_l, fvg_h):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_FVG)
            return EngineResult(instrument, state_before, MarketState.PULLBACK_ZONE_FVG,
                                alert=f"📍 {instrument}: pullback retournement dans FVG [{fvg_l:.4f} – {fvg_h:.4f}]")
        if ob_h and ob_l and price_in_zone(current_price, ob_l, ob_h):
            self.sm.transition(instrument, MarketState.PULLBACK_ZONE_OB)
            return EngineResult(instrument, state_before, MarketState.PULLBACK_ZONE_OB,
                                alert=f"📍 {instrument}: pullback retournement dans OB [{ob_l:.4f} – {ob_h:.4f}]")

        self.sm.increment_run(instrument)
        return EngineResult(instrument, state_before, MarketState.RETOURNEMENT_PULLBACK_ATTENDU)

    # ── FIBONACCI ──
    def _handle_fibonacci(self, instrument, data, state, bias, trend_h4, trend_h1, current_price, h4, h1, m15, m5, state_before):
        if not state.fib_swing_high or not state.fib_swing_low:
            self.sm.transition(instrument, MarketState.NEUTRE)
            return EngineResult(instrument, state_before, MarketState.NEUTRE, skipped=True, skip_reason="Fibonacci données manquantes")
        fib = compute_fibonacci(state.fib_swing_high, state.fib_swing_low, bias)
        if not fib:
            return EngineResult(instrument, state_before, MarketState.NEUTRE)
        near = price_near_fib_level(current_price, fib)
        if near:
            conf = check_confirmation_candle(m15, bias, volume_multiplier=1.5)
            mini = detect_mini_choch(m15, bias)
            if conf.valid or (mini and mini.valid):
                swing_ref = state.fib_swing_low if bias == Direction.BEARISH else state.fib_swing_high
                self.sm.transition(instrument, MarketState.CONFIRMATION_ATTENDUE,
                                   entry_price=current_price, swing_point=swing_ref, signal_direction=bias.value)
                return EngineResult(instrument, state_before, MarketState.CONFIRMATION_ATTENDUE,
                                    alert=f"📐 {instrument}: confirmation Fibonacci sur {near}")
        self.sm.increment_run(instrument)
        return EngineResult(instrument, state_before, MarketState.FIBONACCI_ACTIF)

    # ── ENTREE ACTIVE ──
    def _handle_entree_active(self, instrument, state, current_price, state_before):
        sl, tp1, direction = state.stop_loss, state.tp1, state.signal_direction
        if sl and tp1 and direction:
            if direction == "BEARISH":
                if current_price >= sl:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(instrument, state_before, MarketState.NEUTRE, alert=f"🛑 {instrument}: SL atteint")
                if current_price <= tp1:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(instrument, state_before, MarketState.NEUTRE, alert=f"🎯 {instrument}: TP1 atteint")
            else:
                if current_price <= sl:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(instrument, state_before, MarketState.NEUTRE, alert=f"🛑 {instrument}: SL atteint")
                if current_price >= tp1:
                    self.sm.transition(instrument, MarketState.NEUTRE)
                    return EngineResult(instrument, state_before, MarketState.NEUTRE, alert=f"🎯 {instrument}: TP1 atteint")
        self.sm.increment_run(instrument)
        return EngineResult(instrument, state_before, MarketState.ENTREE_ACTIVE)


def run_scenario_engine(market_data: dict[str, InstrumentData], state_manager: StateManager) -> list[EngineResult]:
    engine = ScenarioEngine(state_manager)
    results = []

    for instrument in market_data:
        try:
            diagnosis = engine.diagnose(instrument, market_data[instrument])
            res = engine.process(instrument, market_data[instrument])
            res.diagnosis = diagnosis

            if os.environ.get("GEMINI_API_KEY"):
                ai_prompt = {
                    "instrument": instrument,
                    "trend_h4": diagnosis.trend_h4.direction.value,
                    "trend_h1": diagnosis.trend_h1.direction.value,
                    "scenario": diagnosis.detected_scenario,
                    "details": diagnosis.scenario_details,
                    "current_price": diagnosis.current_price,
                    "zones": f"OB haussier: {diagnosis.nearest_ob_bull}, OB baissier: {diagnosis.nearest_ob_bear}",
                    "recommendation": diagnosis.recommendation,
                }
                res.ai_diagnosis = analyze_signal_with_gemini(ai_prompt)

            results.append(res)
            logger.info(f"{instrument}: {res.state_before.value} → {res.state_after.value}" +
                        (" [SIGNAL]" if res.signal else "") + (" [SKIP]" if res.skipped else ""))
        except Exception as e:
            logger.error(f"{instrument}: erreur {e}", exc_info=True)
    state_manager.save()
    return results
