"""
market_analyzer.py
==================
Analyse structurelle complète du marché pour chaque instrument.
Retourne un diagnostic détaillé même en l'absence de signal.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from technical_analysis import (
    Candle, Direction, TrendState, BosSignal, ChochSignal,
    detect_trend, detect_bos, detect_choch,
    find_swing_points, find_order_blocks, find_fair_value_gaps,
    find_unmitigated_bullish_targets, find_unmitigated_bearish_targets,
)


@dataclass
class MarketDiagnosis:
    instrument: str
    trend_h4: TrendState
    trend_h1: TrendState
    current_price: float

    # Structure
    last_bos: Optional[BosSignal] = None
    last_choch: Optional[ChochSignal] = None
    bos_count_recent: int = 0
    is_retournement_in_progress: bool = False
    is_range: bool = False

    # Zones techniques
    nearest_ob_bull: Optional[dict] = None
    nearest_ob_bear: Optional[dict] = None
    nearest_fvg_bull: Optional[dict] = None
    nearest_fvg_bear: Optional[dict] = None
    unmitigated_bullish_targets: list = field(default_factory=list)
    unmitigated_bearish_targets: list = field(default_factory=list)

    # Scénario détecté
    detected_scenario: str = "AUCUN"
    scenario_details: str = ""

    # Niveaux clés
    major_high: Optional[float] = None
    major_low: Optional[float] = None
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)

    # Recommandation
    recommendation: str = "ATTENDRE"
    recommendation_reason: str = ""

    # Résumé texte
    summary: str = ""


def analyze_market(instrument: str, h4: List[Candle], h1: List[Candle],
                   m15: List[Candle], m5: List[Candle]) -> MarketDiagnosis:
    """Analyse complète d'un instrument, retourne un diagnostic structuré."""

    current_price = m5[-1]["close"] if m5 else h1[-1]["close"]

    # Tendances
    trend_h4 = detect_trend(h4)
    trend_h1 = detect_trend(h1)

    # Swings
    highs_h1, lows_h1 = find_swing_points(h1, window=3)
    highs_h4, lows_h4 = find_swing_points(h4, window=3)

    major_high = max(h.price for h in highs_h4) if highs_h4 else None
    major_low = min(l.price for l in lows_h4) if lows_h4 else None

    swing_highs = [h.price for h in highs_h1[-5:]] if highs_h1 else []
    swing_lows = [l.price for l in lows_h1[-5:]] if lows_h1 else []

    # BOS récent sur H1
    last_bos = detect_bos(h1)
    bos_count = 0
    if last_bos:
        for i in range(max(0, len(h1)-50), len(h1)):
            sub = h1[:i+1]
            if detect_bos(sub):
                bos_count += 1

    # CHoCH récent
    last_choch = detect_choch(h1, trend_h1.direction)

    # Zones techniques
    obs_bull = find_order_blocks(h1, Direction.BULLISH, lookback=100)
    obs_bear = find_order_blocks(h1, Direction.BEARISH, lookback=100)
    fvgs_bull = [f for f in find_fair_value_gaps(h1, lookback=100) if f.direction == Direction.BULLISH and not f.mitigated]
    fvgs_bear = [f for f in find_fair_value_gaps(h1, lookback=100) if f.direction == Direction.BEARISH and not f.mitigated]

    nearest_ob_bull = {"mid": obs_bull[0].mid, "high": obs_bull[0].high, "low": obs_bull[0].low} if obs_bull else None
    nearest_ob_bear = {"mid": obs_bear[0].mid, "high": obs_bear[0].high, "low": obs_bear[0].low} if obs_bear else None
    nearest_fvg_bull = {"mid": fvgs_bull[0].mid, "high": fvgs_bull[0].high, "low": fvgs_bull[0].low} if fvgs_bull else None
    nearest_fvg_bear = {"mid": fvgs_bear[0].mid, "high": fvgs_bear[0].high, "low": fvgs_bear[0].low} if fvgs_bear else None

    unmitigated_bull = find_unmitigated_bullish_targets(h1, current_price)
    unmitigated_bear = find_unmitigated_bearish_targets(h1, current_price)

    # Déterminer le scénario
    detected_scenario = "AUCUN"
    scenario_details = ""
    is_retournement = False
    is_range = trend_h4.is_range

    if is_range:
        detected_scenario = "RANGE"
        scenario_details = f"Marché en range sur H4 (écart EMA < 0.05%). Attendre une direction claire."
    elif last_choch:
        detected_scenario = "RETOURNEMENT"
        is_retournement = True
        scenario_details = f"CHoCH {last_choch.direction.value} détecté à {last_choch.broken_level:.4f}. Surveillance d'un BOS inverse."
    elif last_bos:
        if (trend_h1.direction == Direction.BULLISH and last_bos.direction == Direction.BULLISH) or \
           (trend_h1.direction == Direction.BEARISH and last_bos.direction == Direction.BEARISH):
            detected_scenario = "CONTINUATION"
            scenario_details = f"BOS {last_bos.direction.value} aligné avec la tendance H1. Attente pullback vers zones techniques."
        else:
            detected_scenario = "CONTINUATION"
            scenario_details = f"BOS {last_bos.direction.value} détecté, mais tendance H1 est {trend_h1.direction.value}."
    else:
        detected_scenario = "ATTENTE"
        scenario_details = "Aucune structure claire récente. En attente d'un BOS ou CHoCH."

    # Recommandation
    recommendation = "ATTENDRE"
    reason = ""
    if detected_scenario == "RETOURNEMENT":
        recommendation = "SURVEILLER"
        reason = f"CHoCH détecté. Attendre un BOS inverse pour confirmation, puis un pullback."
    elif detected_scenario == "CONTINUATION":
        recommendation = "SURVEILLER"
        reason = f"BOS détecté. Surveiller l'entrée du prix dans les zones de pullback (OB/FVG)."
    elif detected_scenario == "RANGE":
        recommendation = "ÉVITER"
        reason = "Pas de direction claire. Attendre une cassure de range."

    # Construire le résumé
    summary_parts = [
        f"📊 {instrument} — Diagnostic",
        f"Tendance H4: {trend_h4.direction.value} | H1: {trend_h1.direction.value}",
        f"Scénario: {detected_scenario}",
        f"{scenario_details}",
        f"Prix: {current_price:.4f}",
    ]
    if major_high and major_low:
        summary_parts.append(f"Niveaux clés → High: {major_high:.4f} | Low: {major_low:.4f}")
    if nearest_ob_bull:
        summary_parts.append(f"OB haussier: [{nearest_ob_bull['low']:.4f} – {nearest_ob_bull['high']:.4f}]")
    if nearest_ob_bear:
        summary_parts.append(f"OB baissier: [{nearest_ob_bear['low']:.4f} – {nearest_ob_bear['high']:.4f}]")
    if nearest_fvg_bull:
        summary_parts.append(f"FVG haussier: [{nearest_fvg_bull['low']:.4f} – {nearest_fvg_bull['high']:.4f}]")
    if nearest_fvg_bear:
        summary_parts.append(f"FVG baissier: [{nearest_fvg_bear['low']:.4f} – {nearest_fvg_bear['high']:.4f}]")
    summary_parts.append(f"Recommandation: {recommendation} — {reason}")

    summary = "\n".join(p for p in summary_parts if p)

    return MarketDiagnosis(
        instrument=instrument,
        trend_h4=trend_h4,
        trend_h1=trend_h1,
        current_price=current_price,
        last_bos=last_bos,
        last_choch=last_choch,
        bos_count_recent=bos_count,
        is_retournement_in_progress=is_retournement,
        is_range=is_range,
        nearest_ob_bull=nearest_ob_bull,
        nearest_ob_bear=nearest_ob_bear,
        nearest_fvg_bull=nearest_fvg_bull,
        nearest_fvg_bear=nearest_fvg_bear,
        unmitigated_bullish_targets=unmitigated_bull,
        unmitigated_bearish_targets=unmitigated_bear,
        detected_scenario=detected_scenario,
        scenario_details=scenario_details,
        major_high=major_high,
        major_low=major_low,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        recommendation=recommendation,
        recommendation_reason=reason,
        summary=summary,
    )
