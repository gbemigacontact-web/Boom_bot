"""
technical_analysis.py
=====================
Moteur d'analyse technique pour le bot Boom & Crash v2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

logger = logging.getLogger("technical_analysis")

Candle = dict

# ─────────────────────────────────────────────────────────────────────────────
# Enums / Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class ConfirmationType(str, Enum):
    REJECTION_WICK = "REJECTION_WICK"
    ENGULFING = "ENGULFING"
    MINI_CHOCH = "MINI_CHOCH"

@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str
    candle: Candle

@dataclass
class BosSignal:
    direction: Direction
    broken_level: float
    bos_candle: Candle
    bos_index: int
    prior_swing: SwingPoint

@dataclass
class ChochSignal:
    direction: Direction
    broken_level: float
    choch_candle: Candle
    choch_index: int

@dataclass
class OrderBlock:
    direction: Direction
    high: float
    low: float
    mid: float
    candle_index: int
    candle: Candle
    mitigated: bool = False

@dataclass
class FairValueGap:
    direction: Direction
    high: float
    low: float
    mid: float
    candle_index: int
    mitigated: bool = False

@dataclass
class FibLevels:
    swing_high: float
    swing_low: float
    trend: Direction
    fib_236: float
    fib_382: float
    fib_500: float
    fib_618: float
    fib_786: float

@dataclass
class BougiConfirmation:
    valid: bool
    kind: ConfirmationType
    entry_candle: Candle
    entry_price: float
    wick_ratio: float
    body_ratio: float
    volume_ratio: float

@dataclass
class TrendState:
    direction: Direction
    ema20_last: float
    ema50_last: float
    is_range: bool


# ─────────────────────────────────────────────────────────────────────────────
# 1. Indicateurs de base
# ─────────────────────────────────────────────────────────────────────────────

def compute_ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema

def detect_trend(candles: list[Candle], fast: int = 20, slow: int = 50) -> TrendState:
    closes = [c["close"] for c in candles]
    ema20 = compute_ema(closes, fast)
    ema50 = compute_ema(closes, slow)
    if not ema20 or not ema50:
        return TrendState(Direction.NEUTRAL, 0, 0, True)
    e20, e50 = ema20[-1], ema50[-1]
    gap_pct = abs(e20 - e50) / e50 if e50 else 0
    is_range = gap_pct < 0.0005
    if is_range:
        direction = Direction.NEUTRAL
    elif e20 > e50:
        direction = Direction.BULLISH
    else:
        direction = Direction.BEARISH
    return TrendState(direction, e20, e50, is_range)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Swing Points
# ─────────────────────────────────────────────────────────────────────────────

def find_swing_points(candles: list[Candle], window: int = 3) -> tuple[list[SwingPoint], list[SwingPoint]]:
    highs, lows = [], []
    for i in range(window, len(candles) - window):
        c = candles[i]
        neighbors = candles[i - window : i + window + 1]
        if c["high"] == max(n["high"] for n in neighbors):
            highs.append(SwingPoint(i, c["high"], "high", c))
        if c["low"] == min(n["low"] for n in neighbors):
            lows.append(SwingPoint(i, c["low"], "low", c))
    return highs, lows


# ─────────────────────────────────────────────────────────────────────────────
# 3. BOS / CHoCH
# ─────────────────────────────────────────────────────────────────────────────

def detect_bos(candles: list[Candle], swing_window: int = 3, min_candles_since_swing: int = 2) -> Optional[BosSignal]:
    highs, lows = find_swing_points(candles, window=swing_window)
    last = candles[-1]
    last_idx = len(candles) - 1

    for sh in reversed(highs):
        if last_idx - sh.index < min_candles_since_swing:
            continue
        if last["close"] > sh.price:
            return BosSignal(Direction.BULLISH, sh.price, last, last_idx, sh)
        break

    for sl in reversed(lows):
        if last_idx - sl.index < min_candles_since_swing:
            continue
        if last["close"] < sl.price:
            return BosSignal(Direction.BEARISH, sl.price, last, last_idx, sl)
        break
    return None

def detect_choch(candles: list[Candle], current_trend: Direction, swing_window: int = 3) -> Optional[ChochSignal]:
    if current_trend == Direction.NEUTRAL:
        return None
    highs, lows = find_swing_points(candles, window=swing_window)
    last = candles[-1]
    last_idx = len(candles) - 1

    if current_trend == Direction.BULLISH:
        for sl in reversed(lows):
            if last_idx - sl.index < 2:
                continue
            if last["close"] < sl.price:
                return ChochSignal(Direction.BEARISH, sl.price, last, last_idx)
            break
    elif current_trend == Direction.BEARISH:
        for sh in reversed(highs):
            if last_idx - sh.index < 2:
                continue
            if last["close"] > sh.price:
                return ChochSignal(Direction.BULLISH, sh.price, last, last_idx)
            break
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Order Blocks
# ─────────────────────────────────────────────────────────────────────────────

def find_order_blocks(candles: list[Candle], direction: Direction, lookback: int = 150, min_impulse_candles: int = 2) -> list[OrderBlock]:
    obs = []
    current_price = candles[-1]["close"]
    search_range = candles[-lookback:] if len(candles) > lookback else candles
    offset = max(0, len(candles) - lookback)

    for i in range(len(search_range) - min_impulse_candles - 1):
        c = search_range[i]
        is_bearish = c["close"] < c["open"]
        is_bullish = c["close"] > c["open"]

        if direction == Direction.BULLISH and is_bearish:
            following = search_range[i+1 : i+1+min_impulse_candles]
            if all(f["close"] > f["open"] for f in following):
                mid = (c["high"] + c["low"]) / 2
                mitigated = current_price < mid
                obs.append(OrderBlock(Direction.BULLISH, c["high"], c["low"], mid, offset+i, c, mitigated))
        elif direction == Direction.BEARISH and is_bullish:
            following = search_range[i+1 : i+1+min_impulse_candles]
            if all(f["close"] < f["open"] for f in following):
                mid = (c["high"] + c["low"]) / 2
                mitigated = current_price > mid
                obs.append(OrderBlock(Direction.BEARISH, c["high"], c["low"], mid, offset+i, c, mitigated))

    obs = [ob for ob in reversed(obs) if not ob.mitigated]
    return obs

def find_ob_nearest_price(candles: list[Candle], direction: Direction, current_price: float) -> Optional[OrderBlock]:
    obs = find_order_blocks(candles, direction)
    if not obs:
        return None
    obs.sort(key=lambda ob: abs(ob.mid - current_price))
    return obs[0]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fair Value Gaps
# ─────────────────────────────────────────────────────────────────────────────

def find_fair_value_gaps(candles: list[Candle], lookback: int = 150) -> list[FairValueGap]:
    fvgs = []
    search = candles[-lookback:] if len(candles) > lookback else candles
    offset = max(0, len(candles) - lookback)

    for i in range(2, len(search)):
        c1, c3 = search[i-2], search[i]
        if c1["high"] < c3["low"]:
            high, low = c3["low"], c1["high"]
            # mitigated si le prix est déjà passé au-dessus de la zone
            mitigated = candles[-1]["low"] <= high  # simplification
            fvgs.append(FairValueGap(Direction.BULLISH, high, low, (high+low)/2, offset+i, mitigated))
        elif c1["low"] > c3["high"]:
            high, low = c1["low"], c3["high"]
            mitigated = candles[-1]["high"] >= low
            fvgs.append(FairValueGap(Direction.BEARISH, high, low, (high+low)/2, offset+i, mitigated))
    return list(reversed(fvgs))

def find_fvg_nearest_price(candles: list[Candle], direction: Direction, current_price: float) -> Optional[FairValueGap]:
    fvgs = [fvg for fvg in find_fair_value_gaps(candles) if fvg.direction == direction and not fvg.mitigated]
    if not fvgs:
        return None
    fvgs.sort(key=lambda fvg: abs(fvg.mid - current_price))
    return fvgs[0]

def price_in_zone(price: float, zone_low: float, zone_high: float) -> bool:
    return zone_low <= price <= zone_high


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fibonacci
# ─────────────────────────────────────────────────────────────────────────────

def compute_fibonacci(swing_high: float, swing_low: float, trend: Direction) -> Optional[FibLevels]:
    if swing_high <= swing_low:
        return None
    diff = swing_high - swing_low
    if trend == Direction.BULLISH:
        return FibLevels(swing_high, swing_low, trend,
                         swing_high - diff*0.236, swing_high - diff*0.382,
                         swing_high - diff*0.5, swing_high - diff*0.618,
                         swing_high - diff*0.786)
    else:
        return FibLevels(swing_high, swing_low, trend,
                         swing_low + diff*0.236, swing_low + diff*0.382,
                         swing_low + diff*0.5, swing_low + diff*0.618,
                         swing_low + diff*0.786)

def price_near_fib_level(price: float, fib: FibLevels, tolerance_pct: float = 0.15) -> Optional[str]:
    range_size = fib.swing_high - fib.swing_low
    tolerance = range_size * (tolerance_pct / 100)
    levels = {"fib_382": fib.fib_382, "fib_500": fib.fib_500, "fib_618": fib.fib_618}
    for name, level in levels.items():
        if abs(price - level) <= tolerance:
            return name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Confirmation de bougie
# ─────────────────────────────────────────────────────────────────────────────

def check_confirmation_candle(
    candles: list[Candle],
    expected_direction: Direction,
    wick_threshold: float = 0.60,
    body_threshold: float = 0.50,
    volume_multiplier: float = 1.5,
    volume_period: int = 20,
) -> BougiConfirmation:
    if len(candles) < volume_period + 1:
        return BougiConfirmation(False, ConfirmationType.REJECTION_WICK, candles[-1], candles[-1]["close"], 0, 0, 0)

    c = candles[-1]
    total_range = c["high"] - c["low"]
    if total_range == 0:
        return BougiConfirmation(False, ConfirmationType.REJECTION_WICK, c, c["close"], 0, 0, 0)

    body = abs(c["close"] - c["open"])
    body_r = body / total_range
    is_bull = c["close"] > c["open"]
    lower_wick = min(c["open"], c["close"]) - c["low"]
    upper_wick = c["high"] - max(c["open"], c["close"])
    lower_r = lower_wick / total_range
    upper_r = upper_wick / total_range

    avg_vol = sum(cx.get("volume", 0) for cx in candles[-(volume_period+1):-1]) / volume_period
    vol_r = (c.get("volume", 0) / avg_vol) if avg_vol > 0 else 1.0
    vol_ok = vol_r >= volume_multiplier

    if expected_direction == Direction.BULLISH and lower_r >= wick_threshold and vol_ok:
        return BougiConfirmation(True, ConfirmationType.REJECTION_WICK, c, c["high"], lower_r, body_r, vol_r)
    if expected_direction == Direction.BEARISH and upper_r >= wick_threshold and vol_ok:
        return BougiConfirmation(True, ConfirmationType.REJECTION_WICK, c, c["low"], upper_r, body_r, vol_r)

    if expected_direction == Direction.BULLISH and is_bull and body_r >= body_threshold and vol_ok:
        return BougiConfirmation(True, ConfirmationType.ENGULFING, c, c["close"], lower_r, body_r, vol_r)
    if expected_direction == Direction.BEARISH and not is_bull and body_r >= body_threshold and vol_ok:
        return BougiConfirmation(True, ConfirmationType.ENGULFING, c, c["close"], upper_r, body_r, vol_r)

    return BougiConfirmation(False, ConfirmationType.REJECTION_WICK, c, c["close"], max(lower_r, upper_r), body_r, vol_r)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Mini-CHoCH M15
# ─────────────────────────────────────────────────────────────────────────────

def detect_mini_choch(candles: list[Candle], expected_direction: Direction, lookback: int = 30) -> Optional[BougiConfirmation]:
    """Détecte un petit changement de structure local sur M15."""
    if len(candles) < lookback:
        return None
    sub = candles[-lookback:]
    highs, lows = find_swing_points(sub, window=2)
    if not lows or not highs:
        return None

    if expected_direction == Direction.BULLISH:
        if len(lows) < 2 or len(highs) < 1:
            return None
        last_low = lows[-1]
        prev_highs = [h for h in highs if h.index < last_low.index]
        if not prev_highs:
            return None
        last_high_before_low = prev_highs[-1]
        # Chercher une cassure au-dessus de ce high après le low
        for i in range(last_low.index + 1, len(sub)):
            if sub[i]["close"] > last_high_before_low.price:
                # mini-CHoCH détecté
                return BougiConfirmation(True, ConfirmationType.MINI_CHOCH, sub[i], sub[i]["close"], 0, 0, 0)
    else:  # BEARISH
        if len(highs) < 2 or len(lows) < 1:
            return None
        last_high = highs[-1]
        prev_lows = [l for l in lows if l.index < last_high.index]
        if not prev_lows:
            return None
        last_low_before_high = prev_lows[-1]
        for i in range(last_high.index + 1, len(sub)):
            if sub[i]["close"] < last_low_before_high.price:
                return BougiConfirmation(True, ConfirmationType.MINI_CHOCH, sub[i], sub[i]["close"], 0, 0, 0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Fausse cassure
# ─────────────────────────────────────────────────────────────────────────────

def detect_false_breakout(
    candles: list[Candle],
    level: float,
    direction_of_breakout: Direction,
    volume_multiplier: float = 1.5,
    volume_period: int = 20,
    wick_ratio_threshold: float = 0.40,
) -> Optional[dict]:
    if len(candles) < volume_period + 1:
        return None
    c = candles[-1]
    avg_vol = sum(cx.get("volume", 0) for cx in candles[-(volume_period+1):-1]) / volume_period
    vol_r = c.get("volume", 0) / avg_vol if avg_vol > 0 else 0
    if vol_r < volume_multiplier:
        return None

    total_range = c["high"] - c["low"]
    if total_range == 0:
        return None

    if direction_of_breakout == Direction.BULLISH:
        if c["high"] > level and c["close"] < level:
            upper_wick = c["high"] - max(c["open"], c["close"])
            if upper_wick / total_range >= wick_ratio_threshold:
                return {"high": c["high"], "volume_ratio": vol_r}
    else:
        if c["low"] < level and c["close"] > level:
            lower_wick = min(c["open"], c["close"]) - c["low"]
            if lower_wick / total_range >= wick_ratio_threshold:
                return {"low": c["low"], "volume_ratio": vol_r}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Cartographie HTF H4 et SL/TP
# ─────────────────────────────────────────────────────────────────────────────

def find_htf_targets(h4_candles: list[Candle], direction: Direction, current_price: float, num_targets: int = 3) -> list[float]:
    targets = []
    obs = find_order_blocks(h4_candles, direction, lookback=200)
    for ob in obs:
        if direction == Direction.BEARISH and ob.mid < current_price:
            targets.append(ob.mid)
        elif direction == Direction.BULLISH and ob.mid > current_price:
            targets.append(ob.mid)

    fvgs = find_fair_value_gaps(h4_candles, lookback=200)
    for fvg in fvgs:
        if fvg.direction == direction:
            if direction == Direction.BEARISH and fvg.mid < current_price:
                targets.append(fvg.mid)
            elif direction == Direction.BULLISH and fvg.mid > current_price:
                targets.append(fvg.mid)

    highs, lows = find_swing_points(h4_candles, window=3)
    if direction == Direction.BEARISH:
        for i in range(len(lows)-1):
            l1, l2 = lows[i].price, lows[i+1].price
            if l2 < current_price and abs(l1-l2)/l1 < 0.0015:
                targets.append(min(l1, l2))
    else:
        for i in range(len(highs)-1):
            h1, h2 = highs[i].price, highs[i+1].price
            if h2 > current_price and abs(h1-h2)/h1 < 0.0015:
                targets.append(max(h1, h2))

    targets = sorted(set(round(t,4) for t in targets), key=lambda t: abs(t - current_price))
    return targets[:num_targets]

def compute_sl_tp(direction: Direction, entry_price: float, sl_ref: float,
                  h4_targets: list[float], m15_candles: list[Candle],
                  buffer_pct: float = 0.002) -> tuple[float, float, float, float]:
    buffer = abs(entry_price) * buffer_pct
    if direction == Direction.BEARISH:
        sl = sl_ref + buffer
        risk = sl - entry_price
        if risk <= 0: risk = abs(entry_price) * 0.01
        _, lows = find_swing_points(m15_candles, window=2)
        tp1_candidates = [sl.price for sl in lows if sl.price < entry_price]
        tp1 = tp1_candidates[-1] if tp1_candidates else entry_price - risk * 1.5
        tp2 = entry_price - risk * 2.5
        tp3 = h4_targets[0] if h4_targets else entry_price - risk * 5.0
    else:
        sl = sl_ref - buffer
        risk = entry_price - sl
        if risk <= 0: risk = abs(entry_price) * 0.01
        highs, _ = find_swing_points(m15_candles, window=2)
        tp1_candidates = [sh.price for sh in highs if sh.price > entry_price]
        tp1 = tp1_candidates[0] if tp1_candidates else entry_price + risk * 1.5
        tp2 = entry_price + risk * 2.5
        tp3 = h4_targets[0] if h4_targets else entry_price + risk * 5.0
    return round(sl,4), round(tp1,4), round(tp2,4), round(tp3,4)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cibles non mitigées (pour fausse cassure)
# ─────────────────────────────────────────────────────────────────────────────

def find_unmitigated_bullish_targets(candles: list[Candle], current_price: float) -> list[tuple]:
    targets = []
    obs = find_order_blocks(candles, Direction.BULLISH)
    for ob in obs:
        if ob.mid > current_price and not ob.mitigated:
            targets.append(('OB', ob.mid, ob.high))
    fvgs = find_fair_value_gaps(candles)
    for fvg in fvgs:
        if fvg.direction == Direction.BULLISH and fvg.low > current_price and not fvg.mitigated:
            targets.append(('FVG', fvg.mid, fvg.high))
    return sorted(targets, key=lambda x: x[1])

def find_unmitigated_bearish_targets(candles: list[Candle], current_price: float) -> list[tuple]:
    targets = []
    obs = find_order_blocks(candles, Direction.BEARISH)
    for ob in obs:
        if ob.mid < current_price and not ob.mitigated:
            targets.append(('OB', ob.mid, ob.low))
    fvgs = find_fair_value_gaps(candles)
    for fvg in fvgs:
        if fvg.direction == Direction.BEARISH and fvg.high < current_price and not fvg.mitigated:
            targets.append(('FVG', fvg.mid, fvg.low))
    return sorted(targets, key=lambda x: x[1], reverse=True)
