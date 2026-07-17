"""
technical_analysis.py
=====================
Moteur d'analyse technique pour le bot Boom & Crash v2 — Version solidifiée.

Solidifications apportées :
  [S1] detect_mini_choch() : détecteur de mini-CHoCH local sur M15
       Signature enrichie avec zone_low/zone_high + volume 1.5×
  [S2] Volume multiplier 1.5× documenté et appliqué par défaut
       dans check_confirmation_candle
  [S3] detect_false_breakout() : fausse cassure avec volume pour Scénario 2
  [S4] detect_choch() : utiliser trend_h1.direction (pas le biais fixe)
       → correction documentée, appel corrigé dans scenario_engine
  [S5] FairValueGap.mitigated : filtre des FVG déjà touchés
       find_fair_value_gaps() marque chaque FVG si comblé
       find_fvg_nearest_price() exclut les mitigés
  [S6] Précision du prix d'entrée pour le rejet (documenté)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("technical_analysis")

Candle = dict  # {epoch, time, open, high, low, close, volume}


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ConfirmationType(str, Enum):
    REJECTION_WICK = "REJECTION_WICK"
    ENGULFING      = "ENGULFING"
    MINI_CHOCH     = "MINI_CHOCH"     # [S1] nouveau type


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str        # "high" ou "low"
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
    mitigated: bool = False      # [S5] True si le prix a déjà traversé la zone


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
    volume_ratio: float          # volume bougie / volume moyen N périodes


@dataclass
class TrendState:
    direction: Direction
    ema20_last: float
    ema50_last: float
    is_range: bool


@dataclass
class FalseBreakout:             # [S3] nouveau dataclass
    level_broken: float          # niveau du swing cassé
    wick_extreme: float          # plus haut (ou plus bas) de la mèche
    close_price: float           # clôture en deçà du niveau
    volume_ratio: float          # volume / moyenne 20
    wick_ratio: float            # mèche / range total
    valid: bool


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
    """
    Tendance via EMA fast vs EMA slow.
    is_range = True si écart < 0.05% du prix.
    """
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

def find_swing_points(
    candles: list[Candle],
    window: int = 3,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Swing highs / lows par pivot (window bougies de chaque côté).
    Pour M15/M5, utiliser window=2 pour plus de sensibilité.
    """
    highs: list[SwingPoint] = []
    lows:  list[SwingPoint] = []

    for i in range(window, len(candles) - window):
        c = candles[i]
        neighbors = candles[i - window: i + window + 1]

        if c["high"] == max(n["high"] for n in neighbors):
            highs.append(SwingPoint(i, c["high"], "high", c))

        if c["low"] == min(n["low"] for n in neighbors):
            lows.append(SwingPoint(i, c["low"], "low", c))

    return highs, lows


# ─────────────────────────────────────────────────────────────────────────────
# 3. BOS
# ─────────────────────────────────────────────────────────────────────────────

def detect_bos(
    candles: list[Candle],
    swing_window: int = 3,
    min_candles_since_swing: int = 2,
) -> Optional[BosSignal]:
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHoCH  [S4 — utiliser trend_h1.direction, pas le biais fixe]
# ─────────────────────────────────────────────────────────────────────────────

def detect_choch(
    candles: list[Candle],
    current_trend: Direction,          # [S4] passer trend_h1.direction ici
    swing_window: int = 3,
) -> Optional[ChochSignal]:
    """
    Détecte un CHoCH : cassure d'un swing dans le sens OPPOSÉ
    à la tendance RÉELLE sur H1.

    [S4] IMPORTANT : l'appelant doit passer trend_h1.direction (tendance
    réelle calculée via detect_trend), PAS le biais fixe de l'instrument.
    Exemple correct dans scenario_engine :
        choch = detect_choch(h1, trend_h1.direction)
    """
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
# 5. Order Blocks
# ─────────────────────────────────────────────────────────────────────────────

def find_order_blocks(
    candles: list[Candle],
    direction: Direction,
    lookback: int = 50,
    min_impulse_candles: int = 2,
) -> list[OrderBlock]:
    obs: list[OrderBlock] = []
    current_price = candles[-1]["close"]
    search_range = candles[-lookback:] if len(candles) > lookback else candles
    offset = max(0, len(candles) - lookback)

    for i in range(len(search_range) - min_impulse_candles - 1):
        c = search_range[i]
        is_bearish = c["close"] < c["open"]
        is_bullish = c["close"] > c["open"]

        if direction == Direction.BULLISH and is_bearish:
            following = search_range[i + 1: i + 1 + min_impulse_candles]
            if len(following) < min_impulse_candles:
                continue
            if all(f["close"] > f["open"] for f in following):
                mid = (c["high"] + c["low"]) / 2
                mitigated = current_price < mid
                obs.append(OrderBlock(
                    direction=Direction.BULLISH,
                    high=c["high"], low=c["low"], mid=mid,
                    candle_index=offset + i, candle=c,
                    mitigated=mitigated,
                ))

        elif direction == Direction.BEARISH and is_bullish:
            following = search_range[i + 1: i + 1 + min_impulse_candles]
            if len(following) < min_impulse_candles:
                continue
            if all(f["close"] < f["open"] for f in following):
                mid = (c["high"] + c["low"]) / 2
                mitigated = current_price > mid
                obs.append(OrderBlock(
                    direction=Direction.BEARISH,
                    high=c["high"], low=c["low"], mid=mid,
                    candle_index=offset + i, candle=c,
                    mitigated=mitigated,
                ))

    return [ob for ob in reversed(obs) if not ob.mitigated]


def find_ob_nearest_price(
    candles: list[Candle],
    direction: Direction,
    current_price: float,
) -> Optional[OrderBlock]:
    obs = find_order_blocks(candles, direction)
    if not obs:
        return None
    obs.sort(key=lambda ob: abs(ob.mid - current_price))
    return obs[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fair Value Gaps  [S5 — filtre des FVG déjà comblés]
# ─────────────────────────────────────────────────────────────────────────────

def find_fair_value_gaps(
    candles: list[Candle],
    lookback: int = 50,
) -> list[FairValueGap]:
    """
    [S5] Chaque FVG est marqué mitigated=True si le prix a déjà traversé
    sa zone depuis sa formation. Un FVG mitigé perd son rôle d'aimant.

    Logique de mitigation :
    - FVG haussier [low, high] : mitigé si une bougie ultérieure a un low
      inférieur au mid du FVG (le prix est descendu dans la zone).
    - FVG baissier [low, high] : mitigé si une bougie ultérieure a un high
      supérieur au mid du FVG.
    """
    fvgs: list[FairValueGap] = []
    search = candles[-lookback:] if len(candles) > lookback else candles
    offset = max(0, len(candles) - lookback)
    n = len(search)

    for i in range(2, n):
        c1, c3 = search[i - 2], search[i]

        if c1["high"] < c3["low"]:
            high = c3["low"]
            low  = c1["high"]
            mid  = (high + low) / 2
            subsequent = search[i + 1:]
            mitigated = any(c["low"] <= mid for c in subsequent)
            fvgs.append(FairValueGap(
                direction=Direction.BULLISH,
                high=high, low=low, mid=mid,
                candle_index=offset + i,
                mitigated=mitigated,
            ))

        elif c1["low"] > c3["high"]:
            high = c1["low"]
            low  = c3["high"]
            mid  = (high + low) / 2
            subsequent = search[i + 1:]
            mitigated = any(c["high"] >= mid for c in subsequent)
            fvgs.append(FairValueGap(
                direction=Direction.BEARISH,
                high=high, low=low, mid=mid,
                candle_index=offset + i,
                mitigated=mitigated,
            ))

    return list(reversed(fvgs))


def find_fvg_nearest_price(
    candles: list[Candle],
    direction: Direction,
    current_price: float,
) -> Optional[FairValueGap]:
    """[S5] Retourne le FVG non mitigé le plus proche du prix actuel."""
    fvgs = [
        fvg for fvg in find_fair_value_gaps(candles)
        if fvg.direction == direction and not fvg.mitigated
    ]
    if not fvgs:
        return None
    fvgs.sort(key=lambda fvg: abs(fvg.mid - current_price))
    return fvgs[0]


def price_in_zone(price: float, zone_low: float, zone_high: float) -> bool:
    return zone_low <= price <= zone_high


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fibonacci
# ─────────────────────────────────────────────────────────────────────────────

def compute_fibonacci(
    swing_high: float,
    swing_low: float,
    trend: Direction,
) -> Optional[FibLevels]:
    if swing_high <= swing_low:
        return None
    diff = swing_high - swing_low

    if trend == Direction.BULLISH:
        return FibLevels(
            swing_high=swing_high, swing_low=swing_low, trend=trend,
            fib_236=swing_high - diff * 0.236,
            fib_382=swing_high - diff * 0.382,
            fib_500=swing_high - diff * 0.500,
            fib_618=swing_high - diff * 0.618,
            fib_786=swing_high - diff * 0.786,
        )
    else:
        return FibLevels(
            swing_high=swing_high, swing_low=swing_low, trend=trend,
            fib_236=swing_low + diff * 0.236,
            fib_382=swing_low + diff * 0.382,
            fib_500=swing_low + diff * 0.500,
            fib_618=swing_low + diff * 0.618,
            fib_786=swing_low + diff * 0.786,
        )


def price_near_fib_level(
    price: float,
    fib: FibLevels,
    tolerance_pct: float = 0.15,
) -> Optional[str]:
    range_size = fib.swing_high - fib.swing_low
    tolerance  = range_size * (tolerance_pct / 100)
    levels = {
        "fib_236": fib.fib_236, "fib_382": fib.fib_382,
        "fib_500": fib.fib_500, "fib_618": fib.fib_618,
        "fib_786": fib.fib_786,
    }
    for name, level in levels.items():
        if abs(price - level) <= tolerance:
            return name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Confirmation de bougie  [S2 — volume 1.5× par défaut]
# ─────────────────────────────────────────────────────────────────────────────

def check_confirmation_candle(
    candles: list[Candle],
    expected_direction: Direction,
    wick_threshold: float = 0.60,
    body_threshold: float = 0.50,
    volume_multiplier: float = 1.5,     # [S2] 1.5× par défaut (était 1.0)
    volume_period: int = 20,
) -> BougiConfirmation:
    """
    Vérifie si la dernière bougie est une bougie de confirmation.

    [S2] volume_multiplier est maintenant 1.5 par défaut pour filtrer
    uniquement les mouvements avec participation institutionnelle.

    Règle 1 — Rejection Wick :
      mèche opposée ≥ wick_threshold du range
      volume ≥ volume_multiplier × moyenne volume_period bougies

    Règle 2 — Engulfing :
      corps ≥ body_threshold du range
      volume ≥ volume_multiplier × moyenne

    [S6] Prix d'entrée :
      BULLISH : entry = c['high']  (entrée si le prix casse le high)
      BEARISH : entry = c['low']   (entrée si le prix casse le low)
    """
    if len(candles) < volume_period + 1:
        return BougiConfirmation(
            valid=False, kind=ConfirmationType.REJECTION_WICK,
            entry_candle=candles[-1], entry_price=candles[-1]["close"],
            wick_ratio=0, body_ratio=0, volume_ratio=0,
        )

    c = candles[-1]
    total_range = c["high"] - c["low"]
    if total_range == 0:
        return BougiConfirmation(
            valid=False, kind=ConfirmationType.REJECTION_WICK,
            entry_candle=c, entry_price=c["close"],
            wick_ratio=0, body_ratio=0, volume_ratio=0,
        )

    body   = abs(c["close"] - c["open"])
    body_r = body / total_range
    is_bull = c["close"] > c["open"]
    is_bear = c["close"] < c["open"]

    lower_wick = min(c["open"], c["close"]) - c["low"]
    upper_wick = c["high"] - max(c["open"], c["close"])
    lower_r = lower_wick / total_range
    upper_r = upper_wick / total_range

    avg_vol = sum(cx["volume"] for cx in candles[-(volume_period + 1):-1]) / volume_period
    vol_r   = (c["volume"] / avg_vol) if avg_vol > 0 else 0
    vol_ok  = vol_r >= volume_multiplier

    # Règle 1 : Rejection Wick
    if expected_direction == Direction.BULLISH and lower_r >= wick_threshold and vol_ok:
        return BougiConfirmation(
            valid=True, kind=ConfirmationType.REJECTION_WICK,
            entry_candle=c, entry_price=c["high"],   # [S6]
            wick_ratio=lower_r, body_ratio=body_r, volume_ratio=vol_r,
        )
    if expected_direction == Direction.BEARISH and upper_r >= wick_threshold and vol_ok:
        return BougiConfirmation(
            valid=True, kind=ConfirmationType.REJECTION_WICK,
            entry_candle=c, entry_price=c["low"],    # [S6]
            wick_ratio=upper_r, body_ratio=body_r, volume_ratio=vol_r,
        )

    # Règle 2 : Engulfing
    if expected_direction == Direction.BULLISH and is_bull and body_r >= body_threshold and vol_ok:
        return BougiConfirmation(
            valid=True, kind=ConfirmationType.ENGULFING,
            entry_candle=c, entry_price=c["close"],
            wick_ratio=lower_r, body_ratio=body_r, volume_ratio=vol_r,
        )
    if expected_direction == Direction.BEARISH and is_bear and body_r >= body_threshold and vol_ok:
        return BougiConfirmation(
            valid=True, kind=ConfirmationType.ENGULFING,
            entry_candle=c, entry_price=c["close"],
            wick_ratio=upper_r, body_ratio=body_r, volume_ratio=vol_r,
        )

    return BougiConfirmation(
        valid=False, kind=ConfirmationType.REJECTION_WICK,
        entry_candle=c, entry_price=c["close"],
        wick_ratio=max(lower_r, upper_r), body_ratio=body_r, volume_ratio=vol_r,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Mini-CHoCH sur M15  [S1 — nouvelle fonction]
# ─────────────────────────────────────────────────────────────────────────────

def detect_mini_choch(
    candles: list[Candle],
    expected_direction: Direction,
    zone_low: Optional[float] = None,
    zone_high: Optional[float] = None,
    volume_multiplier: float = 1.5,
    volume_period: int = 20,
) -> Optional[BougiConfirmation]:
    """
    [S1] Détecte un mini-CHoCH local sur M15 (ou toute timeframe fournie).

    Pour un trade BULLISH (Crash) :
      1. Identifie un swing low récent (stop hunt sous la zone)
      2. Cherche le swing high qui précède ce swing low (≤15 bougies)
      3. Si la bougie courante clôture AU-DESSUS de ce swing high
         → mini-CHoCH haussier confirmé

    Pour un trade BEARISH (Boom) :
      1. Identifie un swing high récent (stop hunt au-dessus de la zone)
      2. Cherche le swing low qui précède ce swing high
      3. Si la bougie courante clôture EN-DESSOUS de ce swing low
         → mini-CHoCH baissier confirmé

    Volume ≥ 1.5× moyenne si disponible (renforce la confirmation).
    Retourne None si aucun mini-CHoCH détecté.
    """
    if len(candles) < 12:
        return None

    window = candles[-20:]   # fenêtre d'analyse : 20 dernières bougies M15
    highs, lows = find_swing_points(window, window=2)

    avg_vol = (
        sum(c["volume"] for c in candles[-volume_period:]) / volume_period
        if len(candles) >= volume_period else 0
    )
    last = window[-1]
    vol_r = (last["volume"] / avg_vol) if avg_vol > 0 else 0

    if expected_direction == Direction.BULLISH:
        if not lows:
            return None
        last_sl = lows[-1]
        if zone_low is not None and last_sl.price > zone_high:
            return None
        prior_highs = [sh for sh in highs if sh.index < last_sl.index]
        if not prior_highs:
            return None
        prior_sh = prior_highs[-1]
        if last["close"] > prior_sh.price:
            vol_ok = vol_r >= volume_multiplier
            logger.debug(
                f"Mini-CHoCH BULLISH: SL={last_sl.price:.4f} → "
                f"close {last['close']:.4f} > prior_SH {prior_sh.price:.4f} "
                f"vol={vol_r:.1f}×"
            )
            return BougiConfirmation(
                valid=True,
                kind=ConfirmationType.MINI_CHOCH,
                entry_candle=last,
                entry_price=prior_sh.price,
                wick_ratio=0.0,
                body_ratio=abs(last["close"] - last["open"]) / (last["high"] - last["low"] or 1),
                volume_ratio=vol_r,
            )

    elif expected_direction == Direction.BEARISH:
        if not highs:
            return None
        last_sh = highs[-1]
        if zone_high is not None and last_sh.price < zone_low:
            return None
        prior_lows = [sl for sl in lows if sl.index < last_sh.index]
        if not prior_lows:
            return None
        prior_sl = prior_lows[-1]
        if last["close"] < prior_sl.price:
            vol_r_val = vol_r
            logger.debug(
                f"Mini-CHoCH BEARISH: SH={last_sh.price:.4f} → "
                f"close {last['close']:.4f} < prior_SL {prior_sl.price:.4f} "
                f"vol={vol_r_val:.1f}×"
            )
            return BougiConfirmation(
                valid=True,
                kind=ConfirmationType.MINI_CHOCH,
                entry_candle=last,
                entry_price=prior_sl.price,
                wick_ratio=0.0,
                body_ratio=abs(last["close"] - last["open"]) / (last["high"] - last["low"] or 1),
                volume_ratio=vol_r,
            )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Détecteur de fausse cassure  [S3 — nouvelle fonction]
# ─────────────────────────────────────────────────────────────────────────────

def detect_false_breakout(
    candles: list[Candle],
    level: float,
    direction_of_breakout: Direction,
    volume_multiplier: float = 1.5,
    volume_period: int = 20,
    wick_ratio_threshold: float = 0.40,
) -> Optional[FalseBreakout]:
    """
    [S3] Détecte une fausse cassure (stop hunt) sur la timeframe fournie.

    Cas BULLISH (fausse cassure haussière → signal baissier) :
      - Une bougie dépasse `level` vers le haut (high > level)
      - Mais clôture EN DESSOUS de level
      - La mèche haute ≥ wick_ratio_threshold du range total
      - Volume ≥ volume_multiplier × moyenne volume_period

    Cas BEARISH (fausse cassure baissière → signal haussier) :
      - Une bougie dépasse `level` vers le bas (low < level)
      - Mais clôture AU-DESSUS de level
      - La mèche basse ≥ wick_ratio_threshold
      - Volume ≥ volume_multiplier × moyenne

    Analyse les 5 dernières bougies pour détecter la fausse cassure la
    plus récente. Retourne None si aucune trouvée.
    """
    if len(candles) < volume_period + 2:
        return None

    avg_vol = sum(c["volume"] for c in candles[-volume_period:]) / volume_period

    recent = candles[-5:]

    for c in reversed(recent):
        total_range = c["high"] - c["low"]
        if total_range == 0:
            continue
        vol_r = c["volume"] / avg_vol if avg_vol > 0 else 0
        vol_ok = vol_r >= volume_multiplier

        if direction_of_breakout == Direction.BULLISH:
            upper_wick = c["high"] - max(c["open"], c["close"])
            wick_r = upper_wick / total_range
            if (c["high"] > level and
                    c["close"] < level and
                    wick_r >= wick_ratio_threshold and
                    vol_ok):
                return FalseBreakout(
                    level_broken=level,
                    wick_extreme=c["high"],
                    close_price=c["close"],
                    volume_ratio=vol_r,
                    wick_ratio=wick_r,
                    valid=True,
                )

        elif direction_of_breakout == Direction.BEARISH:
            lower_wick = min(c["open"], c["close"]) - c["low"]
            wick_r = lower_wick / total_range
            if (c["low"] < level and
                    c["close"] > level and
                    wick_r >= wick_ratio_threshold and
                    vol_ok):
                return FalseBreakout(
                    level_broken=level,
                    wick_extreme=c["low"],
                    close_price=c["close"],
                    volume_ratio=vol_r,
                    wick_ratio=wick_r,
                    valid=True,
                )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cartographie HTF H4
# ─────────────────────────────────────────────────────────────────────────────

def find_htf_targets(
    h4_candles: list[Candle],
    direction: Direction,
    current_price: float,
    num_targets: int = 3,
) -> list[float]:
    """
    Aimants de liquidité H4 pour TP3.
    Sources : OB H4, FVG H4 non mitigés, EQH/EQL H4.
    """
    targets: list[float] = []

    obs = find_order_blocks(h4_candles, direction, lookback=100)
    for ob in obs:
        if direction == Direction.BEARISH and ob.mid < current_price:
            targets.append(ob.mid)
        elif direction == Direction.BULLISH and ob.mid > current_price:
            targets.append(ob.mid)

    fvgs = find_fair_value_gaps(h4_candles, lookback=100)
    for fvg in fvgs:
        if fvg.mitigated:
            continue
        if fvg.direction == direction:
            if direction == Direction.BEARISH and fvg.mid < current_price:
                targets.append(fvg.mid)
            elif direction == Direction.BULLISH and fvg.mid > current_price:
                targets.append(fvg.mid)

    highs, lows = find_swing_points(h4_candles, window=3)

    if direction == Direction.BEARISH:
        for i in range(len(lows) - 1):
            l1, l2 = lows[i].price, lows[i + 1].price
            if l2 < current_price and abs(l1 - l2) / l1 < 0.0015:
                targets.append(min(l1, l2))
    else:
        for i in range(len(highs) - 1):
            h1v, h2v = highs[i].price, highs[i + 1].price
            if h2v > current_price and abs(h1v - h2v) / h1v < 0.0015:
                targets.append(max(h1v, h2v))

    targets = sorted(set(round(t, 4) for t in targets),
                     key=lambda t: abs(t - current_price))
    return targets[:num_targets]


# ─────────────────────────────────────────────────────────────────────────────
# 12. Calcul SL / TP
# ─────────────────────────────────────────────────────────────────────────────

def compute_sl_tp(
    direction: Direction,
    entry_price: float,
    sl_ref: float,
    h4_targets: list[float],
    m15_candles: list[Candle],
    buffer_pct: float = 0.002,
) -> tuple[float, float, float, float]:
    """
    SL  : au-delà du swing point de référence + buffer 0.2%.
    TP1 : prochain swing M15 dans la direction du trade.
    TP2 : 2.5× le risque (si aucune cible structurelle H1).
    TP3 : aimant de liquidité H4 (find_htf_targets).
    """
    buffer = abs(entry_price) * buffer_pct

    if direction == Direction.BEARISH:
        sl   = sl_ref + buffer
        risk = sl - entry_price
        if risk <= 0:
            risk = abs(entry_price) * 0.01

        _, lows = find_swing_points(m15_candles, window=2)
        tp1_candidates = [sl_pt.price for sl_pt in lows if sl_pt.price < entry_price]
        tp1 = tp1_candidates[-1] if tp1_candidates else entry_price - risk * 1.5
        tp2 = entry_price - risk * 2.5
        tp3 = h4_targets[0] if h4_targets else entry_price - risk * 5.0

    else:
        sl   = sl_ref - buffer
        risk = entry_price - sl
        if risk <= 0:
            risk = abs(entry_price) * 0.01

        highs, _ = find_swing_points(m15_candles, window=2)
        tp1_candidates = [sh.price for sh in highs if sh.price > entry_price]
        tp1 = tp1_candidates[0] if tp1_candidates else entry_price + risk * 1.5
        tp2 = entry_price + risk * 2.5
        tp3 = h4_targets[0] if h4_targets else entry_price + risk * 5.0

    return round(sl, 4), round(tp1, 4), round(tp2, 4), round(tp3, 4)
