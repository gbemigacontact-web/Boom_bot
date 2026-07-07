"""
deriv_client.py
===============
Client WebSocket Deriv pour le bot Boom & Crash v2.

Récupère les candles OHLCV sur 5 timeframes simultanément pour
les 6 instruments Boom/Crash. Pas d'authentification requise pour
la lecture des synthetic indices (données publiques Deriv).

Architecture :
- Une seule connexion WebSocket par run
- Requêtes séquentielles avec pause entre chaque (rate-limit Deriv)
- Volume inclus dans chaque bougie (nécessaire pour le filtre de
  confirmation : volume > moyenne 20 périodes)
- Gestion des erreurs avec retry automatique (3 tentatives)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import websockets

logger = logging.getLogger("deriv_client")

# ── Configuration ─────────────────────────────────────────────────────────────

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"

# Symboles officiels Deriv pour les synthetic indices Boom & Crash
INSTRUMENTS: dict[str, str] = {
    "Boom 500":   "BOOM500",
    "Boom 900":   "BOOM900",
    "Boom 1000":  "BOOM1000",
    "Crash 500":  "CRASH500",
    "Crash 900":  "CRASH900",
    "Crash 1000": "CRASH1000",
}

# Granularité en secondes pour chaque timeframe
TIMEFRAME_GRANULARITY: dict[str, int] = {
    "M5":  300,
    "M15": 900,
    "M30": 1800,
    "H1":  3600,
    "H4":  14400,
}

# Nombre de bougies récupérées par timeframe.
# - H4/H1 : 150 bougies = ~25 jours → assez pour EMA 50, swing points HTF
# - M30/M15 : 200 bougies → structure intermédiaire + OB + FVG
# - M5 : 250 bougies → confirmation + volume moyen 20 périodes + Fibonacci
CANDLE_COUNT: dict[str, int] = {
    "H4":  150,
    "H1":  150,
    "M30": 200,
    "M15": 200,
    "M5":  250,
}

# Ordre de récupération : du plus grand au plus petit timeframe
TIMEFRAME_ORDER = ["H4", "H1", "M30", "M15", "M5"]

# Pause entre chaque requête (secondes) — respecte le rate-limit Deriv
REQUEST_DELAY = 0.4

# Nombre de tentatives en cas d'erreur réseau
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Type d'une bougie
Candle = dict  # {time, open, high, low, close, volume, epoch}

# Type des données d'un instrument : timeframe → liste de bougies
InstrumentData = dict[str, list[Candle]]


# ─────────────────────────────────────────────────────────────────────────────
# Client WebSocket
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:
    """
    Client WebSocket Deriv sans authentification.
    Utilise ticks_history avec style='candles' pour récupérer
    les données OHLC + volume sur chaque timeframe.
    """

    def __init__(self, app_id: str = "1089"):
        self.app_id = app_id
        self.ws = None
        self._connected = False

    # ── Connexion ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        url = DERIV_WS_URL.format(app_id=self.app_id)
        self.ws = await websockets.connect(
            url,
            ping_interval=25,
            ping_timeout=15,
            close_timeout=10,
        )
        self._connected = True
        logger.info(f"Connecté à Deriv WebSocket (app_id={self.app_id})")

    async def close(self) -> None:
        if self.ws and self._connected:
            await self.ws.close()
            self._connected = False
            logger.info("Connexion Deriv fermée.")

    # ── Requête de bougies ────────────────────────────────────────────────────

    async def _fetch_candles_raw(
        self, symbol: str, granularity: int, count: int
    ) -> list[dict]:
        """
        Envoie une requête ticks_history et retourne les bougies brutes.
        Lève une RuntimeError si l'API retourne une erreur.
        """
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": granularity,
        }
        await self.ws.send(json.dumps(request))
        response = json.loads(await self.ws.recv())

        if "error" in response:
            raise RuntimeError(
                f"API Deriv error [{symbol} g={granularity}]: "
                f"{response['error']['message']}"
            )
        return response.get("candles", [])

    async def get_candles(
        self, symbol: str, timeframe: str
    ) -> list[Candle]:
        """
        Récupère et parse les bougies pour un symbole + timeframe.
        Retourne une liste de Candle triée chronologiquement (la plus
        ancienne en premier, la plus récente en dernier).

        Chaque Candle contient :
          epoch  : timestamp Unix (int)
          time   : datetime UTC
          open   : float
          high   : float
          low    : float
          close  : float
          volume : float (volume de la bougie, proxy pour l'absorption)
        """
        granularity = TIMEFRAME_GRANULARITY[timeframe]
        count = CANDLE_COUNT[timeframe]

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = await self._fetch_candles_raw(symbol, granularity, count)
                candles = [
                    {
                        "epoch":  int(c["epoch"]),
                        "time":   datetime.fromtimestamp(
                            int(c["epoch"]), tz=timezone.utc
                        ),
                        "open":   float(c["open"]),
                        "high":   float(c["high"]),
                        "low":    float(c["low"]),
                        "close":  float(c["close"]),
                        # Deriv ne fournit pas de volume sur les synthetic
                        # indices. On calcule un proxy basé sur le range de
                        # la bougie multiplié par 1000 (relatif, suffit pour
                        # comparer les bougies entre elles sur 20 périodes).
                        "volume": round(
                            (float(c["high"]) - float(c["low"])) * 1000, 2
                        ),
                    }
                    for c in raw
                ]
                logger.debug(
                    f"{symbol} {timeframe}: {len(candles)} bougies récupérées "
                    f"(dernière: {candles[-1]['time'].strftime('%H:%M') if candles else 'N/A'})"
                )
                return candles

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"{symbol} {timeframe} tentative {attempt}/{MAX_RETRIES} "
                        f"échouée: {e} — retry dans {RETRY_DELAY}s"
                    )
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error(
                        f"{symbol} {timeframe}: échec après {MAX_RETRIES} "
                        f"tentatives — {e}"
                    )

        return []  # retourne liste vide si toutes les tentatives échouent

    async def get_all_timeframes(self, symbol: str) -> InstrumentData:
        """
        Récupère les 5 timeframes pour un symbole dans l'ordre H4→M5.
        Retourne un dict {timeframe: [candles]}.
        """
        result: InstrumentData = {}
        for tf in TIMEFRAME_ORDER:
            result[tf] = await self.get_candles(symbol, tf)
            await asyncio.sleep(REQUEST_DELAY)
        return result

    # ── Validation des données ─────────────────────────────────────────────────

    def validate_instrument_data(
        self, instrument: str, data: InstrumentData
    ) -> bool:
        """
        Vérifie que les données récupérées sont suffisantes pour l'analyse.
        Retourne False si un timeframe critique est vide ou insuffisant.
        """
        minimums = {"H4": 60, "H1": 60, "M30": 80, "M15": 80, "M5": 30}
        for tf, min_count in minimums.items():
            candles = data.get(tf, [])
            if len(candles) < min_count:
                logger.warning(
                    f"{instrument} {tf}: seulement {len(candles)} bougies "
                    f"(min requis: {min_count}) — données insuffisantes"
                )
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires sur les bougies
# ─────────────────────────────────────────────────────────────────────────────

def get_current_price(data: InstrumentData) -> float:
    """Retourne le prix de clôture de la dernière bougie M5."""
    m5 = data.get("M5", [])
    if not m5:
        raise ValueError("Pas de données M5 disponibles")
    return m5[-1]["close"]


def is_spread_abnormal(data: InstrumentData, multiplier: float = 4.0) -> bool:
    """
    Détecte si le spread est anormalement élevé sur les 5 dernières
    bougies M5 (proxy : range moyen des 5 dernières vs les 50 précédentes).
    Un spread anormal indique une période de news ou de faible liquidité
    → le bot doit s'abstenir.
    """
    m5 = data.get("M5", [])
    if len(m5) < 55:
        return False

    recent = m5[-5:]
    baseline = m5[-55:-5]

    recent_avg_range = sum(c["high"] - c["low"] for c in recent) / len(recent)
    baseline_avg_range = sum(c["high"] - c["low"] for c in baseline) / len(baseline)

    if baseline_avg_range == 0:
        return False

    ratio = recent_avg_range / baseline_avg_range
    if ratio > multiplier:
        logger.warning(
            f"Spread anormal détecté: range récent {ratio:.1f}× la normale"
        )
        return True
    return False


def compute_volume_average(candles: list[Candle], period: int = 20) -> float:
    """Calcule la moyenne du volume sur les `period` dernières bougies."""
    if len(candles) < period:
        return 0.0
    return sum(c["volume"] for c in candles[-period:]) / period


def candle_body_ratio(candle: Candle) -> float:
    """
    Ratio corps/range total (0 à 1).
    Mesure la force directionnelle de la bougie.
    > 0.6 = bougie forte (confirmation valide selon notre stratégie).
    """
    total_range = candle["high"] - candle["low"]
    if total_range == 0:
        return 0.0
    body = abs(candle["close"] - candle["open"])
    return body / total_range


def wick_ratio(candle: Candle, side: str = "lower") -> float:
    """
    Ratio mèche/range total.
    side='lower' → mèche basse (rejet haussier)
    side='upper' → mèche haute (rejet baissier)
    > 0.6 = longue mèche de rejet (confirmation valide).
    """
    total_range = candle["high"] - candle["low"]
    if total_range == 0:
        return 0.0
    if side == "lower":
        wick = min(candle["open"], candle["close"]) - candle["low"]
    else:
        wick = candle["high"] - max(candle["open"], candle["close"])
    return wick / total_range


def is_spike_candle(
    candle: Candle,
    avg_range: float,
    multiplier: float = 3.0,
) -> bool:
    """
    Détecte une bougie spike Boom/Crash : range > multiplier × range moyen.
    Les spikes invalident toute analyse en cours sur ce timeframe.
    """
    if avg_range == 0:
        return False
    return (candle["high"] - candle["low"]) > avg_range * multiplier


def detect_recent_spike(
    candles: list[Candle],
    lookback: int = 30,
    recent_window: int = 5,
    multiplier: float = 3.0,
) -> bool:
    """
    Retourne True si un spike a été détecté dans les `recent_window`
    dernières bougies. Utilisé pour suspendre l'analyse post-spike.
    """
    if len(candles) < lookback + recent_window:
        return False
    baseline = candles[-(lookback + recent_window):-recent_window]
    avg_range = sum(c["high"] - c["low"] for c in baseline) / len(baseline)
    recent = candles[-recent_window:]
    return any(is_spike_candle(c, avg_range, multiplier) for c in recent)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_all_market_data() -> dict[str, InstrumentData]:
    """
    Récupère les données de marché pour tous les instruments.
    Retourne un dict {instrument_name: {timeframe: [candles]}}.
    Instruments avec données insuffisantes sont exclus du résultat.
    """
    app_id = os.environ.get("DERIV_APP_ID", "1089")
    client = DerivClient(app_id=app_id)

    all_data: dict[str, InstrumentData] = {}

    try:
        await client.connect()
        for name, symbol in INSTRUMENTS.items():
            logger.info(f"Récupération: {name} ({symbol})...")
            data = await client.get_all_timeframes(symbol)
            if client.validate_instrument_data(name, data):
                all_data[name] = data
            else:
                logger.warning(f"{name}: données insuffisantes — exclu de l'analyse")
    finally:
        await client.close()

    logger.info(
        f"Données récupérées: {len(all_data)}/{len(INSTRUMENTS)} instruments valides"
    )
    return all_data
