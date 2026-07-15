"""
deriv_client.py
===============
Client WebSocket Deriv pour le bot Boom & Crash v2 — Architecture hybride IA.

RÔLE DANS LA NOUVELLE ARCHITECTURE
-----------------------------------
Ce module est désormais un pur FOURNISSEUR DE DONNÉES BRUTES. Il ne fait
aucune interprétation SMC. Ses données OHLCV alimentent deux consommateurs :

  1. chart_renderer.py → génère les images (H4/H1 vue large ~500 bougies,
     M15 zoom) envoyées à Gemini pour le jugement structurel complet.
  2. fallback_engine.py → logique Python pure utilisée uniquement si
     l'appel IA échoue (quota, erreur réseau, JSON invalide).

POURQUOI 500 BOUGIES SUR H4/H1
-------------------------------
Une IA qui ne voit qu'une fenêtre étroite (150 bougies ≈ 25 jours en H4)
ne peut pas identifier correctement les vrais objectifs institutionnels
(liquidité majeure, anciens plus hauts/plus bas, zones d'accumulation/
distribution de fond). Une vue de 500 bougies donne :
  - H4 : ~83 jours de contexte
  - H1 : ~20,8 jours de contexte
C'est cette profondeur qui permet à l'IA de distinguer un vrai retournement
institutionnel d'un simple pullback local — c'est la garantie contre les
faux signaux dus à une vue trop courte.

Note technique importante : augmenter `count` dans la requête Deriv
n'ajoute PAS de round-trip réseau supplémentaire. Une requête
`ticks_history` avec count=500 prend le même nombre d'allers-retours
qu'avec count=150 — seul le volume de la réponse change. L'impact sur le
budget de temps du run GitHub Actions (12 min) reste donc négligeable.

Architecture générale (inchangée) :
- Une seule connexion WebSocket par run
- Requêtes séquentielles avec pause entre chaque (rate-limit Deriv)
- Volume inclus dans chaque bougie (proxy basé sur le range)
- Retry automatique (3 tentatives) + timeout explicite par requête
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
#
# - H4/H1 : 500 bougies → VUE INSTITUTIONNELLE LARGE, indispensable pour
#   que l'IA identifie les vrais objectifs de liquidité et la tendance de
#   fond avant de juger un scénario SMC. C'est le cœur de la fiabilité du
#   jugement Gemini — ne pas réduire sans raison forte.
# - M30 : 200 bougies → conservé pour le moteur de fallback Python
#   (non utilisé dans les images IA).
# - M15 : 200 bougies → fournit assez d'historique pour le fallback et
#   pour permettre au futur chart_renderer de zoomer sur la zone de
#   pullback active sans manquer de contexte immédiat.
# - M5 : 250 bougies → confirmation finale d'entrée + moyenne de volume
#   sur 20 périodes.
CANDLE_COUNT: dict[str, int] = {
    "H4":  500,
    "H1":  500,
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

# Timeout explicite par requête individuelle (secondes). Sans ceci, un
# ws.recv() qui ne répond jamais peut consommer tout le budget du run
# sans jamais déclencher le mécanisme de retry.
REQUEST_TIMEOUT = 15.0

# Timeout pour l'établissement de la connexion initiale
CONNECT_TIMEOUT = 10.0

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
        try:
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    ping_interval=25,
                    ping_timeout=15,
                    close_timeout=10,
                ),
                timeout=CONNECT_TIMEOUT,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Connexion Deriv WebSocket: timeout après {CONNECT_TIMEOUT}s"
            ) from e
        self._connected = True
        logger.info(f"Connecté à Deriv WebSocket (app_id={self.app_id})")

    async def close(self) -> None:
        if self.ws and self._connected:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Erreur à la fermeture de la connexion: {e}")
            finally:
                self._connected = False
                logger.info("Connexion Deriv fermée.")

    # ── Requête de bougies ────────────────────────────────────────────────────

    async def _fetch_candles_raw(
        self, symbol: str, granularity: int, count: int
    ) -> list[dict]:
        """
        Envoie une requête ticks_history et retourne les bougies brutes.
        Lève une RuntimeError si l'API retourne une erreur ou si la
        requête dépasse REQUEST_TIMEOUT secondes.
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

        if not self._connected or self.ws is None:
            raise RuntimeError(f"WebSocket non connecté pour {symbol}")

        try:
            await asyncio.wait_for(
                self.ws.send(json.dumps(request)), timeout=REQUEST_TIMEOUT
            )
            raw_response = await asyncio.wait_for(
                self.ws.recv(), timeout=REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Timeout Deriv [{symbol} g={granularity} count={count}] "
                f"après {REQUEST_TIMEOUT}s"
            ) from e

        response = json.loads(raw_response)

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

        Seuils H4/H1 relevés à 200 (au lieu de 60) : si Deriv ne renvoie
        pas assez d'historique pour offrir une vraie vue institutionnelle
        large à l'IA, on exclut l'instrument du run plutôt que de nourrir
        Gemini avec une vue tronquée qui fausserait son jugement de
        tendance de fond et d'objectifs de liquidité.
        """
        minimums = {"H4": 200, "H1": 200, "M30": 80, "M15": 80, "M5": 30}
        for tf, min_count in minimums.items():
            candles = data.get(tf, [])
            if len(candles) < min_count:
                logger.warning(
                    f"{instrument} {tf}: seulement {len(candles)} bougies "
                    f"(min requis: {min_count}) — données insuffisantes "
                    f"pour une vue institutionnelle fiable"
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
