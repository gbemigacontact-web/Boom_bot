"""
deriv_client.py
===============
Client Deriv WebSocket pour récupérer les données multi-timeframe
avec profondeur adaptée à l'analyse SMC long terme.
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger("deriv_client")

APP_ID = 1089  # ID d'application Deriv (testing)
INSTRUMENTS = ["BOOM500", "BOOM900", "BOOM1000", "CRASH500", "CRASH900", "CRASH1000"]
TIMEFRAMES = {
    "H4": 14400,
    "H1": 3600,
    "M15": 900,
    "M5": 300,
}
# Nombre de bougies à récupérer
CANDLE_COUNTS = {
    "H4": 400,
    "H1": 400,
    "M15": 200,
    "M5": 200,
}

InstrumentData = Dict[str, list]  # {"H4": [...], "H1": [...], ...}


async def fetch_candles(ws, instrument: str, granularity: int, count: int) -> list:
    """Récupère les `count` dernières bougies pour un instrument et un timeframe."""
    req = {
        "ticks_history": instrument,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": granularity,
        "style": "candles",
    }
    await ws.send(json.dumps(req))
    response = await ws.recv()
    data = json.loads(response)
    if "candles" in data:
        return data["candles"]
    logger.error(f"Erreur candles {instrument} {granularity}: {data.get('error', 'inconnu')}")
    return []


async def fetch_all_market_data() -> Dict[str, InstrumentData]:
    """
    Récupère toutes les données pour tous les instruments et timeframes.
    """
    uri = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
    async with websockets.connect(uri) as ws:
        # Autoriser les ticks
        await ws.send(json.dumps({"ticks": "R_100", "subscribe": 0}))  # bidon pour init
        _ = await ws.recv()

        logger.info("Connecté à Deriv WebSocket (app_id=%s)", APP_ID)
        result = {}

        for symbol in INSTRUMENTS:
            instrument_name = symbol.replace("BOOM", "Boom ").replace("CRASH", "Crash ")
            logger.info("Récupération: %s (%s)...", instrument_name, symbol)
            instrument_data = {}

            for tf, secs in TIMEFRAMES.items():
                count = CANDLE_COUNTS[tf]
                candles = await fetch_candles(ws, symbol, secs, count)
                # Transformer les candles en dicts utilisables
                formatted = []
                for c in candles:
                    formatted.append({
                        "epoch": int(c["epoch"]),
                        "time": datetime.fromtimestamp(int(c["epoch"]), tz=timezone.utc).isoformat(),
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": 0.0,  # Deriv ne fournit pas le volume, on le met à 0
                    })
                instrument_data[tf] = formatted

            result[instrument_name] = instrument_data

        logger.info("Connexion Deriv fermée.")
    return result


def is_spread_abnormal(data: InstrumentData) -> bool:
    """Filtre de sécurité : toujours False car pas de spread réel avec Deriv synthétiques."""
    return False


def detect_recent_spike(m5_candles: list) -> bool:
    """
    Détecte un spike récent (mèche > 0.5% du range sur les 5 dernières bougies M5).
    """
    if len(m5_candles) < 5:
        return False
    recent = m5_candles[-5:]
    for c in recent:
        high = c["high"]
        low = c["low"]
        if (high - low) > 0.005 * low:  # 0.5%
            return True
    return False


def get_current_price(data: InstrumentData) -> float:
    """Retourne le prix actuel à partir de la dernière bougie M5."""
    m5 = data.get("M5", [])
    if m5:
        return m5[-1]["close"]
    return 0.0
