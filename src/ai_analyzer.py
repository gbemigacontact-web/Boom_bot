"""
ai_analyzer.py
==============
Analyse IA optionnelle via l'API Gemini de Google.
Appelée depuis scenario_engine pour valider les signaux.
"""

import os
import json
import logging
import requests

logger = logging.getLogger("ai_analyzer")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def analyze_signal_with_gemini(context: dict) -> str:
    """
    Envoie un résumé textuel du contexte de marché à Gemini et retourne
    l'analyse. Retourne une chaîne vide si la clé API est absente.
    """
    if not GEMINI_API_KEY:
        return ""

    # Construire un prompt structuré
    prompt = f"""
Tu es un analyste financier expert en Smart Money Concepts (SMC).
Analyse la configuration suivante sur Boom & Crash et donne un avis clair (favorable/défavorable) avec une brève justification.

Instrument : {context.get('instrument')}
Tendance H4 : {context.get('trend_h4')}
Tendance H1 : {context.get('trend_h1')}
Scénario : {context.get('scenario')}
Direction attendue : {context.get('direction')}
Prix actuel : {context.get('current_price')}

Structures détectées :
- BOS : {context.get('bos')}
- CHoCH : {context.get('choch')}
- OB : {context.get('ob')}
- FVG : {context.get('fvg')}
- Fibonacci : {context.get('fib')}

Confirmation M15 : {context.get('confirmation')}

Donne ta recommandation (ENTRER / ATTENDRE / IGNORER) et explique en 2 phrases.
"""
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        params = {"key": GEMINI_API_KEY}
        resp = requests.post(GEMINI_URL, params=params, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            logger.warning(f"Gemini API error {resp.status_code}: {resp.text}")
            return ""
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return ""
