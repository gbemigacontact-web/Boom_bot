"""
telegram_notifier.py
====================
Envoi des signaux et alertes via Telegram.
"""

import os
import logging
import requests
from typing import List

logger = logging.getLogger("telegram_notifier")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_message(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram non configuré")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Erreur Telegram: {e}")
        return False


def notify_results(results: List, run_time_seconds: float, send_summary: bool = True):
    signals = [r for r in results if r.signal]
    alerts = [r for r in results if r.alert and not r.signal]
    skipped = [r for r in results if r.skipped]

    for alert in alerts:
        send_message(f"⚠️ Alerte: {alert.alert}")

    for sig in signals:
        s = sig.signal
        text = (
            f"🚨 <b>Signal {s.instrument} — {s.scenario}</b>\n"
            f"Direction: {s.direction.value}\n"
            f"Entrée: {s.entry_price:.4f}\n"
            f"Stop: {s.stop_loss:.4f}\n"
            f"TP1: {s.tp1:.4f} | TP2: {s.tp2:.4f} | TP3: {s.tp3:.4f}\n"
            f"Confirmation: {s.confirmation_kind}\n"
            + "\n".join(f"• {l}" for l in s.context_lines)
        )
        if s.ai_analysis:
            text += f"\n\n🤖 Analyse IA:\n{s.ai_analysis}"
        send_message(text)

    if send_summary:
        summary = f"🏁 Run terminé en {run_time_seconds:.1f}s — {len(signals)} signal(s), {len(alerts)} alerte(s), {len(skipped)} ignoré(s)"
        send_message(summary)
