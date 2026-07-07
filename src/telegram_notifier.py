"""
telegram_notifier.py
====================
Module de notification Telegram pour le bot Boom & Crash v2.

Génère des messages structurés "analyste" — pas de simples flèches.
Chaque message explique le contexte, la zone, la confirmation et
les objectifs. Le trader lit le message et comprend POURQUOI.

3 types de messages :
  1. Alerte intermédiaire : BOS détecté, prix en zone, CHoCH, etc.
  2. Signal d'entrée complet : direction, entrée, SL, TP1/TP2/TP3, contexte
  3. Résumé de run : état de tous les instruments après chaque scan
"""

import logging
import os
from datetime import datetime, timezone

import requests

from scenario_engine import EngineResult, SignalResult
from state_machine import MarketState
from technical_analysis import Direction

logger = logging.getLogger("telegram_notifier")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# Emojis par direction
DIR_EMOJI = {
    Direction.BULLISH: "🟢",
    Direction.BEARISH: "🔴",
    "BUY":  "🟢",
    "SELL": "🔴",
}

# Emojis par état FSM
STATE_EMOJI = {
    MarketState.NEUTRE:                    "⚪",
    MarketState.BOS_DETECTE:              "🔔",
    MarketState.PULLBACK_ZONE_FVG:        "📍",
    MarketState.PULLBACK_ZONE_OB:         "📍",
    MarketState.CONFIRMATION_ATTENDUE:    "✅",
    MarketState.ENTREE_ACTIVE:            "🎯",
    MarketState.RISQUE_CHOCH:             "⚠️",
    MarketState.RETOURNEMENT_SURVEILLANCE:"🔄",
    MarketState.FIBONACCI_ACTIF:          "📐",
}


# ─────────────────────────────────────────────────────────────────────────────
# Formatage des messages
# ─────────────────────────────────────────────────────────────────────────────

def _sep() -> str:
    return "─" * 28


def format_signal_message(signal: SignalResult) -> str:
    """
    Message complet d'entrée — format analyste.
    Structure :
      En-tête direction + instrument
      Scénario identifié
      Prix d'entrée / SL / TP1 / TP2 / TP3
      Ratio R:R sur TP1 et TP3
      Détail de la confirmation
      Contexte multi-timeframe
      Avertissement
    """
    direction = signal.direction
    emoji = DIR_EMOJI.get(direction, "⚪")
    dir_label = "ACHAT (BUY)" if direction == Direction.BULLISH else "VENTE (SELL)"

    risk = abs(signal.entry_price - signal.stop_loss)
    rr_tp1 = abs(signal.tp1 - signal.entry_price) / risk if risk else 0
    rr_tp2 = abs(signal.tp2 - signal.entry_price) / risk if risk else 0
    rr_tp3 = abs(signal.tp3 - signal.entry_price) / risk if risk else 0

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    lines = [
        f"{emoji} <b>SIGNAL {dir_label}</b>",
        f"<b>{signal.instrument}</b> — {now}",
        _sep(),
        f"📋 <b>Scénario :</b> {signal.scenario}",
        _sep(),
        f"<b>Entrée    :</b> <code>{signal.entry_price:.4f}</code>",
        f"<b>Stop Loss :</b> <code>{signal.stop_loss:.4f}</code>",
        "",
        f"<b>TP1</b> (M15 swing)  : <code>{signal.tp1:.4f}</code>  "
        f"→ RR <b>1:{rr_tp1:.1f}</b>",
        f"<b>TP2</b> (H1 OB)      : <code>{signal.tp2:.4f}</code>  "
        f"→ RR <b>1:{rr_tp2:.1f}</b>",
        f"<b>TP3</b> (H4 cible)   : <code>{signal.tp3:.4f}</code>  "
        f"→ RR <b>1:{rr_tp3:.1f}</b>",
        _sep(),
        f"🔍 <b>Confirmation :</b> {signal.confirmation_kind}",
        f"  Mèche   : {signal.wick_ratio:.0%} du range",
        f"  Corps   : {signal.body_ratio:.0%} du range",
        f"  Volume  : {signal.volume_ratio:.1f}× la moyenne 20",
    ]

    if signal.context_lines:
        lines.append(_sep())
        lines.append("📊 <b>Contexte :</b>")
        for cl in signal.context_lines:
            lines.append(f"  {cl}")

    lines += [
        _sep(),
        "💡 <b>Gestion suggérée :</b>",
        "  • Clôturer 40% à TP1 → SL au breakeven",
        "  • Clôturer 35% à TP2 → laisser courir",
        "  • Laisser 25% jusqu'à TP3 (objectif long)",
        _sep(),
        "⚠️ <i>Signal automatique — vérifier la structure avant d'exécuter.</i>",
    ]

    return "\n".join(lines)


def format_alert_message(instrument: str, alert: str, state_after: MarketState) -> str:
    """
    Message d'alerte intermédiaire (BOS, zone touchée, CHoCH...).
    Plus court, informatif, sans signal d'entrée.
    """
    emoji = STATE_EMOJI.get(state_after, "ℹ️")
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    lines = [
        f"{emoji} <b>[{instrument}]</b> — {now}",
        alert,
        f"<i>État : {state_after.value}</i>",
    ]
    return "\n".join(lines)


def format_run_summary(
    results: list[EngineResult],
    run_time_seconds: float,
) -> str:
    """
    Résumé envoyé à chaque run — même si aucun signal.
    Confirme que le bot tourne et donne l'état de chaque instrument.
    """
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    signals_count = sum(1 for r in results if r.signal)
    alerts_count  = sum(1 for r in results if r.alert and not r.signal)
    skipped_count = sum(1 for r in results if r.skipped)

    lines = [
        f"🔍 <b>Scan terminé</b> — {now}",
        f"Durée : {run_time_seconds:.0f}s | "
        f"Signaux : {signals_count} | "
        f"Alertes : {alerts_count} | "
        f"Ignorés : {skipped_count}",
        _sep(),
    ]

    for r in results:
        if r.skipped:
            lines.append(
                f"⏭ <code>{r.instrument:<12}</code> — ignoré ({r.skip_reason[:30]})"
            )
        else:
            emoji = STATE_EMOJI.get(r.state_after, "⚪")
            transition = (
                f"{r.state_before.value} → {r.state_after.value}"
                if r.state_before != r.state_after
                else r.state_after.value
            )
            suffix = " 🎯 SIGNAL" if r.signal else ""
            lines.append(
                f"{emoji} <code>{r.instrument:<12}</code> {transition}{suffix}"
            )

    return "\n".join(lines)


def format_entree_active_update(
    instrument: str,
    current_price: float,
    entry_price: float,
    stop_loss: float,
    tp1: float,
    direction: str,
) -> str:
    """
    Mise à jour d'un trade actif (TP ou SL atteint).
    """
    risk = abs(entry_price - stop_loss)
    pnl_pips = (
        (entry_price - current_price) if direction == "BEARISH"
        else (current_price - entry_price)
    )
    pnl_rr = pnl_pips / risk if risk else 0
    emoji = "🎯" if pnl_pips > 0 else "🛑"

    return (
        f"{emoji} <b>Mise à jour trade — {instrument}</b>\n"
        f"Prix actuel : <code>{current_price:.4f}</code>\n"
        f"P&L estimé  : {pnl_rr:+.2f}R"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Envoi Telegram
# ─────────────────────────────────────────────────────────────────────────────

def _send(message: str, parse_mode: str = "HTML") -> bool:
    """Envoie un message via l'API Telegram Bot."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant.")
        return False

    try:
        resp = requests.post(
            TELEGRAM_API.format(token=token),
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error(f"Erreur Telegram: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Interface publique
# ─────────────────────────────────────────────────────────────────────────────

def notify_results(
    results: list[EngineResult],
    run_time_seconds: float,
    send_summary: bool = True,
) -> None:
    """
    Traite tous les résultats d'un run et envoie les messages appropriés.

    Ordre d'envoi :
      1. Signaux d'entrée (priorité maximale)
      2. Alertes intermédiaires importantes
      3. Résumé de run (toujours en dernier)
    """
    # ── 1. Signaux d'entrée ───────────────────────────────────────────────────
    for result in results:
        if result.signal:
            msg = format_signal_message(result.signal)
            sent = _send(msg)
            logger.info(
                f"Signal {result.instrument} envoyé: {sent}"
            )

    # ── 2. Alertes intermédiaires ─────────────────────────────────────────────
    # On envoie seulement les alertes "importantes" pour ne pas spammer.
    # États qui méritent une alerte Telegram :
    important_states = {
        MarketState.BOS_DETECTE,
        MarketState.PULLBACK_ZONE_FVG,
        MarketState.PULLBACK_ZONE_OB,
        MarketState.RISQUE_CHOCH,
        MarketState.RETOURNEMENT_SURVEILLANCE,
        MarketState.FIBONACCI_ACTIF,
    }

    for result in results:
        if result.signal:
            continue   # déjà traité
        if result.alert and result.state_after in important_states:
            # N'envoyer l'alerte que si l'état a changé (évite le spam)
            if result.state_before != result.state_after:
                msg = format_alert_message(
                    result.instrument, result.alert, result.state_after
                )
                _send(msg)

    # ── 3. Résumé de run ──────────────────────────────────────────────────────
    if send_summary:
        summary = format_run_summary(results, run_time_seconds)
        _send(summary)
