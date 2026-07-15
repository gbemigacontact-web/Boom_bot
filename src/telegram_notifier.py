"""
telegram_notifier.py
====================
Notifications Telegram — Architecture hybride IA du bot Boom & Crash v2.

Consomme désormais le format unifié `OrchestratorResult` /
`OrchestratorSignal` (scenario_orchestrator.py), qu'un résultat vienne de
Gemini ("IA") ou du moteur de secours Python ("FALLBACK") — ce module
n'a plus besoin de distinguer les deux formats séparément.

RÈGLES D'ENVOI (pour éviter le spam Telegram) :
  1. Signaux d'entrée → toujours envoyés (priorité maximale).
  2. Clôtures de position (SL/TP) → toujours envoyées.
  3. Alertes structurelles → seulement si `result.alert` est rempli,
     ce que l'orchestrateur ne fait que sur un changement réel (nouvelle
     lecture, invalidation, erreur) — jamais pour une simple confirmation
     "rien de nouveau".
  4. Résumé de fin de run → toujours envoyé, avec le statut compact de
     CHAQUE instrument (`status_text`), qu'il ait ou non déclenché une
     notification individuelle. Aucune information n'est donc perdue.
"""

import logging
import os
from datetime import datetime, timezone

import requests

from scenario_orchestrator import OrchestratorResult, OrchestratorSignal

logger = logging.getLogger("telegram_notifier")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

DIR_EMOJI = {"HAUSSIER": "🟢", "BAISSIER": "🔴"}

SOURCE_EMOJI = {
    "IA":       "🧠",
    "FALLBACK": "🐍",
    "GUARD":    "⏭",
    "POSITION": "📊",
    "ERROR":    "⚠️",
}

SOURCE_LABEL = {
    "IA":       "Gemini",
    "FALLBACK": "Moteur de secours Python",
}

MAX_TEXT_BLOCK = 500  # limite de troncature pour les textes libres (analyse Gemini)


def _sep() -> str:
    return "─" * 28


def _truncate(text: str, max_len: int = MAX_TEXT_BLOCK) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Formatage des messages
# ─────────────────────────────────────────────────────────────────────────────

def format_signal_message(signal: OrchestratorSignal) -> str:
    """
    Message complet d'entrée — format analyste.
    Inclut la source (IA/fallback), le raisonnement structurel complet
    si disponible (Gemini), et le score de confiance.
    """
    dir_emoji = DIR_EMOJI.get(signal.direction, "⚪")
    dir_label = "ACHAT (BUY)" if signal.direction == "HAUSSIER" else "VENTE (SELL)"
    source_emoji = SOURCE_EMOJI.get(signal.source, "")
    source_label = SOURCE_LABEL.get(signal.source, signal.source)

    risk = abs(signal.entry_price - signal.stop_loss)
    rr_tp1 = abs(signal.tp1 - signal.entry_price) / risk if risk else 0
    rr_tp2 = abs(signal.tp2 - signal.entry_price) / risk if risk else 0
    rr_tp3 = abs(signal.tp3 - signal.entry_price) / risk if risk else 0

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    lines = [
        f"{dir_emoji} <b>SIGNAL {dir_label}</b>  {source_emoji}",
        f"<b>{signal.instrument}</b> — {now}",
        f"<i>Source : {source_label}</i>",
        _sep(),
        f"📋 <b>Scénario :</b> {signal.scenario}",
    ]

    if signal.confiance is not None:
        lines.append(f"🎯 <b>Confiance IA :</b> {signal.confiance}%")

    lines += [
        _sep(),
        f"<b>Entrée    :</b> <code>{signal.entry_price:.4f}</code>",
        f"<b>Stop Loss :</b> <code>{signal.stop_loss:.4f}</code>",
        "",
        f"<b>TP1</b> : <code>{signal.tp1:.4f}</code>  → RR <b>1:{rr_tp1:.1f}</b>",
        f"<b>TP2</b> : <code>{signal.tp2:.4f}</code>  → RR <b>1:{rr_tp2:.1f}</b>",
        f"<b>TP3</b> : <code>{signal.tp3:.4f}</code>  → RR <b>1:{rr_tp3:.1f}</b>",
        _sep(),
        f"🔍 <b>Confirmation :</b> {signal.confirmation_kind}",
    ]

    if signal.analyse_structurelle:
        lines.append(_sep())
        lines.append("📊 <b>Analyse structurelle :</b>")
        lines.append(_truncate(signal.analyse_structurelle))

    if signal.commentaire_strategique:
        lines.append(_sep())
        lines.append(f"💬 {_truncate(signal.commentaire_strategique, 300)}")

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


def format_alert_message(result: OrchestratorResult) -> str:
    """Message d'alerte individuelle — uniquement pour un évènement notable."""
    emoji = SOURCE_EMOJI.get(result.source, "ℹ️")
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        f"{emoji} <b>[{result.instrument}]</b> — {now}",
        result.alert or "",
    ]
    return "\n".join(lines)


def format_position_closed_message(result: OrchestratorResult) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return (
        f"📊 <b>[{result.instrument}]</b> — {now}\n"
        f"{result.position_close_reason}"
    )


def format_run_summary(
    results: list[OrchestratorResult],
    run_time_seconds: float,
    quota_exhausted: bool = False,
) -> str:
    """
    Résumé envoyé à chaque run — donne le statut de CHAQUE instrument,
    même ceux sans notification individuelle, pour ne jamais perdre
    d'information.
    """
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    signals_count = sum(1 for r in results if r.signal)
    closed_count = sum(1 for r in results if r.position_closed)
    skipped_count = sum(1 for r in results if r.skipped)
    ia_count = sum(1 for r in results if r.source == "IA")
    fallback_count = sum(1 for r in results if r.source == "FALLBACK")

    lines = [
        f"🔍 <b>Scan terminé</b> — {now}",
        f"Durée : {run_time_seconds:.0f}s | Signaux : {signals_count} | "
        f"Clôtures : {closed_count} | Ignorés : {skipped_count}",
        f"Analysé par IA : {ia_count} | Fallback Python : {fallback_count}"
        + (" | ⚠️ quota IA épuisé ce run" if quota_exhausted else ""),
        _sep(),
    ]

    for r in results:
        emoji = SOURCE_EMOJI.get(r.source, "⚪")
        status = r.status_text or r.skip_reason or r.alert or r.source
        suffix = (
            " 🎯 SIGNAL" if r.signal else
            " ✅ CLÔTURÉ" if r.position_closed else ""
        )
        lines.append(
            f"{emoji} <code>{r.instrument:<12}</code> {status}{suffix}"
        )

    return "\n".join(lines)


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
    results: list[OrchestratorResult],
    run_time_seconds: float,
    send_summary: bool = True,
    quota_exhausted: bool = False,
) -> None:
    """
    Traite tous les résultats d'un run et envoie les messages appropriés.

    Ordre d'envoi :
      1. Signaux d'entrée (priorité maximale)
      2. Clôtures de position
      3. Alertes structurelles notables
      4. Résumé de run (toujours en dernier)
    """
    # ── 1. Signaux d'entrée ───────────────────────────────────────────────────
    for result in results:
        if result.signal:
            sent = _send(format_signal_message(result.signal))
            logger.info(f"Signal {result.instrument} envoyé: {sent}")

    # ── 2. Clôtures de position ───────────────────────────────────────────────
    for result in results:
        if result.position_closed:
            sent = _send(format_position_closed_message(result))
            logger.info(f"Clôture {result.instrument} envoyée: {sent}")

    # ── 3. Alertes structurelles notables ─────────────────────────────────────
    for result in results:
        if result.signal or result.position_closed:
            continue  # déjà traités
        if result.alert:
            _send(format_alert_message(result))

    # ── 4. Résumé de run ──────────────────────────────────────────────────────
    if send_summary:
        summary = format_run_summary(results, run_time_seconds, quota_exhausted)
        _send(summary)
