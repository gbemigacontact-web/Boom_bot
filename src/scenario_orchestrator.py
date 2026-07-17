"""
scenario_orchestrator.py
==========================
Chef d'orchestre du bot Boom & Crash v2 — Architecture hybride IA.

RÔLE
-----
Ce module relie tous les autres :
  - state_machine.py    → mémoire (positions actives + continuité IA)
  - chart_renderer.py   → images pour Gemini
  - ai_analyzer.py       → jugement structurel complet
  - fallback_engine.py   → filet de sécurité Python pur (ex scenario_engine.py)
  - technical_analysis.py → calculs numériques exacts (SL/TP, confirmation)

DÉCISION PAR INSTRUMENT, À CHAQUE RUN (dans cet ordre) :
  1. Position déjà active ? → vérification de prix uniquement (pas d'IA).
  2. Marché anormal (spread/spike) ? → run ignoré pour cet instrument.
  3. Quota Gemini déjà épuisé ce run ? → bascule directe sur le fallback.
  4. Appel Gemini → si succès, application du verdict avec recalcul
     Python des prix exacts (jamais un chiffre estimé visuellement).
  5. Échec Gemini (panne, quota, image manquante) → bascule automatique
     et silencieuse sur fallback_engine.py pour CET instrument seulement
     — les autres instruments du run ne sont jamais affectés.

GARANTIE DE SÉCURITÉ NUMÉRIQUE
---------------------------------
Le niveau d'invalidation choisi par Gemini est vérifié pour cohérence
directionnelle (un swing bas pour un achat, un swing haut pour une
vente) avant d'être utilisé. En cas d'incohérence ou d'absence de
candidat exploitable, le niveau est recalculé directement depuis la
structure H1 par Python — jamais depuis une estimation de l'IA.
Un signal est annulé (jamais émis) si le ratio risque/rendement sur
TP1 est inférieur à 1:1.5, quel que soit le verdict de Gemini.

⚠️ Dépendance vers fallback_engine.py (fichier encore à livrer dans cet
échange — ce sera le scenario_engine.py actuel, renommé et conservé
intact). L'interface attendue : classe ScenarioEngine(state_manager)
avec méthode .process(instrument, data) -> EngineResult, et une
constante INSTRUMENT_BIAS: dict[str, Direction].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from deriv_client import (
    InstrumentData, INSTRUMENTS,
    get_current_price, is_spread_abnormal, detect_recent_spike,
)
from state_machine import StateManager, ActiveTrade
from technical_analysis import (
    Direction, detect_trend, find_htf_targets, compute_sl_tp,
    check_confirmation_candle, detect_mini_choch, find_swing_points,
)
from chart_renderer import render_instrument_charts
from ai_analyzer import (
    analyze_instrument, reset_quota_state, is_quota_exhausted,
    AnalyzerResult, ActionRequise, Candidate,
)
from fallback_engine import ScenarioEngine, EngineResult, INSTRUMENT_BIAS

logger = logging.getLogger("scenario_orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# Résultat unifié — utilisé par main.py / telegram_notifier.py, quelle que
# soit la source (IA ou fallback). Ça évite à tout le reste du bot de devoir
# distinguer les deux origines.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorSignal:
    instrument: str
    direction: str          # "HAUSSIER" | "BAISSIER"
    scenario: str
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confirmation_kind: str
    commentaire_strategique: str
    analyse_structurelle: str
    source: str              # "IA" | "FALLBACK"
    confiance: Optional[int] = None   # non applicable au fallback (None)


@dataclass
class OrchestratorResult:
    instrument: str
    source: str               # "IA" | "FALLBACK" | "GUARD" | "POSITION" | "ERROR"
    signal: Optional[OrchestratorSignal] = None
    # `alert` : uniquement rempli pour un évènement NOTABLE (changement de
    # lecture, invalidation, erreur) → déclenche un message Telegram
    # individuel. Reste à None pour une confirmation "rien de nouveau",
    # afin d'éviter le spam à chaque run.
    alert: Optional[str] = None
    # `status_text` : TOUJOURS rempli, résumé compact utilisé dans le
    # résumé de fin de run — même quand `alert` est None.
    status_text: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""
    position_closed: bool = False
    position_close_reason: str = ""
    fsm_transition: Optional[str] = None  # info de debug pour les résultats fallback


def _family(instrument: str) -> str:
    return "Boom" if instrument.startswith("Boom") else "Crash"


def _direction_to_french(d: Direction) -> str:
    return "HAUSSIER" if d == Direction.BULLISH else "BAISSIER"


# ─────────────────────────────────────────────────────────────────────────────
# Vérification d'une position déjà active (aucun appel IA nécessaire)
# ─────────────────────────────────────────────────────────────────────────────

def _check_active_position(
    instrument: str,
    data: InstrumentData,
    state_manager: StateManager,
    current_price: float,
    fallback: ScenarioEngine,
) -> OrchestratorResult:
    """
    Si la position active vient de l'IA (mémoire dédiée), on fait la
    vérification de prix nous-mêmes ici — c'est une simple comparaison
    numérique, aucun besoin de rappeler Gemini.

    Si la position vient du fallback Python (FSM legacy ENTREE_ACTIVE),
    on laisse fallback_engine.py gérer via son propre handler existant
    (déjà testé et validé) — pas de duplication de logique.
    """
    ai_trade = state_manager.get_active_trade(instrument)

    if ai_trade is None:
        # Position ouverte par le fallback → sa propre logique s'en occupe
        return _run_fallback(instrument, data, fallback)

    direction = ai_trade.direction
    sl, tp1 = ai_trade.stop_loss, ai_trade.tp1

    if direction == "BAISSIER":
        hit_sl = current_price >= sl
        hit_tp1 = current_price <= tp1
    else:
        hit_sl = current_price <= sl
        hit_tp1 = current_price >= tp1

    if hit_sl:
        state_manager.clear_active_trade(
            instrument, reason=f"SL touché @ {current_price:.4f}"
        )
        return OrchestratorResult(
            instrument=instrument, source="POSITION",
            position_closed=True,
            position_close_reason=f"🛑 SL touché @ {current_price:.4f}",
        )

    if hit_tp1:
        state_manager.clear_active_trade(
            instrument, reason=f"TP1 atteint @ {current_price:.4f}"
        )
        return OrchestratorResult(
            instrument=instrument, source="POSITION",
            position_closed=True,
            position_close_reason=f"🎯 TP1 atteint @ {current_price:.4f}",
        )

    # Position toujours ouverte, rien de nouveau : pas d'alerte individuelle
    # (éviterait un message Telegram identique à chaque run), seulement le
    # statut visible dans le résumé de fin de run.
    return OrchestratorResult(
        instrument=instrument, source="POSITION",
        status_text=(
            f"Position {direction} en cours — prix {current_price:.4f} "
            f"(entrée {ai_trade.entry_price:.4f}, SL {sl:.4f}, TP1 {tp1:.4f})"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Résolution sécurisée du niveau d'invalidation
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_invalidation_level(
    ai_result: AnalyzerResult,
    h1: list[dict],
    direction: Direction,
) -> Optional[float]:
    """
    Utilise le candidat d'invalidation choisi par Gemini UNIQUEMENT s'il
    est cohérent avec la direction du trade (swing bas pour un achat,
    swing haut pour une vente). Sinon, recalcule directement depuis la
    structure H1 — Gemini ne peut jamais faire "passer" un niveau
    incohérent.
    """
    cand: Optional[Candidate] = ai_result.resolved_invalidation
    expected_prefix = "SWING_LOW" if direction == Direction.BULLISH else "SWING_HIGH"

    if cand and cand.price is not None:
        if cand.id.startswith(expected_prefix):
            return cand.price
        logger.warning(
            f"Candidat d'invalidation '{cand.id}' incohérent avec la "
            f"direction {direction.value} — recalcul direct depuis H1."
        )

    highs, lows = find_swing_points(h1, window=3)
    if direction == Direction.BULLISH and lows:
        return lows[-1].price
    if direction == Direction.BEARISH and highs:
        return highs[-1].price
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Application du verdict IA
# ─────────────────────────────────────────────────────────────────────────────

def _apply_ai_verdict(
    instrument: str,
    data: InstrumentData,
    state_manager: StateManager,
    ai_result: AnalyzerResult,
    current_price: float,
) -> OrchestratorResult:
    verdict = ai_result.verdict
    h1, h4, m15 = data["H1"], data["H4"], data["M15"]

    zone_label = ai_result.resolved_zone.label if ai_result.resolved_zone else None

    # Mémoire IA mise à jour dans TOUS les cas (continuité du run suivant),
    # y compris quand aucune action n'est prise. update_ai_memory incrémente
    # consecutive_watch_runs si la lecture (scénario + action) est identique
    # au run précédent — on s'en sert juste après pour décider si ce
    # run mérite une notification individuelle ou juste une ligne de résumé.
    mem = state_manager.update_ai_memory(
        instrument,
        scenario=verdict.scenario_identifie.value,
        biais=verdict.biais_institutionnel.value,
        tendance_h4=verdict.tendance_h4.value,
        tendance_h1=verdict.tendance_h1.value,
        zone_label=zone_label,
        confirmation_m15=verdict.confirmation_m15,
        type_confirmation=verdict.type_confirmation.value,
        action=verdict.action_requise.value,
        confiance=verdict.confiance,
        commentaire=verdict.commentaire_strategique,
        source="IA",
    )
    is_new_reading = mem.consecutive_watch_runs == 1

    status = (
        f"{verdict.scenario_identifie.value} | biais {verdict.biais_institutionnel.value} "
        f"| confiance {verdict.confiance}%"
    )

    if verdict.action_requise == ActionRequise.ATTENDRE:
        alert_text = None
        if is_new_reading:
            # Nouvelle lecture (différente du run précédent) : ça vaut la
            # peine de notifier. Une lecture répétée reste dans le résumé
            # uniquement, pour éviter le spam.
            alert_text = (
                f"👁️ {instrument} — {verdict.scenario_identifie.value} | "
                f"biais {verdict.biais_institutionnel.value} | "
                f"confiance {verdict.confiance}%\n{verdict.commentaire_strategique}"
            )
        return OrchestratorResult(
            instrument=instrument, source="IA",
            alert=alert_text, status_text=status,
        )

    if verdict.action_requise == ActionRequise.INVALIDE:
        return OrchestratorResult(
            instrument=instrument, source="IA",
            alert=f"❌ {instrument}: structure invalidée — {verdict.commentaire_strategique}",
            status_text=f"INVALIDÉ | {status}",
        )

    # ── ACHETER / VENDRE ──────────────────────────────────────────────────
    direction_ta = (
        Direction.BULLISH if verdict.action_requise == ActionRequise.ACHETER
        else Direction.BEARISH
    )

    # Prix d'entrée exact recalculé par Python (jamais celui de Gemini)
    conf = check_confirmation_candle(m15, direction_ta, volume_multiplier=1.5)
    if not conf.valid:
        mini = detect_mini_choch(m15, direction_ta)
        if mini:
            conf = mini

    if conf.valid:
        entry_price = conf.entry_price
    else:
        entry_price = current_price
        logger.warning(
            f"{instrument}: confirmation visuelle Gemini non corroborée "
            f"numériquement par Python — entrée au prix courant par prudence."
        )

    invalidation = _resolve_invalidation_level(ai_result, h1, direction_ta)
    if invalidation is None:
        logger.warning(f"{instrument}: aucun niveau d'invalidation exploitable")
        return OrchestratorResult(
            instrument=instrument, source="IA",
            alert=f"⚠️ {instrument}: signal IA annulé — niveau d'invalidation introuvable",
        )

    h4_targets = find_htf_targets(h4, direction_ta, current_price)
    sl, tp1, tp2, tp3 = compute_sl_tp(direction_ta, entry_price, invalidation, h4_targets, m15)

    risk = abs(entry_price - sl)
    rr_tp1 = abs(tp1 - entry_price) / risk if risk else 0
    if rr_tp1 < 1.5:
        logger.info(f"{instrument}: RR insuffisant ({rr_tp1:.1f}) — signal IA annulé")
        return OrchestratorResult(
            instrument=instrument, source="IA",
            alert=f"❌ {instrument}: RR insuffisant ({rr_tp1:.1f}) malgré confirmation IA — annulé",
        )

    trade_direction = "HAUSSIER" if verdict.action_requise == ActionRequise.ACHETER else "BAISSIER"

    trade = ActiveTrade(
        direction=trade_direction,
        entry_price=entry_price, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        scenario=verdict.scenario_identifie.value,
        confirmation_kind=verdict.type_confirmation.value,
        source="IA",
    )
    state_manager.set_active_trade(instrument, trade)

    signal = OrchestratorSignal(
        instrument=instrument, direction=trade_direction,
        scenario=verdict.scenario_identifie.value,
        entry_price=entry_price, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        confirmation_kind=verdict.type_confirmation.value,
        commentaire_strategique=verdict.commentaire_strategique,
        analyse_structurelle=verdict.analyse_structurelle,
        confiance=verdict.confiance,
        source="IA",
    )
    return OrchestratorResult(
        instrument=instrument, source="IA", signal=signal,
        status_text=f"SIGNAL {trade_direction} | {verdict.scenario_identifie.value}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bascule fallback
# ─────────────────────────────────────────────────────────────────────────────

def _convert_fallback_result(instrument: str, er: EngineResult) -> OrchestratorResult:
    """Convertit un EngineResult (fallback_engine.py) vers le format unifié."""
    transition_note = None
    if er.state_before != er.state_after:
        transition_note = f"{er.state_before.value} → {er.state_after.value}"

    if er.skipped:
        return OrchestratorResult(
            instrument=instrument, source="FALLBACK",
            skipped=True, skip_reason=er.skip_reason,
            status_text=f"ignoré : {er.skip_reason}",
            fsm_transition=transition_note,
        )

    if er.signal:
        s = er.signal
        signal = OrchestratorSignal(
            instrument=instrument,
            direction=_direction_to_french(s.direction),
            scenario=s.scenario,
            entry_price=s.entry_price, stop_loss=s.stop_loss,
            tp1=s.tp1, tp2=s.tp2, tp3=s.tp3,
            confirmation_kind=s.confirmation_kind,
            commentaire_strategique=(
                "Signal généré par le moteur de secours Python "
                "(IA indisponible pour ce run)."
            ),
            analyse_structurelle="\n".join(s.context_lines),
            confiance=None,
            source="FALLBACK",
        )
        return OrchestratorResult(
            instrument=instrument, source="FALLBACK", signal=signal,
            status_text=f"SIGNAL {signal.direction} | {signal.scenario} (fallback)",
            fsm_transition=transition_note,
        )

    return OrchestratorResult(
        instrument=instrument, source="FALLBACK",
        alert=er.alert,
        status_text=(transition_note or er.state_after.value),
        fsm_transition=transition_note,
    )


def _run_fallback(
    instrument: str, data: InstrumentData, fallback: ScenarioEngine
) -> OrchestratorResult:
    engine_result = fallback.process(instrument, data)
    return _convert_fallback_result(instrument, engine_result)


# ─────────────────────────────────────────────────────────────────────────────
# Traitement d'un instrument
# ─────────────────────────────────────────────────────────────────────────────

def _process_instrument(
    instrument: str,
    data: InstrumentData,
    state_manager: StateManager,
    fallback: ScenarioEngine,
) -> OrchestratorResult:
    if not data.get("H4") or not data.get("H1") or not data.get("M15") or not data.get("M5"):
        return OrchestratorResult(
            instrument=instrument, source="GUARD",
            skipped=True, skip_reason="Données manquantes",
            status_text="ignoré : données manquantes",
        )

    current_price = get_current_price(data)

    # 1. Position déjà active → pas d'appel IA
    if state_manager.has_active_position(instrument):
        return _check_active_position(instrument, data, state_manager, current_price, fallback)

    # 2. Gardes globaux
    if is_spread_abnormal(data):
        return OrchestratorResult(
            instrument=instrument, source="GUARD",
            skipped=True, skip_reason="Spread anormal",
            status_text="ignoré : spread anormal",
        )
    if detect_recent_spike(data.get("M5", [])):
        return OrchestratorResult(
            instrument=instrument, source="GUARD",
            skipped=True, skip_reason="Spike récent M5",
            status_text="ignoré : spike récent M5",
        )

    h4, h1 = data["H4"], data["H1"]
    bias = INSTRUMENT_BIAS[_family(instrument)]
    trend_h4 = detect_trend(h4)
    trend_h1 = detect_trend(h1)

    # 3-4. Tentative IA (sauf quota déjà épuisé ce run)
    if not is_quota_exhausted():
        try:
            charts = render_instrument_charts(instrument, data)
        except Exception as e:
            logger.error(
                f"{instrument}: échec génération graphiques — {e}", exc_info=True
            )
            charts = None

        if charts and charts.is_complete():
            previous_context = state_manager.get_previous_ai_context(instrument)
            ai_result = analyze_instrument(
                instrument, bias, current_price, trend_h4, trend_h1,
                h4, h1, charts, previous_context,
            )
            if ai_result.success:
                return _apply_ai_verdict(instrument, data, state_manager, ai_result, current_price)
            logger.warning(
                f"{instrument}: analyse IA indisponible ({ai_result.error}) "
                f"— bascule fallback Python"
            )
        else:
            errors = ", ".join(charts.errors) if charts else "génération échouée"
            logger.warning(
                f"{instrument}: graphiques incomplets ({errors}) — bascule fallback Python"
            )

    # 5. Fallback
    return _run_fallback(instrument, data, fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée du run
# ─────────────────────────────────────────────────────────────────────────────

def run_orchestrator(
    market_data: dict[str, InstrumentData],
    state_manager: StateManager,
) -> list[OrchestratorResult]:
    """
    Point d'entrée appelé par main.py à chaque run GitHub Actions.
    Traite chaque instrument indépendamment — une erreur ou une panne
    sur l'un n'affecte jamais le traitement des autres.
    """
    reset_quota_state()
    fallback = ScenarioEngine(state_manager)
    results: list[OrchestratorResult] = []

    for instrument in INSTRUMENTS:
        if instrument not in market_data:
            logger.warning(f"{instrument}: pas de données — ignoré")
            continue
        try:
            result = _process_instrument(instrument, market_data[instrument], state_manager, fallback)
        except Exception as e:
            logger.error(f"{instrument}: erreur orchestrateur — {e}", exc_info=True)
            result = OrchestratorResult(
                instrument=instrument, source="ERROR",
                skipped=True, skip_reason=f"Erreur interne: {e}",
                status_text=f"erreur : {e}",
            )

        results.append(result)
        tag = (
            " [SIGNAL]" if result.signal else
            " [CLOSED]" if result.position_closed else
            " [SKIP]" if result.skipped else ""
        )
        logger.info(f"{instrument}: source={result.source}{tag}")

    state_manager.save()
    return results
