"""
ai_analyzer.py
===============
Cœur de la décision — Architecture hybride IA du bot Boom & Crash v2.

RÔLE DANS L'ARCHITECTURE
-------------------------
C'est ICI que le jugement structurel SMC complet est effectué, à CHAQUE
run, par Gemini. Python ne fait que :
  1. Calculer des candidats de zones (OB/FVG/swings/Fibonacci) avec leurs
     PRIX EXACTS (jamais estimés visuellement).
  2. Générer le contexte texte + transmettre les 3 images larges
     (H4, H1, M15 — voir chart_renderer.py).
  3. Appeler Gemini avec un schéma de sortie STRICT et garanti par Google
     (response_schema), pour éliminer le risque de JSON mal formé.
  4. Résoudre les identifiants de zones renvoyés par Gemini vers leurs
     prix exacts d'origine — Gemini ne renvoie JAMAIS un prix inventé,
     seulement un identifiant de candidat.

Gemini, lui, a la responsabilité complète du jugement structurel :
tendance institutionnelle réelle (à partir de la vue large, pas
seulement des indicateurs locaux), scénario SMC applicable, zone la
plus pertinente parmi les candidats, et validation visuelle de la
confirmation M15. C'est la décision architecturale validée avec
l'utilisateur : Python fournit les données, l'IA juge.

GESTION DES PANNES
--------------------
Toute erreur (quota 429, timeout, réponse invalide) renvoie un
AnalyzerResult(success=False, ...) — jamais d'exception qui remonterait
jusqu'au run. C'est scenario_orchestrator.py (fichier suivant) qui
décidera alors de basculer sur fallback_engine.py pour cet instrument.

Un coupe-circuit de quota global est également intégré : dès la première
erreur 429 dans un run, tous les appels suivants du même run sont
court-circuités immédiatement (sans nouvelle requête réseau) pour éviter
de gaspiller du temps sur des appels voués à échouer.

Dépendances à ajouter à requirements.txt : google-genai, pydantic.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from technical_analysis import (
    Direction, TrendState,
    find_order_blocks, find_fair_value_gaps, find_swing_points,
    compute_fibonacci,
)

logger = logging.getLogger("ai_analyzer")

Candle = dict


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Modèle Gemini utilisé. "gemini-3.5-flash" est le modèle gratuit actuel
# pour les nouveaux comptes (lancé à Google I/O 2026, sans carte
# bancaire requise). Le catalogue de modèles Gemini change régulièrement
# et Google retire parfois d'anciens modèles pour les nouveaux comptes
# (c'est arrivé à "gemini-2.5-flash") — si ce modèle venait à son tour à
# ne plus être disponible, vérifie la liste à jour dans Google AI Studio
# et ajuste via la variable d'environnement GEMINI_MODEL, sans toucher
# au code.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# Niveau de raisonnement interne du modèle (si supporté). "HIGH" pour
# maximiser la qualité de jugement structurel — c'est un bot de trading,
# pas un chatbot, la profondeur de raisonnement prime sur la latence.
GEMINI_THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "HIGH")

# Timeout réseau par appel (millisecondes)
GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "45000"))

# Température basse : on veut un jugement discipliné et reproductible,
# pas de créativité — c'est de l'analyse structurelle, pas de la prose.
GEMINI_TEMPERATURE = 0.2

MAX_CANDIDATES_PER_KIND = 5


# ─────────────────────────────────────────────────────────────────────────────
# Méthodologie SMC — embarquée intégralement dans le prompt système à
# CHAQUE appel, pour que Gemini raisonne toujours avec la structure
# complète (jamais une version tronquée ou supposée mémorisée).
# ─────────────────────────────────────────────────────────────────────────────

SMC_METHODOLOGY = """
MÉTHODOLOGIE SMC (Smart Money Concepts) — à appliquer intégralement :

FONDAMENTAUX :
- BOS (Break of Structure) : le prix clôture au-delà d'un précédent
  sommet (tendance haussière) ou d'un précédent creux (tendance
  baissière). Preuve de continuation de tendance.
- CHoCH (Change of Character) : cassure d'un niveau opposé à la
  tendance en cours (ex: cassure d'un higher low en tendance haussière).
  C'est un AVERTISSEMENT de retournement possible, PAS une confirmation.
- OB (Order Block) : dernière bougie opposée à l'impulsion avant un BOS
  ou CHoCH. Zone d'intérêt institutionnel, aimant lors des pullbacks.
- FVG (Fair Value Gap) : vide de prix entre deux bougies non
  chevauchantes suite à une impulsion rapide. Attire le prix pour
  rééquilibrer le marché.
- Liquidité : zones de stops accumulés au-dessus des sommets / en
  dessous des creux. Les institutions chassent cette liquidité avant
  leurs vrais mouvements — c'est un élément CENTRAL de ton analyse de
  la vue large (H4/H1 sur ~500 bougies) : identifie les objectifs de
  liquidité majeurs (anciens extrêmes, equal highs/lows) que Python ne
  peut pas déduire seul depuis une vue étroite.

SCÉNARIO 1 — CONTINUATION :
BOS aligné avec la tendance H4 → cartographier OB (pullback profond) et
FVG (pullback court) → attendre le pullback → confirmation M15
obligatoire (bougie de rejet mèche ≥60% + volume ≥1.5× moyenne, OU
mini-CHoCH local) → entrée. Invalidation : cassure du swing point
d'origine du BOS.

SCÉNARIO 2 — RETOURNEMENT COMPLET :
Pullback qui casse le swing d'invalidation → CHoCH confirmé (alerte,
pas encore un retournement) → pullback de distribution/réaccumulation
vers OB/FVG opposé (fausse cassure optionnelle mais renforçante) → BOS
INVERSE obligatoire (seule confirmation absolue du retournement) →
nouveau pullback sur l'OB du BOS inverse → confirmation M15 → entrée.
Tant que le BOS inverse n'est pas formé, la tendance précédente peut
toujours reprendre — ne jamais traiter un simple CHoCH comme un
retournement validé.

SCÉNARIO 3 — FIBONACCI (absence d'OB/FVG) :
Retracements 0.382 / 0.5 / 0.618 du swing d'impulsion comme zones de
pullback. Même exigence de confirmation M15.

RÈGLES DE VALIDITÉ :
- Un pullback n'est JAMAIS terminé sans confirmation M15 explicite.
- Un CHoCH seul n'est pas un retournement.
- L'invalidation est structurelle et non négociable : si le prix casse
  le swing point d'invalidation sans la confirmation attendue, le setup
  est mort.
- La fausse cassure (scénario 2) est un bonus de conviction, pas une
  obligation.
""".strip()

DOMAIN_CONTEXT = """
CONTEXTE DOMAINE — Indices synthétiques Boom & Crash (Deriv) :
- Les indices "Boom" ont une dérive baissière de fond ponctuée de
  pics haussiers brefs et violents ("spikes"). Le biais institutionnel
  par défaut y est donc BAISSIER (on cherche des ventes en continuation).
- Les indices "Crash" ont une dérive haussière de fond ponctuée de pics
  baissiers brefs et violents. Le biais institutionnel par défaut y est
  donc HAUSSIER.
- Ce biais par défaut n'est qu'un point de départ statistique — s'il y a
  une preuve structurelle claire de retournement complet (CHoCH + BOS
  inverse confirmés, scénario 2), tu dois le signaler et l'utiliser
  malgré le biais par défaut. Ta lecture de la structure prime toujours
  sur ce biais générique.
- Le "volume" affiché n'existe pas réellement sur ces indices
  synthétiques — les graphiques que tu reçois n'en affichent
  volontairement pas. Base ton jugement de confirmation sur la forme des
  bougies (mèches, corps, mini-CHoCH), pas sur du volume.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Schéma de sortie structurée (garanti par Gemini via response_schema)
# ─────────────────────────────────────────────────────────────────────────────

class TendanceLue(str, Enum):
    HAUSSIER = "HAUSSIER"
    BAISSIER = "BAISSIER"
    RANGE = "RANGE"


class ScenarioSMC(str, Enum):
    CONTINUATION = "continuation"
    RETOURNEMENT = "retournement"
    FIBONACCI = "fibonacci"
    AUCUN = "aucun"


class TypeConfirmation(str, Enum):
    REJET_MECHE = "rejet_meche"
    MINI_CHOCH = "mini_choch"
    AUCUNE = "aucune"


class ActionRequise(str, Enum):
    ATTENDRE = "ATTENDRE"
    ACHETER = "ACHETER"
    VENDRE = "VENDRE"
    INVALIDE = "INVALIDE"


class GeminiVerdict(BaseModel):
    """
    Schéma strict de la réponse Gemini. Le champ `zone_retenue_id` DOIT
    correspondre exactement à un identifiant fourni dans la liste des
    candidats (jamais un prix libre) — c'est cette contrainte qui
    garantit qu'aucun prix invente ne peut se glisser dans le système.
    """
    analyse_structurelle: str = Field(
        description="Lecture SMC complète en français : tendance de fond "
        "vue sur les images larges, objectifs de liquidité identifiés, "
        "raisonnement du scénario retenu."
    )
    tendance_h4: TendanceLue
    tendance_h1: TendanceLue
    scenario_identifie: ScenarioSMC
    biais_institutionnel: TendanceLue = Field(
        description="Direction institutionnelle réelle retenue par ton "
        "analyse — peut différer du biais par défaut de l'instrument si "
        "un retournement complet est structurellement prouvé."
    )
    zone_retenue_id: Optional[str] = Field(
        default=None,
        description="Identifiant EXACT d'un candidat fourni dans le "
        "contexte (ex: 'OB_HAUSSIER_0'). Null si aucune zone candidate "
        "n'est pertinente.",
    )
    niveau_invalidation_id: Optional[str] = Field(
        default=None,
        description="Identifiant EXACT d'un candidat de swing fourni "
        "dans le contexte, servant de niveau d'invalidation structurelle.",
    )
    confirmation_m15: bool
    type_confirmation: TypeConfirmation
    prix_entree_estime: Optional[float] = Field(
        default=None,
        description="Estimation informative uniquement — NE SERA PAS "
        "utilisée pour le calcul final (recalculé précisément côté "
        "Python à partir de la bougie de confirmation).",
    )
    action_requise: ActionRequise
    commentaire_strategique: str = Field(
        description="Commentaire court destiné au message Telegram, "
        "explique la décision en langage trader."
    )
    confiance: int = Field(ge=0, le=100, description="Niveau de confiance 0-100")


def _normalize_enum_value(value, enum_cls) -> object:
    """
    Corrige les variations mineures que Gemini peut produire sur une
    valeur censée correspondre à un Enum strict : casse différente,
    accents manquants/en trop, espaces superflus. Retourne la valeur
    d'origine si aucune correspondance tolérante n'est trouvée (la
    validation Pydantic échouera alors normalement, avec un message
    clair dans les logs).
    """
    if not isinstance(value, str):
        return value

    def _strip_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    target = _strip_accents(value.strip().upper())
    for member in enum_cls:
        candidate = _strip_accents(member.value.strip().upper())
        if target == candidate:
            return member.value
    return value


def _normalize_verdict_dict(data: dict) -> dict:
    """
    Applique la tolérance de normalisation à tous les champs de type
    Enum du schéma, et corrige les types numériques inattendus (Gemini
    renvoie parfois un nombre en texte ou en flottant). Utilisée
    uniquement en filet de secours quand le parsing strict automatique
    (response.parsed) échoue.
    """
    enum_fields = {
        "tendance_h4": TendanceLue,
        "tendance_h1": TendanceLue,
        "biais_institutionnel": TendanceLue,
        "scenario_identifie": ScenarioSMC,
        "type_confirmation": TypeConfirmation,
        "action_requise": ActionRequise,
    }
    for field_name, enum_cls in enum_fields.items():
        if field_name in data:
            data[field_name] = _normalize_enum_value(data[field_name], enum_cls)

    if "confiance" in data and isinstance(data["confiance"], (float, str)):
        try:
            data["confiance"] = int(round(float(data["confiance"])))
        except (TypeError, ValueError):
            pass

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Candidats de zones (calculés en Python — prix garantis exacts)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    id: str
    kind: str          # "OB" | "FVG" | "SWING" | "FIB"
    direction: str      # "HAUSSIER" | "BAISSIER"
    low: Optional[float] = None
    high: Optional[float] = None
    price: Optional[float] = None
    label: str = ""

    def to_prompt_line(self) -> str:
        if self.low is not None and self.high is not None:
            return f"  [{self.id}] {self.label} : {self.low:.4f} – {self.high:.4f}"
        return f"  [{self.id}] {self.label} : {self.price:.4f}"


def _opposite(direction: Direction) -> Direction:
    return Direction.BEARISH if direction == Direction.BULLISH else Direction.BULLISH


def _dir_label(direction: Direction) -> str:
    return "HAUSSIER" if direction == Direction.BULLISH else "BAISSIER"


def build_candidates(
    h4: list[Candle],
    h1: list[Candle],
    bias: Direction,
) -> list[Candidate]:
    """
    Construit la liste des candidats de zones à partir des données H1
    (zones tactiques) et H4 (structure/Fibonacci). Les DEUX directions
    (biais par défaut + opposée) sont incluses, car un scénario de
    retournement (Scénario 2) nécessite des zones dans le sens inverse
    du biais par défaut.
    """
    candidates: list[Candidate] = []

    for direction in (bias, _opposite(bias)):
        dlabel = _dir_label(direction)

        obs = find_order_blocks(h1, direction, lookback=100)
        for i, ob in enumerate(obs[:MAX_CANDIDATES_PER_KIND]):
            candidates.append(Candidate(
                id=f"OB_{dlabel}_{i}", kind="OB", direction=dlabel,
                low=ob.low, high=ob.high,
                label=f"Order Block {dlabel} (H1)",
            ))

        fvgs = [f for f in find_fair_value_gaps(h1, lookback=100)
                if f.direction == direction and not f.mitigated]
        for i, fvg in enumerate(fvgs[:MAX_CANDIDATES_PER_KIND]):
            candidates.append(Candidate(
                id=f"FVG_{dlabel}_{i}", kind="FVG", direction=dlabel,
                low=fvg.low, high=fvg.high,
                label=f"Fair Value Gap {dlabel} (H1)",
            ))

    # Swings H1 : candidats de niveau d'invalidation (structure tactique)
    highs_h1, lows_h1 = find_swing_points(h1, window=3)
    for i, sp in enumerate(highs_h1[-MAX_CANDIDATES_PER_KIND:]):
        candidates.append(Candidate(
            id=f"SWING_HIGH_H1_{i}", kind="SWING", direction="HAUSSIER",
            price=sp.price, label="Swing high H1 (invalidation potentielle)",
        ))
    for i, sp in enumerate(lows_h1[-MAX_CANDIDATES_PER_KIND:]):
        candidates.append(Candidate(
            id=f"SWING_LOW_H1_{i}", kind="SWING", direction="BAISSIER",
            price=sp.price, label="Swing low H1 (invalidation potentielle)",
        ))

    # Swings H4 : structure majeure / objectifs de liquidité de fond
    highs_h4, lows_h4 = find_swing_points(h4, window=5)
    for i, sp in enumerate(highs_h4[-3:]):
        candidates.append(Candidate(
            id=f"SWING_HIGH_H4_{i}", kind="SWING", direction="HAUSSIER",
            price=sp.price, label="Swing high H4 majeur (structure de fond)",
        ))
    for i, sp in enumerate(lows_h4[-3:]):
        candidates.append(Candidate(
            id=f"SWING_LOW_H4_{i}", kind="SWING", direction="BAISSIER",
            price=sp.price, label="Swing low H4 majeur (structure de fond)",
        ))

    # Fibonacci sur le dernier swing H4 (scénario 3)
    if highs_h4 and lows_h4:
        sh = max(highs_h4, key=lambda x: x.price)
        sl = min(lows_h4, key=lambda x: x.price)
        fib = compute_fibonacci(sh.price, sl.price, bias)
        if fib:
            for name, val in (
                ("382", fib.fib_382), ("500", fib.fib_500), ("618", fib.fib_618),
            ):
                candidates.append(Candidate(
                    id=f"FIB_{name}", kind="FIB", direction=_dir_label(bias),
                    price=val, label=f"Fibonacci {name[0]}.{name[1:]}%",
                ))

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Construction du prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_text(
    instrument: str,
    bias: Direction,
    current_price: float,
    trend_h4: TrendState,
    trend_h1: TrendState,
    candidates: list[Candidate],
    previous_context: Optional[dict] = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    lines = [
        f"INSTRUMENT : {instrument}",
        f"HORODATAGE DU RUN : {now}",
        f"PRIX ACTUEL EXACT : {current_price:.4f}",
        f"BIAIS PAR DÉFAUT DE L'INSTRUMENT : {_dir_label(bias)} "
        f"(voir contexte domaine — reste indicatif, pas une contrainte)",
        "",
        f"Indicateurs Python (informatifs, ta lecture visuelle de la vue "
        f"large prime) :",
        f"  Tendance EMA20/50 H4 : {trend_h4.direction.value} "
        f"(range={trend_h4.is_range})",
        f"  Tendance EMA20/50 H1 : {trend_h1.direction.value} "
        f"(range={trend_h1.is_range})",
        "",
        "CANDIDATS DE ZONES (prix exacts calculés par Python — tu dois "
        "choisir PARMI ces identifiants, jamais inventer un prix) :",
    ]

    if candidates:
        for c in candidates:
            lines.append(c.to_prompt_line())
    else:
        lines.append("  (aucun candidat détecté sur cette fenêtre)")

    if previous_context:
        lines.append("")
        lines.append("CONTINUITÉ AVEC LE RUN PRÉCÉDENT (mémoire FSM) :")
        for k, v in previous_context.items():
            lines.append(f"  {k} : {v}")

    lines.append("")
    lines.append(
        "IMAGES FOURNIES DANS L'ORDRE : (1) H4 vue large, (2) H1 vue "
        "large, (3) M15 zoom sur la zone active. Utilise (1) et (2) pour "
        "juger la tendance institutionnelle réelle et les objectifs de "
        "liquidité. Utilise (3) pour juger la confirmation M15."
    )

    return "\n".join(lines)


def _build_system_instruction() -> str:
    return (
        "Tu es un analyste institutionnel spécialisé en Smart Money "
        "Concepts (SMC/ICT), en charge de la validation de setups de "
        "trading sur indices synthétiques Boom & Crash.\n\n"
        + SMC_METHODOLOGY
        + "\n\n"
        + DOMAIN_CONTEXT
        + "\n\n"
        "RÈGLES STRICTES :\n"
        "- Tu ne dois JAMAIS estimer ou inventer un prix. Pour toute "
        "zone ou niveau d'invalidation, utilise EXCLUSIVEMENT les "
        "identifiants fournis dans le contexte texte.\n"
        "- Si aucun candidat fourni ne correspond à ce que tu vois, "
        "laisse le champ correspondant à null plutôt que d'inventer.\n"
        "- confirmation_m15 doit être true UNIQUEMENT si tu vois "
        "clairement une bougie de rejet (mèche ≥60% + clôture au-delà du "
        "milieu du range) OU un mini-CHoCH sur l'image M15.\n"
        "- action_requise ne peut être ACHETER ou VENDRE QUE si "
        "confirmation_m15 est true ET que le scénario est cohérent avec "
        "la structure large que tu observes sur H4/H1.\n"
        "- Utilise la vue large (H4/H1, ~500 bougies) en priorité pour "
        "juger la tendance institutionnelle et les objectifs de "
        "liquidité — ne te fie pas uniquement aux indicateurs locaux "
        "fournis en texte, ils sont informatifs seulement."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Résultat de l'analyse
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalyzerResult:
    instrument: str
    success: bool
    verdict: Optional[GeminiVerdict] = None
    resolved_zone: Optional[Candidate] = None
    resolved_invalidation: Optional[Candidate] = None
    candidates_sent: list[Candidate] = field(default_factory=list)
    error: Optional[str] = None
    latency_seconds: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Coupe-circuit de quota (partagé pour tout le run)
# ─────────────────────────────────────────────────────────────────────────────

_quota_exhausted = False


def reset_quota_state() -> None:
    """À appeler par l'orchestrateur au début de CHAQUE run."""
    global _quota_exhausted
    _quota_exhausted = False


def is_quota_exhausted() -> bool:
    return _quota_exhausted


def _mark_quota_exhausted() -> None:
    global _quota_exhausted
    if not _quota_exhausted:
        logger.error(
            "Quota Gemini épuisé (429) — bascule fallback pour tous les "
            "instruments restants de ce run."
        )
    _quota_exhausted = True


# ─────────────────────────────────────────────────────────────────────────────
# Client Gemini (initialisation paresseuse)
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional["genai.Client"] = None


def _get_client() -> "genai.Client":
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY manquant dans l'environnement")
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def analyze_instrument(
    instrument: str,
    bias: Direction,
    current_price: float,
    trend_h4: TrendState,
    trend_h1: TrendState,
    h4: list[Candle],
    h1: list[Candle],
    charts,  # InstrumentCharts (chart_renderer.py) — typé en commentaire
    # pour éviter une dépendance circulaire d'import
    previous_context: Optional[dict] = None,
) -> AnalyzerResult:
    """
    Interroge Gemini pour un jugement structurel SMC complet sur un
    instrument, à partir des 3 images larges + du contexte texte.

    Ne lève jamais d'exception : toute erreur est capturée et renvoyée
    dans AnalyzerResult(success=False, error=...), pour permettre à
    l'orchestrateur de basculer proprement sur le fallback Python.
    """
    if is_quota_exhausted():
        return AnalyzerResult(
            instrument=instrument, success=False,
            error="Quota Gemini déjà épuisé pour ce run — appel court-circuité",
        )

    if charts is None or not charts.is_complete():
        return AnalyzerResult(
            instrument=instrument, success=False,
            error="Jeu d'images incomplet (H4/H1/M15) — analyse IA impossible",
        )

    candidates = build_candidates(h4, h1, bias)

    context_text = _build_context_text(
        instrument, bias, current_price, trend_h4, trend_h1,
        candidates, previous_context,
    )
    system_instruction = _build_system_instruction()

    contents = [
        types.Part.from_text(text=context_text),
        types.Part.from_text(text="Image 1/3 — H4 vue large :"),
        types.Part.from_bytes(
            data=base64.b64decode(charts.h4.base64_data),
            mime_type=charts.h4.mime_type,
        ),
        types.Part.from_text(text="Image 2/3 — H1 vue large :"),
        types.Part.from_bytes(
            data=base64.b64decode(charts.h1.base64_data),
            mime_type=charts.h1.mime_type,
        ),
        types.Part.from_text(text="Image 3/3 — M15 zoom (zone active) :"),
        types.Part.from_bytes(
            data=base64.b64decode(charts.m15.base64_data),
            mime_type=charts.m15.mime_type,
        ),
    ]

    config_kwargs = dict(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=GeminiVerdict,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=2000,
    )

    start = time.monotonic()
    try:
        client = _get_client()

        # Le paramètre thinking_level n'est pas supporté par toutes les
        # versions du SDK / tous les modèles : on tente avec, on
        # retombe sans si le SDK le rejette, plutôt que de faire
        # échouer tout l'appel pour un paramètre de confort.
        try:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=GEMINI_THINKING_LEVEL
                ),
                **config_kwargs,
            )
        except Exception:
            config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        latency = time.monotonic() - start

        verdict: Optional[GeminiVerdict] = getattr(response, "parsed", None)

        if verdict is None:
            # Le parsing strict automatique a échoué (schéma non respecté à
            # la lettre) — on tente une validation manuelle tolérante avant
            # d'abandonner. Ça récupère les cas bénins (casse, accents,
            # nombre en texte) sans jamais accepter une valeur hors schéma.
            raw_text = getattr(response, "text", None)
            if raw_text:
                try:
                    data = json.loads(raw_text)
                    data = _normalize_verdict_dict(data)
                    verdict = GeminiVerdict.model_validate(data)
                    logger.info(
                        f"{instrument}: verdict Gemini récupéré après "
                        f"normalisation manuelle du schéma"
                    )
                except Exception as parse_err:
                    logger.error(
                        f"{instrument}: échec de validation du schéma Gemini "
                        f"— {parse_err}\nRéponse brute (tronquée à 1500 "
                        f"caractères) : {raw_text[:1500]}"
                    )
            else:
                logger.error(
                    f"{instrument}: réponse Gemini vide ou sans texte exploitable"
                )

        if verdict is None:
            return AnalyzerResult(
                instrument=instrument, success=False,
                candidates_sent=candidates,
                error="Réponse Gemini reçue mais non conforme au schéma attendu",
                latency_seconds=time.monotonic() - start,
            )

        # Garde-fou de cohérence (cahier des charges) : une action ne
        # peut être émise sans confirmation M15 explicite.
        if verdict.action_requise in (ActionRequise.ACHETER, ActionRequise.VENDRE) \
                and not verdict.confirmation_m15:
            logger.warning(
                f"{instrument}: Gemini a renvoyé {verdict.action_requise.value} "
                f"sans confirmation_m15 — forcé à ATTENDRE (incohérence corrigée)"
            )
            verdict.action_requise = ActionRequise.ATTENDRE

        candidates_by_id = {c.id: c for c in candidates}
        resolved_zone = candidates_by_id.get(verdict.zone_retenue_id) \
            if verdict.zone_retenue_id else None
        resolved_invalidation = candidates_by_id.get(verdict.niveau_invalidation_id) \
            if verdict.niveau_invalidation_id else None

        if verdict.zone_retenue_id and resolved_zone is None:
            logger.warning(
                f"{instrument}: Gemini a référencé un ID de zone inconnu "
                f"'{verdict.zone_retenue_id}' — zone non résolue"
            )

        return AnalyzerResult(
            instrument=instrument, success=True,
            verdict=verdict,
            resolved_zone=resolved_zone,
            resolved_invalidation=resolved_invalidation,
            candidates_sent=candidates,
            latency_seconds=latency,
        )

    except Exception as e:
        latency = time.monotonic() - start
        error_text = str(e)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text.upper() \
                or "quota" in error_text.lower():
            _mark_quota_exhausted()
        logger.error(f"{instrument}: erreur appel Gemini — {e}", exc_info=True)
        return AnalyzerResult(
            instrument=instrument, success=False,
            candidates_sent=candidates,
            error=error_text,
            latency_seconds=latency,
        )
