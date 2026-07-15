"""
chart_renderer.py
==================
Génération des images de graphique pour l'analyse visuelle par l'IA
(Gemini) — Architecture hybride IA du bot Boom & Crash v2.

RÔLE DANS L'ARCHITECTURE
-------------------------
Ce module transforme les bougies brutes (deriv_client.py) en images PNG
prêtes à être envoyées à Gemini. Il ne fait AUCUNE interprétation SMC
(pas de BOS/CHoCH/OB/FVG dessinés). Il ne trace que des éléments
objectifs et non-biaisants :

  - EMA 20 / EMA 50 (tendance mathématique pure)
  - Pivots de swing (sommets/creux locaux détectés par comparaison de
    voisinage — un fait géométrique, pas une interprétation)
  - Ligne horizontale du prix actuel (ancrage avec le texte du prompt)

POURQUOI NE PAS DESSINER LES ZONES OB/FVG SUR L'IMAGE
--------------------------------------------------------
Si Python pré-dessine les zones qu'il juge pertinentes, Gemini serait
tenté de simplement valider ce qui est déjà tracé plutôt que de former
son propre jugement structurel à partir de la vue d'ensemble. Cela
viderait de son sens le choix architectural retenu (l'IA juge toute la
structure SMC elle-même). Les zones candidates calculées par
technical_analysis.py sont transmises séparément, en TEXTE avec leurs
prix exacts, dans ai_analyzer.py — l'image sert au jugement visuel
global, le texte garantit la précision numérique.

POURQUOI PAS DE PANNEAU DE VOLUME
-----------------------------------
Deriv ne fournit pas de volume réel sur les indices synthétiques
(deriv_client.py utilise un proxy basé sur l'amplitude des bougies).
L'afficher comme un vrai volume d'échange induirait l'IA en erreur.

DEUX MODES DE RENDU
---------------------
  - "wide" (H4, H1) : jusqu'à 500 bougies, vue institutionnelle large,
    pivots de swing filtrés pour ne garder que les majeurs.
  - "zoom" (M15) : fenêtre resserrée sur la zone active, pivots plus
    fins (utile pour repérer un mini-CHoCH visuellement).

Dépendances à ajouter à requirements.txt : mplfinance, pandas,
matplotlib, Pillow.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # OBLIGATOIRE : environnement headless (GitHub Actions)
import matplotlib.pyplot as plt

try:
    import mplfinance as mpf
    import pandas as pd
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "chart_renderer.py nécessite mplfinance, pandas et Pillow. "
        "Ajoute-les à requirements.txt : mplfinance, pandas, matplotlib, Pillow"
    ) from e

from technical_analysis import compute_ema, find_swing_points

logger = logging.getLogger("chart_renderer")

Candle = dict


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Dossier de sortie des images (nettoyé à chaque run — voir clear_chart_directory)
CHART_OUTPUT_DIR = os.environ.get(
    "CHART_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "boom_crash_charts")
)

# Nombre de bougies M15 affichées en mode zoom (~30h de contexte immédiat)
ZOOM_CANDLE_COUNT = 120

# En dessous de ce nombre de bougies, on ne génère pas de graphique
# (image non exploitable, il vaut mieux basculer sur le fallback Python)
MIN_CANDLES_FOR_CHART = 30

# Fenêtre de détection des pivots de swing :
# - "wide" (H4/H1) : fenêtre large → ne garde que les pivots MAJEURS,
#   évite de saturer une image de 500 bougies de petits marqueurs.
# - "zoom" (M15) : fenêtre fine → détecte les micro-pivots utiles au
#   repérage visuel d'un mini-CHoCH.
SWING_WINDOW_WIDE = 5
SWING_WINDOW_ZOOM = 2

EMA_FAST = 20
EMA_SLOW = 50

WIDE_FIGSIZE = (20, 9)
ZOOM_FIGSIZE = (16, 9)
DPI = 110

# Config par timeframe : mode de rendu + nombre de bougies affichées
# (None = toutes les bougies disponibles)
CHART_TIMEFRAME_CONFIG: dict[str, dict] = {
    "H4":  {"mode": "wide", "display_count": None},
    "H1":  {"mode": "wide", "display_count": None},
    "M15": {"mode": "zoom", "display_count": ZOOM_CANDLE_COUNT},
}


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChartImage:
    """Une image de graphique générée, prête pour l'envoi à l'IA."""
    instrument: str
    timeframe: str
    mode: str
    file_path: str
    base64_data: str
    mime_type: str = "image/png"
    width_px: int = 0
    height_px: int = 0
    candle_count_displayed: int = 0
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    current_price: float = 0.0


@dataclass
class InstrumentCharts:
    """Les 3 graphiques (H4/H1/M15) générés pour un instrument."""
    instrument: str
    h4: Optional[ChartImage] = None
    h1: Optional[ChartImage] = None
    m15: Optional[ChartImage] = None
    errors: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """True si les 3 graphiques nécessaires au jugement IA sont présents."""
        return self.h4 is not None and self.h1 is not None and self.m15 is not None


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(instrument: str) -> str:
    return instrument.lower().replace(" ", "_")


def _to_dataframe(candles: list[Candle]) -> "pd.DataFrame":
    """Convertit une liste de Candle en DataFrame compatible mplfinance."""
    return pd.DataFrame(
        {
            "Open":  [c["open"]  for c in candles],
            "High":  [c["high"]  for c in candles],
            "Low":   [c["low"]   for c in candles],
            "Close": [c["close"] for c in candles],
        },
        index=pd.DatetimeIndex(
            [c["time"].replace(tzinfo=None) for c in candles], name="Date"
        ),
    )


def _aligned_ema(candles: list[Candle], period: int) -> list[float]:
    """
    Calcule l'EMA sur la série COMPLÈTE fournie (garantit une période de
    warm-up correcte), puis retourne un tableau de la même longueur que
    `candles`, avec NaN pour les indices trop précoces pour avoir une
    valeur d'EMA valide. C'est cet alignement qui permet ensuite de
    découper (zoom) sans perdre la précision de l'EMA.
    """
    closes = [c["close"] for c in candles]
    ema_vals = compute_ema(closes, period)
    n = len(candles)
    aligned = [float("nan")] * n
    if ema_vals:
        start = period - 1
        for i, v in enumerate(ema_vals):
            idx = start + i
            if idx < n:
                aligned[idx] = v
    return aligned


def _aligned_swing_markers(
    candles: list[Candle], window: int
) -> tuple[list[float], list[float]]:
    """
    Détecte les pivots de swing sur la série COMPLÈTE fournie, retourne
    deux tableaux (highs, lows) de la même longueur que `candles`, avec
    NaN partout sauf à l'indice du pivot (valeur = prix du pivot).
    """
    highs, lows = find_swing_points(candles, window=window)
    n = len(candles)
    high_marks = [float("nan")] * n
    low_marks = [float("nan")] * n
    for sp in highs:
        if 0 <= sp.index < n:
            high_marks[sp.index] = sp.price
    for sp in lows:
        if 0 <= sp.index < n:
            low_marks[sp.index] = sp.price
    return high_marks, low_marks


def _has_data(series: list[float]) -> bool:
    return any(not math.isnan(v) for v in series)


def _build_style():
    """Style visuel haute lisibilité (fond clair, fort contraste haussier/baissier)."""
    market_colors = mpf.make_marketcolors(
        up="#089981", down="#F23645",
        edge="inherit", wick="inherit",
    )
    return mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=market_colors,
        gridstyle="--",
        gridcolor="#e0e0e0",
        facecolor="white",
        figcolor="white",
        y_on_right=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rendu d'un graphique unique
# ─────────────────────────────────────────────────────────────────────────────

def render_chart(
    candles: list[Candle],
    instrument: str,
    timeframe: str,
    mode: str,
    output_dir: str = CHART_OUTPUT_DIR,
    display_count: Optional[int] = None,
    reference_zones: Optional[list[dict]] = None,
) -> Optional[ChartImage]:
    """
    Génère une image de graphique pour un instrument/timeframe donné.

    Args:
        candles: série complète de bougies (utilisée pour calculer EMA/
            swings avec un warm-up correct, même si l'affichage est
            ensuite recadré en mode zoom).
        mode: "wide" ou "zoom" — détermine la fenêtre de détection des
            pivots et la taille de figure.
        display_count: nombre de bougies affichées (None = toutes).
            Le calcul EMA/swings reste basé sur la série complète.
        reference_zones: liste optionnelle de zones à surligner, ex.
            [{"low": 1.095, "high": 1.0965, "label": "Zone mémorisée",
            "color": "#FFC107"}]. Non utilisé par défaut — réservé pour
            un usage futur ponctuel, afin de ne pas biaiser le jugement
            structurel de l'IA sur les vues larges H4/H1.

    Retourne None si les données sont insuffisantes ou si le rendu échoue
    (erreur loguée, jamais levée — un échec de rendu ne doit jamais
    interrompre l'analyse des autres instruments).
    """
    if len(candles) < MIN_CANDLES_FOR_CHART:
        logger.warning(
            f"{instrument} {timeframe}: seulement {len(candles)} bougies "
            f"(min {MIN_CANDLES_FOR_CHART}) — graphique non généré"
        )
        return None

    try:
        swing_window = SWING_WINDOW_WIDE if mode == "wide" else SWING_WINDOW_ZOOM

        ema20_full = _aligned_ema(candles, EMA_FAST)
        ema50_full = _aligned_ema(candles, EMA_SLOW)
        high_marks_full, low_marks_full = _aligned_swing_markers(candles, swing_window)

        if display_count and display_count < len(candles):
            offset = len(candles) - display_count
            display_candles = candles[offset:]
            ema20 = ema20_full[offset:]
            ema50 = ema50_full[offset:]
            high_marks = high_marks_full[offset:]
            low_marks = low_marks_full[offset:]
        else:
            display_candles = candles
            ema20, ema50 = ema20_full, ema50_full
            high_marks, low_marks = high_marks_full, low_marks_full

        if len(display_candles) < MIN_CANDLES_FOR_CHART:
            logger.warning(
                f"{instrument} {timeframe}: fenêtre d'affichage trop courte "
                f"après recadrage — graphique non généré"
            )
            return None

        df = _to_dataframe(display_candles)
        current_price = display_candles[-1]["close"]

        addplots = []
        if _has_data(ema20):
            addplots.append(mpf.make_addplot(
                ema20, color="#2962FF", width=1.3, panel=0
            ))
        if _has_data(ema50):
            addplots.append(mpf.make_addplot(
                ema50, color="#FF6D00", width=1.3, panel=0
            ))
        if _has_data(high_marks):
            addplots.append(mpf.make_addplot(
                high_marks, type="scatter", markersize=70, marker="v",
                color="#D50000", panel=0
            ))
        if _has_data(low_marks):
            addplots.append(mpf.make_addplot(
                low_marks, type="scatter", markersize=70, marker="^",
                color="#00A152", panel=0
            ))

        figsize = WIDE_FIGSIZE if mode == "wide" else ZOOM_FIGSIZE
        time_start = display_candles[0]["time"]
        time_end = display_candles[-1]["time"]

        title = (
            f"{instrument} — {timeframe} ({mode.upper()}) — "
            f"{time_start.strftime('%d/%m/%Y %H:%M')} → "
            f"{time_end.strftime('%d/%m/%Y %H:%M')} UTC\n"
            f"Prix actuel: {current_price:.4f}  |  "
            f"{len(display_candles)} bougies affichées"
        )

        fig, axes = mpf.plot(
            df,
            type="candle",
            style=_build_style(),
            addplot=addplots if addplots else None,
            figsize=figsize,
            returnfig=True,
            title=title,
            hlines=dict(
                hlines=[current_price],
                colors=["#616161"],
                linestyle="--",
                linewidths=[1.0],
            ),
            datetime_format="%d-%b" if mode == "wide" else "%d-%b %H:%M",
            xrotation=20,
            tight_layout=True,
            volume=False,
        )

        ax = axes[0]
        if reference_zones:
            for zone in reference_zones:
                ax.axhspan(
                    zone["low"], zone["high"],
                    color=zone.get("color", "#FFC107"), alpha=0.15,
                )
                if zone.get("label"):
                    ax.text(
                        0.01, zone["high"], zone["label"],
                        transform=ax.get_yaxis_transform(),
                        fontsize=8, color="#795548", va="bottom",
                    )

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{_slugify(instrument)}_{timeframe}_{mode}.png"
        file_path = os.path.join(output_dir, filename)
        fig.savefig(file_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        b64_data = base64.b64encode(raw_bytes).decode("utf-8")

        with Image.open(file_path) as img:
            width_px, height_px = img.size

        return ChartImage(
            instrument=instrument,
            timeframe=timeframe,
            mode=mode,
            file_path=file_path,
            base64_data=b64_data,
            width_px=width_px,
            height_px=height_px,
            candle_count_displayed=len(display_candles),
            time_range_start=time_start,
            time_range_end=time_end,
            current_price=current_price,
        )

    except Exception as e:
        logger.error(
            f"{instrument} {timeframe}: échec du rendu graphique — {e}",
            exc_info=True,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration multi-timeframe / multi-instrument
# ─────────────────────────────────────────────────────────────────────────────

def render_instrument_charts(
    instrument: str,
    data: dict[str, list[Candle]],
    output_dir: str = CHART_OUTPUT_DIR,
    m15_reference_zones: Optional[list[dict]] = None,
) -> InstrumentCharts:
    """
    Génère les 3 graphiques (H4 large, H1 large, M15 zoom) d'un
    instrument. Chaque graphique est généré indépendamment : l'échec de
    l'un n'empêche pas la génération des autres.
    """
    results: dict[str, Optional[ChartImage]] = {}
    errors: list[str] = []

    for tf, cfg in CHART_TIMEFRAME_CONFIG.items():
        candles = data.get(tf, [])
        zones = m15_reference_zones if tf == "M15" else None
        img = render_chart(
            candles, instrument, tf, cfg["mode"],
            output_dir=output_dir,
            display_count=cfg["display_count"],
            reference_zones=zones,
        )
        results[tf.lower()] = img
        if img is None:
            errors.append(f"{tf}: graphique non généré (données insuffisantes ou erreur)")

    return InstrumentCharts(
        instrument=instrument,
        h4=results.get("h4"),
        h1=results.get("h1"),
        m15=results.get("m15"),
        errors=errors,
    )


def render_all_charts(
    market_data: dict[str, dict[str, list[Candle]]],
    output_dir: str = CHART_OUTPUT_DIR,
    clean_before: bool = True,
) -> dict[str, InstrumentCharts]:
    """
    Génère les graphiques pour tous les instruments d'un run.

    Robustesse : l'échec total de rendu pour UN instrument (exception
    imprévue) n'interrompt jamais le traitement des autres — l'instrument
    concerné recevra simplement un InstrumentCharts incomplet, ce qui
    déclenchera le fallback Python pour lui seul (géré en aval par
    scenario_orchestrator.py).
    """
    if clean_before:
        clear_chart_directory(output_dir)

    results: dict[str, InstrumentCharts] = {}
    for instrument, data in market_data.items():
        try:
            results[instrument] = render_instrument_charts(instrument, data, output_dir)
        except Exception as e:
            logger.error(
                f"{instrument}: échec total de génération des graphiques — {e}",
                exc_info=True,
            )
            results[instrument] = InstrumentCharts(
                instrument=instrument, errors=[str(e)]
            )

    ready = sum(1 for c in results.values() if c.is_complete())
    logger.info(
        f"Graphiques générés: {ready}/{len(results)} instruments avec jeu complet H4+H1+M15"
    )
    return results


def clear_chart_directory(output_dir: str = CHART_OUTPUT_DIR) -> None:
    """Vide le dossier d'images avant un nouveau run (évite toute confusion inter-run)."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
