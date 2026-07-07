# Boom & Crash Bot v2 — SMC Engine

Bot de signaux de trading pour les indices synthétiques Boom & Crash (Deriv).
Architecture SMC complète avec machine à états finis (FSM) et mémoire persistante.

## Instruments
Boom 500 / 900 / 1000 — Crash 500 / 900 / 1000

## Architecture
```
src/
├── main.py               # Orchestrateur principal
├── deriv_client.py       # WebSocket Deriv — données OHLC
├── technical_analysis.py # Détecteurs SMC (BOS, CHoCH, OB, FVG, Fib)
├── scenario_engine.py    # Moteur FSM — 3 scénarios SMC
├── state_machine.py      # Machine à états — mémoire entre runs
└── telegram_notifier.py  # Messages analyste vers Telegram
data/
└── market_states.json    # État FSM persistant (mis à jour à chaque run)
.github/workflows/
└── boom_crash_v2.yml     # GitHub Actions — 6 runs/jour
```

## Secrets GitHub requis
| Secret | Description |
|--------|-------------|
| `DERIV_APP_ID` | App ID Deriv (utiliser `1089` pour les tests) |
| `TELEGRAM_BOT_TOKEN` | Token du bot Telegram |
| `TELEGRAM_CHAT_ID` | ID du canal/chat de réception |

## Scénarios implémentés
1. **Continuation après BOS** — Pullback vers OB/FVG + confirmation M15
2. **Retournement** — CHoCH → fausse cassure → BOS inverse
3. **Fibonacci** — Niveaux 38.2/50/61.8 quand aucun OB/FVG visible

## Objectifs de prix
- TP1 : prochain swing M15 (court terme)
- TP2 : OB H1 structurel (moyen terme)
- TP3 : aimant de liquidité H4 — EQH/EQL/OB H4 (objectif long)

## Déploiement
1. Créer un repo GitHub et pousser ces fichiers
2. Ajouter les 3 secrets dans Settings → Secrets → Actions
3. Activer GitHub Actions
4. Tester via Actions → Run workflow
