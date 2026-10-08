# nba_reb_line

Kalshi NBA player-rebound (`KXNBAREB`) pricing and trading engine.

- `SYSTEM.md` — how it works and why (model bake-off, model vs market, ported MLB lessons)
- `WORKFLOW.md` — commands, cold start, game-day and weekly loops

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python run_pipeline.py etl
.venv/bin/python run_pipeline.py train
.venv/bin/python run_pipeline.py scan            # dry run
.venv/bin/python -m pytest -q
```

Sister project: `mlb_tb_line` (Kalshi MLB total bases). Most of the trading stack here
was ported from it on 2026-10-08.
