# Ramesh's Automated Trading Agent

Multi-strategy, regime-adaptive, risk-managed trading agent.

## Strategies

- **Gold (XAUUSD)** — Golden Pullback (EMA 200/50 + Smart Money Concepts)
- **Oil (USOIL)** — SMA Cluster Reclaim (100/200 + liquidity sweep)
- **Forex (GBPUSD, EURUSD)** — correlation hedge (planned)

## Risk management

- Max 1% risk per trade
- 5% daily loss halt
- Equity floor / circuit breakers

## Modes

- `TradingMode.PAPER` (default) — fully simulated, no live orders
- `TradingMode.LIVE` — sends orders to IG Australia via a webhook

## Setup

```bash
pip install -r requirements.txt
python trading_agent.py
```

For live trading, set `ig_api_key` on the agent and export `IG_WEBHOOK_URL`
pointing at your execution webhook before switching `mode` to `TradingMode.LIVE`.
