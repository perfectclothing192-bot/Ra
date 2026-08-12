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

## Running continuously (PAPER mode, OANDA price data)

`run_loop.py` polls [OANDA](https://developer.oanda.com/rest-live-v20/introduction/)
for candle data and feeds it to the agent's implemented strategies
(Golden Pullback on XAUUSD, SMA Cluster Reclaim on USOIL) on a timer. It
always runs the agent in `TradingMode.PAPER` — no real orders are placed.

### 1. Get an OANDA practice account and API key

1. Sign up for a free practice account at https://www.oanda.com/demo-account/tpa/personal_finance.
2. In the OANDA dashboard, go to **Manage API Access** and generate a
   personal access token — this is `OANDA_API_KEY`.
3. Note your practice account ID (format `101-011-XXXXXXX-001`) — this is
   `OANDA_ACCOUNT_ID`.

### 2. Deploy to Railway

1. Create a account at https://railway.app and log in.
2. **New Project → Deploy from GitHub repo**, select this repository
   (`perfectclothing192-bot/Ra`) and the branch you want to deploy.
3. Railway will detect `Procfile` and offer a `worker` process — use that
   process type (not a web service; this app doesn't listen on a port).
4. Under the service's **Variables** tab, set:
   - `OANDA_API_KEY` — from step 1
   - `OANDA_ACCOUNT_ID` — from step 1
   - `OANDA_ENV` — `practice` (default; use `live` only with a funded live
     OANDA account)
   - Optional: `POLL_INTERVAL_SECONDS` (default `300`), `OANDA_GRANULARITY`
     (default `M15`), `ACCOUNT_EQUITY`, `RISK_PER_TRADE`, `DAILY_LOSS_LIMIT`
5. Deploy. Open the service's **Logs** tab to watch the agent poll for
   data, log signals, and open/close PAPER positions.

### Local run

```bash
export OANDA_API_KEY=...
export OANDA_ACCOUNT_ID=...
python run_loop.py
```
