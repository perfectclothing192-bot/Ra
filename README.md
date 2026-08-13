# Ramesh's Automated Trading Agent

Multi-strategy, regime-adaptive, risk-managed trading agent.

## Strategies

- **Gold (XAUUSD)** — Golden Pullback (EMA 200/50 + Smart Money Concepts)
- **Oil (USOIL)** — SMA Cluster Reclaim (100/200 + liquidity sweep)
- **Forex (GBPUSD, EURUSD)** — Correlation Hedge (rolling spread z-score, paired mean-reversion)
- **Gold (XAUUSD) scalping** — EMA Ribbon Momentum Scalp (EMA 5/13/50 + RSI filter), run by a
  separate `ScalpingTradingAgent` on fast M1 candles — see below

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
for candle data and feeds it to the agent's implemented strategies (Golden
Pullback on XAUUSD, SMA Cluster Reclaim on USOIL, Correlation Hedge on the
GBPUSD/EURUSD pair) on a timer. It always runs the agent in
`TradingMode.PAPER` — no real orders are placed.

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

## Running the scalping agent (PAPER mode, Gold/XAUUSD)

`run_scalp_loop.py` runs a separate `ScalpingTradingAgent` (in
`scalping_agent.py`) dedicated to fast, short-timeframe scalping on XAUUSD.
It's a standalone process from `run_loop.py`, with its own OANDA candle
granularity, poll interval, risk settings, and state file, so it can poll
much faster (M1 candles, default every 60s) without changing the swing
strategies' M15/300s cadence.

Strategy: **EMA Ribbon Momentum Scalp** — enters on a fresh EMA 5/EMA 13
crossover in the direction of the EMA 50 slope, filtered by RSI to avoid
chasing an already-exhausted move, with a tight ATR-based stop/target (0.6x
/ 1.0x ATR) sized for a quick in-and-out trade rather than a multi-bar swing.

Environment variables (all optional, shown with defaults):
- `SCALP_GRANULARITY` (`M1`), `SCALP_POLL_INTERVAL_SECONDS` (`60`)
- `SCALP_ACCOUNT_EQUITY` (`150000`), `SCALP_RISK_PER_TRADE` (`0.0025`, i.e.
  0.25% — smaller than the swing agent's 1% given the higher trade
  frequency), `SCALP_DAILY_LOSS_LIMIT` (`0.03`, i.e. 3%)
- `SCALP_STATE_FILE` (`scalp_state.json`)

It reuses `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`, and `OANDA_ENV` from the main
setup. Always runs in `TradingMode.PAPER` — no real orders are placed.

Built to stay up unattended for continuous 24/7 operation: OANDA requests
retry with backoff on transient network/rate-limit/server errors, a failed
poll cycle is logged and skipped rather than crashing the process, and
polling is skipped cleanly over the weekend FX/Gold market closure instead
of erroring every cycle.

```bash
export OANDA_API_KEY=...
export OANDA_ACCOUNT_ID=...
python run_scalp_loop.py
```

On Railway, deploy this as a second service pointed at the same repo/branch,
using the `scalper` process type from `Procfile` (the existing swing
strategies keep running as the `worker` process type in their own service).
