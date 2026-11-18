"""
TradingAgents (TauricResearch) multi-agent LLM strategy wrapper.

*** CANDIDATE, NOT WIRED INTO THE LIVE LOOP. *** run_loop.py's STRATEGIES
dict does not reference anything here yet - nothing in this file runs
during a normal poll cycle until that changes deliberately, after this has
actually been tested against a live LLM API key.

Unlike every other strategy in trading_agent.py (a technical-indicator
calculation over OANDA candles, effectively free and instant), this wraps
https://github.com/TauricResearch/TradingAgents - a multi-agent LLM
pipeline (fundamentals/sentiment/news/technical analysts -> bull/bear
researcher debate -> trader -> risk team -> portfolio manager) that makes
one real LLM-backed decision call per (asset, date).

Things a caller MUST respect that don't apply to any other strategy here:
- Costs real LLM API tokens per call.
- Takes real wall-clock time (several LLM round-trips with debate rounds),
  not an instant indicator calc.
- Produces at most one genuinely new decision per UTC calendar day per
  asset - _load_cache()/_save_cache() below enforce that via CACHE_FILE,
  so even if this gets called on every hourly/backup poll cycle, the LLM
  itself is only actually invoked once per asset per day.
- Requires OPENAI_API_KEY (or another provider's key, with llm_provider
  changed in config below) - see tradingagents.llm_clients.api_key_env for
  the full provider -> env-var mapping. Without a working key,
  get_llm_signal() returns None rather than raising, so a misconfigured
  environment degrades to "no signal" (the caller should treat that as
  HOLD) instead of crashing the whole poll cycle.

Asset-type handling: TradingAgents only distinguishes "stock" vs "crypto"
(see its cli/prompts.py:detect_asset_type) - every FX/commodity symbol
Ra trades (XAUUSD, EURUSD, GBPUSD, USOIL, XAGUSD) falls through to the
"stock" pipeline by that classification, which would otherwise include a
Fundamentals/SEC-filings analyst that has nothing to analyze (there's no
10-K for gold or EURUSD). FOREX_COMMODITY_ASSETS below is used to drop
"fundamentals" from selected_analysts for those by hand, leaving
market/social/news - the only analysts that can produce a real opinion on
an FX pair or a commodity future. BTCUSD gets fundamentals dropped for
free by TradingAgents' own crypto asset-type filtering, but is listed
there too for clarity (harmless - "fundamentals" just isn't added twice).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

CACHE_FILE = os.environ.get("LLM_STRATEGY_CACHE_FILE", "llm_decision_cache.json")

# Maps Ra's own synthetic asset keys to the ticker TradingAgents' symbol
# resolver expects (tradingagents.dataflows.symbols.normalize_symbol).
# Several of Ra's keys are synthetic variants sharing one instrument (e.g.
# XAUUSD_FIB/XAUUSD_M15/XAUUSD_ORB all trade XAU_USD - see
# ASSET_TO_OANDA_INSTRUMENT in oanda_client.py); all of them resolve to the
# same base ticker here, since the underlying asset the LLM is asked to
# analyze doesn't change - only Ra's own position-sizing/slot bookkeeping
# differs between the synthetic keys, which this module has no part in.
ASSET_TO_TRADINGAGENTS_TICKER = {
    "XAUUSD": "XAUUSD",
    "XAUUSD_M15": "XAUUSD",
    "XAUUSD_FIB": "XAUUSD",
    "XAUUSD_ORB": "XAUUSD",
    "USOIL": "USOIL",
    "USOIL_FIB": "USOIL",
    "GBPUSD": "GBPUSD",
    "GBPUSD_FIB": "GBPUSD",
    "EURUSD": "EURUSD",
    "EURUSD_FIB": "EURUSD",
    "BTCUSD": "BTCUSD",
    "BTCUSD_FIB": "BTCUSD",
    "XAGUSD": "XAGUSD",
    "XAGUSD_RSI2": "XAGUSD",
}

# See module docstring - these need "fundamentals" dropped from
# selected_analysts by hand since TradingAgents' own asset-type detection
# only does that automatically for crypto.
FOREX_COMMODITY_ASSETS = {
    "XAUUSD", "XAUUSD_M15", "XAUUSD_FIB", "XAUUSD_ORB",
    "USOIL", "USOIL_FIB",
    "GBPUSD", "GBPUSD_FIB",
    "EURUSD", "EURUSD_FIB",
    "XAGUSD", "XAGUSD_RSI2",
}

# TradingAgents' 5-tier rating scale (tradingagents.agents.rating) collapsed
# to Ra's 3-way BUY/SELL/HOLD signal contract.
RATING_TO_DIRECTION = {
    "buy": "BUY",
    "overweight": "BUY",
    "hold": "HOLD",
    "underweight": "SELL",
    "sell": "SELL",
}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    # Write-then-rename so a crash mid-write can't leave a half-written,
    # unparseable cache file behind for the next poll cycle to trip over.
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, CACHE_FILE)


def get_llm_signal(asset: str, logger=None) -> str | None:
    """Return "BUY"/"SELL"/"HOLD" for `asset`, or None if no decision is
    available (missing/invalid API key, the call failed, or the model's
    output had no parseable rating). None is NOT the same as "HOLD" -
    callers should treat it as "skip this cycle, try again later", not as
    a bearish/neutral call that was actually made.

    Calls the real TradingAgents multi-agent pipeline at most once per UTC
    calendar day per `asset`; every other call that day reuses the cached
    decision from CACHE_FILE instead of re-spending LLM tokens on an
    unchanged answer.
    """
    ticker = ASSET_TO_TRADINGAGENTS_TICKER.get(asset)
    if ticker is None:
        if logger:
            logger.error(f"[LLM_STRATEGY] no TradingAgents ticker mapping for asset {asset!r}")
        return None

    today = _today()
    cache = _load_cache()
    cached = cache.get(asset)
    if cached and cached.get("date") == today:
        return cached.get("direction")

    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except ImportError:
        if logger:
            logger.error("[LLM_STRATEGY] tradingagents package not installed - skipping")
        return None

    config = DEFAULT_CONFIG.copy()
    analysts = ["market", "social", "news"]
    if asset not in FOREX_COMMODITY_ASSETS:
        analysts.append("fundamentals")  # only meaningful for a real equity ticker

    try:
        graph = TradingAgentsGraph(selected_analysts=analysts, debug=False, config=config)
        _, rating = graph.propagate(ticker, today)
    except Exception as e:
        if logger:
            logger.error(f"[LLM_STRATEGY] TradingAgents call failed for {asset} ({ticker}): {e}")
        return None

    direction = RATING_TO_DIRECTION.get(str(rating).strip().lower())
    if direction is None:
        # Unparseable / "REVIEW" rating (see tradingagents.agents.rating's
        # module docstring) - not cached, so the next poll retries instead
        # of getting stuck on a bad rating for the rest of the day.
        if logger:
            logger.error(f"[LLM_STRATEGY] {asset}: unparseable rating {rating!r} - no signal, not cached")
        return None

    cache[asset] = {"date": today, "direction": direction, "raw_rating": str(rating)}
    _save_cache(cache)
    return direction
