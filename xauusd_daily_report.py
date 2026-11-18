"""
Daily XAUUSD market-read report: price, structure/trend, order-block
zones, and an order-flow (CVD) proxy. For the user's own informational
daily briefing - NOT a trading strategy, does not place orders, does not
touch trading_state.json. Reuses the same detection logic already live
in trading_agent.py (smc_signal's fractal swing structure) and already
backtested there (order_block_signal's zone definition), and the same
honest order-flow proxy from the order-flow backtest research
(buy/sell volume split from OHLCV, since OANDA's REST API has no real
bid/ask depth - this is a proxy, not genuine tape data).

Run standalone: OANDA_API_KEY=... OANDA_ACCOUNT_ID=... python3 xauusd_daily_report.py
Writes a JSON data file (default: xauusd_report_data.json) for the
dashboard artifact to render.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from oanda_client import fetch_candles, fetch_current_price

OUTPUT_FILE = os.environ.get("XAUUSD_REPORT_FILE", "xauusd_report_data.json")
GRANULARITY = os.environ.get("XAUUSD_REPORT_GRANULARITY", "M15")
LOOKBACK_BARS = int(os.environ.get("XAUUSD_REPORT_LOOKBACK", "1000"))  # ~10 days on M15

FRACTAL = 3                 # same as smc_signal's swing-point fractal width
OB_LOOKBACK = 50            # order-block scan window, same as order_block_signal
OB_DISPLACEMENT_BARS = 3
OB_DISPLACEMENT_MULT = 2.0
CVD_DIVERGENCE_WINDOW = 50  # ~12.5h on M15, same order of magnitude as order_block_signal's lookback


def _atr(highs, lows, closes, period=14):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(
        np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])
    ))
    atr = np.zeros(len(closes))
    atr[period] = tr[:period].mean()
    for i in range(period + 1, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    atr[:period] = atr[period]
    return atr


def find_swing_points(highs, lows, fractal=FRACTAL):
    n = len(highs)
    swing_low_idxs = [i for i in range(fractal, n - fractal)
                       if lows[i] == min(lows[i - fractal:i + fractal + 1])]
    swing_high_idxs = [i for i in range(fractal, n - fractal)
                        if highs[i] == max(highs[i - fractal:i + fractal + 1])]
    return swing_high_idxs, swing_low_idxs


def structure_bias(highs, lows, swing_high_idxs, swing_low_idxs):
    """Bullish if the last two swing highs AND the last two swing lows are
    both rising (higher highs, higher lows); bearish if both falling;
    mixed/ranging otherwise."""
    if len(swing_high_idxs) < 2 or len(swing_low_idxs) < 2:
        return "insufficient_data"
    hh = highs[swing_high_idxs[-1]] > highs[swing_high_idxs[-2]]
    hl = lows[swing_low_idxs[-1]] > lows[swing_low_idxs[-2]]
    lh = highs[swing_high_idxs[-1]] < highs[swing_high_idxs[-2]]
    ll = lows[swing_low_idxs[-1]] < lows[swing_low_idxs[-2]]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"


def find_order_blocks(opens, highs, lows, closes, atr, timestamps, lookback=OB_LOOKBACK):
    """Same zone definition as trading_agent.py's order_block_signal:
    a base candle immediately followed by a strong impulsive displacement
    that breaks its structure. Unlike the live strategy (which only tracks
    the single most recent still-fresh block), this collects every
    qualifying block in the lookback window and marks each one fresh
    (price hasn't closed back through it yet) or tested (it has), since a
    report is more useful showing several real zones than just one.
    """
    n = len(closes)
    earliest_k = max(1, n - lookback)
    zones = []

    for k in range(n - OB_DISPLACEMENT_BARS - 2, earliest_k - 1, -1):
        window_end = k + OB_DISPLACEMENT_BARS
        if window_end >= n - 1:
            continue
        a = atr[k] if atr[k] > 0 else atr[atr > 0].mean()

        # Bullish order block: bearish base candle, then a strong up-displacement.
        if closes[k] < opens[k]:
            displaced = (highs[k + 1:window_end + 1].max() - lows[k]) >= OB_DISPLACEMENT_MULT * a
            broke_structure = closes[window_end] > highs[k]
            if displaced and broke_structure:
                zone_low, zone_high = lows[k], highs[k]
                tested = any(lows[j] <= zone_high for j in range(window_end + 1, n))
                zones.append({
                    "direction": "bullish", "low": float(zone_low), "high": float(zone_high),
                    "formed_at": timestamps[k], "tested": tested,
                })

        # Bearish order block: bullish base candle, then a strong down-displacement.
        if closes[k] > opens[k]:
            displaced = (highs[k] - lows[k + 1:window_end + 1].min()) >= OB_DISPLACEMENT_MULT * a
            broke_structure = closes[window_end] < lows[k]
            if displaced and broke_structure:
                zone_low, zone_high = lows[k], highs[k]
                tested = any(highs[j] >= zone_low for j in range(window_end + 1, n))
                zones.append({
                    "direction": "bearish", "low": float(zone_low), "high": float(zone_high),
                    "formed_at": timestamps[k], "tested": tested,
                })

    # Most recent first; cap to the 6 most recent per direction so the
    # chart doesn't drown in stale zones.
    zones.sort(key=lambda z: z["formed_at"], reverse=True)
    bullish = [z for z in zones if z["direction"] == "bullish"][:6]
    bearish = [z for z in zones if z["direction"] == "bearish"][:6]
    return bullish + bearish


def compute_cvd(volumes, highs, lows, closes):
    """Order-flow proxy from OHLCV (OANDA's REST API has no real bid/ask
    depth - see module docstring). delta = estimated buy volume - sell
    volume for the bar; CVD = its cumulative sum."""
    rng = highs - lows
    buy_vol = np.where(rng > 0, volumes * (closes - lows) / np.where(rng > 0, rng, 1), volumes * 0.5)
    sell_vol = np.where(rng > 0, volumes * (highs - closes) / np.where(rng > 0, rng, 1), volumes * 0.5)
    delta = buy_vol - sell_vol
    cvd = np.cumsum(delta)
    return delta, cvd


def cvd_divergence_read(closes, cvd, window=CVD_DIVERGENCE_WINDOW):
    if len(closes) < window + 1:
        return "insufficient_data"
    price_change = closes[-1] - closes[-window]
    cvd_change = cvd[-1] - cvd[-window]
    if price_change < 0 and cvd_change > 0:
        return "bullish_divergence"   # price down, order flow accumulating
    if price_change > 0 and cvd_change < 0:
        return "bearish_divergence"   # price up, order flow distributing
    if price_change >= 0 and cvd_change >= 0:
        return "confirming_up"
    return "confirming_down"


def main():
    bars = fetch_candles("XAUUSD", granularity=GRANULARITY, count=LOOKBACK_BARS)
    if len(bars) < 100:
        print(f"Not enough bars ({len(bars)}) to build a report", file=sys.stderr)
        sys.exit(1)

    opens = np.array([b.open for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    volumes = np.array([float(b.volume) for b in bars])
    timestamps = [b.timestamp.isoformat() for b in bars]

    atr = _atr(highs, lows, closes)
    swing_high_idxs, swing_low_idxs = find_swing_points(highs, lows)
    bias = structure_bias(highs, lows, swing_high_idxs, swing_low_idxs)
    zones = find_order_blocks(opens, highs, lows, closes, atr, timestamps)
    delta, cvd = compute_cvd(volumes, highs, lows, closes)
    divergence = cvd_divergence_read(closes, cvd)

    current_price = fetch_current_price("XAUUSD")
    day_ago_idx = max(0, len(closes) - int(24 * 60 / 15))  # ~24h back on M15
    day_change = current_price - closes[day_ago_idx]
    day_change_pct = (day_change / closes[day_ago_idx]) * 100 if closes[day_ago_idx] else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instrument": "XAUUSD",
        "granularity": GRANULARITY,
        "current_price": current_price,
        "day_change": day_change,
        "day_change_pct": day_change_pct,
        "structure_bias": bias,
        "cvd_divergence": divergence,
        "zones": zones,
        "candles": [
            {"t": timestamps[i], "o": float(opens[i]), "h": float(highs[i]),
             "l": float(lows[i]), "c": float(closes[i])}
            for i in range(len(bars))
        ],
        "cvd_series": [{"t": timestamps[i], "cvd": float(cvd[i])} for i in range(len(bars))],
        "swing_highs": [{"t": timestamps[i], "price": float(highs[i])} for i in swing_high_idxs[-10:]],
        "swing_lows": [{"t": timestamps[i], "price": float(lows[i])} for i in swing_low_idxs[-10:]],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        "current_price": current_price, "day_change_pct": round(day_change_pct, 3),
        "structure_bias": bias, "cvd_divergence": divergence,
        "zone_count": len(zones), "output_file": OUTPUT_FILE,
    }))


if __name__ == "__main__":
    main()
