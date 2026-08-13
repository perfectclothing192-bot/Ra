"""
Minimal HTTP status server exposing an agent's current state as JSON, so an
external dashboard can poll it. Runs in a background thread alongside a
run_loop.py / run_scalp_loop.py polling loop; doesn't interfere with it.
"""

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_state_lock = threading.Lock()
_state = {}


def serialize_positions(positions: dict, current_prices: dict = None) -> dict:
    current_prices = current_prices or {}
    result = {}
    for asset, pos in positions.items():
        entry = {**asdict(pos), "entry_time": pos.entry_time.isoformat()}
        price = current_prices.get(asset)
        if price:
            direction_multiplier = 1 if pos.direction == "LONG" else -1
            entry["current_price"] = price
            entry["unrealized_pnl"] = (price - pos.entry_price) * pos.quantity * direction_multiplier
            entry["unrealized_pnl_percent"] = (
                (price - pos.entry_price) / pos.entry_price * 100 * direction_multiplier
            )
        result[asset] = entry
    return result


def serialize_trades(trades: list, limit: int = 20) -> list:
    return [
        {
            **asdict(t),
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
        }
        for t in trades[-limit:]
    ]


def update_status(**kwargs):
    with _state_lock:
        _state.update(kwargs)
        _state["updated_at"] = datetime.now(timezone.utc).isoformat()


def _snapshot() -> dict:
    with _state_lock:
        return dict(_state)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/status"):
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(_snapshot(), default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # the agent's own logger already covers activity logging


def start_status_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
