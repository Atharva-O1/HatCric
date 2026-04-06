"""
HatCric Backend Proxy
=====================
A lightweight Flask proxy server that:
  - Hides your Live Score API key from client-side code
  - Caches responses for 15 seconds to protect your API quota
  - Sanitizes the external API response to only return fields HatCric needs
  - Allows CORS from your local frontend dev servers
"""

import time
import logging
from flask import Flask, jsonify, abort
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import os

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()   # Reads LIVE_SCORE_API_KEY from .env

app = Flask(__name__)

# Configure Python's built-in logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── CORS ───────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# ── Environment / Config ───────────────────────────────────────────────────────
API_KEY = os.getenv("LIVE_SCORE_API_KEY")
if not API_KEY:
    raise EnvironmentError("LIVE_SCORE_API_KEY is not set in your .env file.")

CACHE_TTL_SECONDS = 15
REQUEST_TIMEOUT = 8

# ── In-Memory Cache ────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}

def get_cached(match_id: str):
    entry = _cache.get(match_id)
    if entry is None:
        return None
    age = time.time() - entry["cached_at"]
    if age < CACHE_TTL_SECONDS:
        logger.info("Cache HIT  for match_id=%s (age=%.1fs)", match_id, age)
        return entry["data"]
    return None

def set_cached(match_id: str, data: dict):
    _cache[match_id] = {"data": data, "cached_at": time.time()}

# ── Score Parsing Helper ───────────────────────────────────────────────────────
def parse_score_response(raw: dict) -> dict:
    """
    Extracts runs, wickets, and overs specifically from the Cricbuzz RapidAPI 'hscard' endpoint.
    """
    try:
        miniscore = raw.get("miniscore")
        
        if not miniscore:
            return {"current_score": 0, "wickets": 0, "overs_bowled": 0.0}

        bat_team = miniscore.get("batTeam", {})
        
        current_score = int(bat_team.get("teamScore", 0))
        wickets = int(bat_team.get("teamWkts", 0))
        overs_bowled = float(miniscore.get("overs", 0.0))

        return {
            "current_score": current_score,
            "wickets": wickets,
            "overs_bowled": overs_bowled,
        }

    except Exception as exc:
        logger.warning("Could not parse Cricbuzz score payload — %s", exc)
        return {"current_score": 0, "wickets": 0, "overs_bowled": 0.0}

# ── Proxy Endpoint ─────────────────────────────────────────────────────────────
@app.route("/api/match/<string:match_id>", methods=["GET"])
def get_match_data(match_id: str):
    
    # ── 0. MOCK BYPASS (For testing when no games are live) ──
    if match_id == "mock_test":
        logger.info("Serving Mock Data for UI testing!")
        return jsonify({
            "match_id": "mock_test",
            "current_score": 156,
            "wickets": 4,
            "overs_bowled": 16.2,
            "last_5_overs": ["10", "W", "4", "12", "8"], # <-- ADD THIS LINE
            "from_cache": False
        })

    # Basic input validation
    if not match_id.replace("-", "").replace("_", "").isalnum():
        abort(400, description="Invalid match_id format.")

    # 1. Cache check
    cached_data = get_cached(match_id)
    if cached_data is not None:
        return jsonify({**cached_data, "match_id": match_id, "from_cache": True})

    # 2. External API request (Cricbuzz RapidAPI Format)
    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{match_id}/hscard"
    headers = {
        "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com",
        "x-rapidapi-key": API_KEY
    }

    logger.info("Cache MISS — calling external API for match_id=%s", match_id)

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

    except requests.exceptions.Timeout:
        return jsonify({"error": "External API timed out. Please retry."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not reach the live score service."}), 502
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        if status == 404:
            abort(404, description=f"Match '{match_id}' not found.")
        return jsonify({"error": f"External API error ({status})."}), 502

    # 3. Parse & sanitise
    raw_json = response.json()
    
    logger.info("=== RAW RAPIDAPI RESPONSE ===")
    logger.info(raw_json)
    logger.info("=============================")
    
    sanitised = parse_score_response(raw_json)

    # 4. Cache & respond
    set_cached(match_id, sanitised)
    logger.info(
        "Served fresh data for match_id=%s  score=%s/%s  overs=%s",
        match_id, sanitised["current_score"], sanitised["wickets"], sanitised["overs_bowled"]
    )

    return jsonify({**sanitised, "match_id": match_id, "from_cache": False})

# ── Health-check Endpoint ──────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "api_key_set": bool(API_KEY)})

if __name__ == "__main__":
    logger.info("Starting HatCric proxy on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)