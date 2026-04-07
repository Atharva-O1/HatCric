"""
HatCric Backend Proxy
=====================
"""

import time
import logging
from flask import Flask, jsonify, abort
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:5500", "http://127.0.0.1:5500",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8080", "http://127.0.0.1:8080",
]

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

API_KEY = os.getenv("LIVE_SCORE_API_KEY")
if not API_KEY:
    raise EnvironmentError("LIVE_SCORE_API_KEY is not set in your .env file.")

CACHE_TTL_SECONDS = 15
REQUEST_TIMEOUT = 8
_cache: dict[str, dict] = {}

def get_cached(match_id: str):
    entry = _cache.get(match_id)
    if entry is None: return None
    age = time.time() - entry["cached_at"]
    if age < CACHE_TTL_SECONDS:
        return entry["data"]
    return None

def set_cached(match_id: str, data: dict):
    _cache[match_id] = {"data": data, "cached_at": time.time()}

def parse_score_response(data):
    try:
        is_complete = data.get("ismatchcomplete", False)
        status = data.get("status", "Match Live")
        
        # CRITICAL FIX: Cricbuzz uses 'matchInfo' for live matches, not 'matchHeader'
        match_info = data.get("matchInfo", data.get("matchHeader", {}))
        state = match_info.get("state", data.get("state", "Preview"))
        
        team1 = match_info.get("team1", {}).get("shortName", "RR")
        team2 = match_info.get("team2", {}).get("shortName", "MI")

        res = {
            "innings": 1, "target": 0, "current_score": 0, "wickets": 0, "overs_bowled": 0.0,
            "team_a": team1, "team_b": team2, "status": status,
            "is_complete": is_complete, "state": state,
            "win_probability": {"team_a": 43, "team_b": 57}
        }

        # 1. LIVE DATA (Miniscore block)
        if "miniscore" in data:
            mini = data["miniscore"]
            bat = mini.get("batTeam", {})
            res.update({
                "state": "Live", # Force state out of preview
                "innings": mini.get("inningsId", 1),
                "target": mini.get("target", 0),
                "current_score": bat.get("teamScore", 0),
                "wickets": bat.get("teamWkts", 0),
                "overs_bowled": bat.get("overs", 0.0),
            })
            
        # 2. LIVE/COMPLETED DATA FALLBACK (Scorecard block)
        # Cricbuzz continually updates the scorecard. If miniscore is missing, pull from here.
        if "scorecard" in data and len(data["scorecard"]) > 0:
            sc = data["scorecard"]
            last_innings = sc[-1]
            
            # If current_score is 0, miniscore missed it. Grab it from scorecard.
            if res["current_score"] == 0:
                res.update({
                    "state": "Live" if not is_complete else "Complete",
                    "innings": last_innings.get("inningsid", 1),
                    "current_score": last_innings.get("score", 0),
                    "wickets": last_innings.get("wickets", 0),
                    "overs_bowled": last_innings.get("overs", 0.0)
                })
            
            # Always safely calculate the target from the 1st innings
            if len(sc) >= 2:
                res["target"] = sc[0].get("score", 0) + 1

        # 3. OVERRIDE 'PREVIEW' IF TOSS HAPPENED OR MATCH IS LIVE
        if "Preview" in res["state"] and ("toss" in status.lower() or "Match Live" in status):
            res["state"] = "Live"

        # 4. WIN PROBABILITY MATH
        if is_complete:
            if res["team_a"].lower() in status.lower() or "won" in status.lower():
                if any(word in status.lower() for word in res["team_a"].lower().split()):
                    res["win_probability"] = {"team_a": 100, "team_b": 0}
                else:
                    res["win_probability"] = {"team_a": 0, "team_b": 100}
        elif res["innings"] == 2 and res["target"] > 0:
            runs_needed = res["target"] - res["current_score"]
            prob_b = max(5, min(95, 100 - (runs_needed / 2)))
            res["win_probability"] = {"team_a": 100 - prob_b, "team_b": prob_b}

        return res

    except Exception as e:
        logger.error(f"Parser Error: {e}")
        return {"innings": 1, "target": 0, "current_score": 0, "wickets": 0, "overs_bowled": 0, "is_complete": False, "state": "Error"}

@app.route("/api/match/<string:match_id>", methods=["GET"])
def get_match_data(match_id: str):
    if match_id == "mock_test":
        return jsonify({
            "match_id": "mock_test", "innings": 2, "target": 185, "current_score": 120,
            "wickets": 3, "overs_bowled": 14.2, "team_a": "RR", "team_b": "MI",
            "is_complete": False, "state": "Live", "status": "Live Match",
            "win_probability": {"team_a": 58, "team_b": 42}
        })

    cached_data = get_cached(match_id)
    if cached_data is not None:
        return jsonify({**cached_data, "match_id": match_id, "from_cache": True})

    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{match_id}/hscard"
    headers = {"x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com", "x-rapidapi-key": API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200: return jsonify({"error": "API Error"}), response.status_code
        
        raw_json = response.json()
        sanitised = parse_score_response(raw_json)
        set_cached(match_id, sanitised)
        return jsonify({**sanitised, "match_id": match_id, "from_cache": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)