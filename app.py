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
LIVE_MATCHES_CACHE_KEY = "__live_matches__"


def overs_to_balls(overs) -> int:
    try:
        overs_float = float(overs or 0)
    except (TypeError, ValueError):
        return 0
    whole_overs = int(overs_float)
    balls_part = int(round((overs_float - whole_overs) * 10))
    return max(0, whole_overs * 6 + min(max(balls_part, 0), 5))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def detect_winner(status: str, team_a: str, team_b: str) -> str | None:
    status_l = (status or "").lower()
    if "won" not in status_l:
        return None

    team_a_l = (team_a or "").lower()
    team_b_l = (team_b or "").lower()
    if team_a_l and team_a_l in status_l:
        return "team_a"
    if team_b_l and team_b_l in status_l:
        return "team_b"
    return None


def project_first_innings_score(runs: int, wickets: int, overs_bowled: float) -> int:
    balls_bowled = overs_to_balls(overs_bowled)
    balls_remaining = max(0, 120 - balls_bowled)
    if balls_bowled <= 0:
        return 0

    wickets_lost = max(0, wickets)
    wickets_left = max(0, 10 - wickets_lost)
    current_rr = runs / (balls_bowled / 6)

    if balls_bowled <= 36:
        expected_rr = (current_rr * 0.6) + (8.8 * 0.4)
    elif balls_bowled <= 90:
        expected_rr = (current_rr * 0.75) + ((8.2 + wickets_left * 0.15) * 0.25)
    else:
        expected_rr = (current_rr * 0.7) + ((9.5 + wickets_left * 0.3) * 0.3)

    expected_rr -= max(0, wickets_lost - 2) * 0.45
    expected_rr = clamp(expected_rr, 5.5, 13.5)

    projected_total = runs + (balls_remaining / 6) * expected_rr
    return int(round(projected_total))


def chase_win_probability(runs: int, wickets: int, overs_bowled: float, target: int) -> float:
    balls_bowled = overs_to_balls(overs_bowled)
    balls_remaining = max(0, 120 - balls_bowled)
    runs_needed = max(0, target - runs)

    if runs_needed <= 0:
        return 99
    if balls_remaining <= 0:
        return 1

    wickets_left = max(0, 10 - max(0, wickets))
    current_rr = (runs / (balls_bowled / 6)) if balls_bowled > 0 else 0
    required_rr = runs_needed / (balls_remaining / 6)

    rate_edge = (current_rr - required_rr) * 7.5
    wicket_edge = (wickets_left - 5) * 4.5
    pressure_edge = (balls_remaining / 6 - runs_needed / 6) * 1.2
    chase_score = 50 + rate_edge + wicket_edge + pressure_edge

    return round(clamp(chase_score, 1, 99))


def first_innings_win_probability(projected_total: int, overs_bowled: float, wickets: int) -> float:
    balls_bowled = overs_to_balls(overs_bowled)
    par_score = 168
    if balls_bowled <= 36:
        par_score = 172
    elif balls_bowled >= 96:
        par_score = 165

    wickets_penalty = max(0, wickets - 4) * 3
    advantage = projected_total - (par_score - wickets_penalty)
    return round(clamp(50 + advantage * 0.6, 10, 90))

def get_cached(match_id: str):
    entry = _cache.get(match_id)
    if entry is None: return None
    age = time.time() - entry["cached_at"]
    if age < CACHE_TTL_SECONDS:
        return entry["data"]
    return None

def set_cached(match_id: str, data: dict):
    _cache[match_id] = {"data": data, "cached_at": time.time()}


def get_api_headers() -> dict:
    return {
        "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com",
        "x-rapidapi-key": API_KEY,
    }


def fetch_live_matches() -> list[dict]:
    cached = get_cached(LIVE_MATCHES_CACHE_KEY)
    if cached is not None:
        return cached

    url = "https://cricbuzz-cricket.p.rapidapi.com/matches/v1/live"
    response = requests.get(url, headers=get_api_headers(), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    raw = response.json()

    matches = []
    for wrapper in raw.get("seriesMatches", []):
        adapter = wrapper.get("seriesAdWrapper", {})
        for match in adapter.get("matches", []):
            info = match.get("matchInfo", {}) or {}
            if info:
                matches.append(info)

    set_cached(LIVE_MATCHES_CACHE_KEY, matches)
    return matches


def get_match_info_from_live(match_id: str) -> dict:
    try:
        for info in fetch_live_matches():
            if str(info.get("matchId")) == str(match_id):
                return info
    except Exception as exc:
        logger.warning("Live matches enrichment failed for %s: %s", match_id, exc)
    return {}

def parse_score_response(data, match_id: str | None = None):
    try:
        is_complete = data.get("ismatchcomplete", False)
        status = data.get("status", "Match Live")
        live_info = get_match_info_from_live(match_id) if match_id else {}
        match_info = data.get("matchInfo") or data.get("matchHeader") or live_info or {}
        state = (
            match_info.get("state")
            or data.get("state")
            or ("Complete" if is_complete else "Preview")
        )

        team1_info = match_info.get("team1", {})
        team2_info = match_info.get("team2", {})
        team1 = team1_info.get("shortName") or team1_info.get("teamSName") or "TEAM A"
        team2 = team2_info.get("shortName") or team2_info.get("teamSName") or "TEAM B"

        res = {
            "innings": 1, "target": 0, "current_score": 0, "wickets": 0, "overs_bowled": 0.0,
            "team_a": team1, "team_b": team2, "status": status,
            "is_complete": is_complete, "state": state,
            "batting_team": None,
            "predicted_score": 0,
            "prediction_note": "Awaiting live data",
            "win_probability": {"team_a": 50, "team_b": 50}
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
                "batting_team": bat.get("shortName") or bat.get("teamSName"),
            })
             
        # 2. LIVE/COMPLETED DATA FALLBACK (Scorecard block)
        # Cricbuzz continually updates the scorecard. If miniscore is missing, pull from here.
        if "scorecard" in data and len(data["scorecard"]) > 0:
            sc = data["scorecard"]
            first_innings = sc[0]
            last_innings = sc[-1]

            batting_short = (
                last_innings.get("batteamsname")
                or last_innings.get("batTeamSName")
                or last_innings.get("batteamname")
            )
            batting_full = last_innings.get("batteamname")

            if team1 == "TEAM A" and batting_short:
                team1 = batting_short
                res["team_a"] = team1
            if team2 == "TEAM B":
                first_batting_short = (
                    first_innings.get("batteamsname")
                    or first_innings.get("batTeamSName")
                    or first_innings.get("batteamname")
                )
                if first_batting_short and first_batting_short != team1:
                    team2 = first_batting_short
                    res["team_b"] = team2
             
            # If current_score is 0, miniscore missed it. Grab it from scorecard.
            if res["current_score"] == 0:
                batting_team_id = (
                    last_innings.get("batTeamDetails", {}).get("batTeamId")
                    or last_innings.get("teamId")
                )
                batting_team = (
                    team1 if batting_team_id == team1_info.get("teamId")
                    else team2 if batting_team_id == team2_info.get("teamId")
                    else batting_short
                    or batting_full
                )
                res.update({
                    "state": "Live" if not is_complete else "Complete",
                    "innings": last_innings.get("inningsid", 1),
                    "current_score": last_innings.get("score", 0),
                    "wickets": last_innings.get("wickets", 0),
                    "overs_bowled": last_innings.get("overs", 0.0),
                    "batting_team": batting_team
                })
             
            # Always safely calculate the target from the 1st innings
            if len(sc) >= 2:
                res["target"] = sc[0].get("score", 0) + 1
                if team1 == "TEAM A":
                    first_team = (
                        first_innings.get("batteamsname")
                        or first_innings.get("batTeamSName")
                        or first_innings.get("batteamname")
                    )
                    if first_team:
                        team1 = first_team
                        res["team_a"] = team1
                if team2 == "TEAM B" and res["batting_team"]:
                    chasing_team = res["batting_team"]
                    if chasing_team != team1:
                        team2 = chasing_team
                        res["team_b"] = team2

        # 3. OVERRIDE 'PREVIEW' IF TOSS HAPPENED OR MATCH IS LIVE
        if "Preview" in res["state"] and ("toss" in status.lower() or "Match Live" in status):
            res["state"] = "Live"

        runs = int(res["current_score"] or 0)
        wickets = int(res["wickets"] or 0)
        overs_bowled = float(res["overs_bowled"] or 0)

        if not res["is_complete"] and res["state"] != "Preview":
            if res["innings"] == 2 and res["target"] > 0:
                chasing_win = chase_win_probability(runs, wickets, overs_bowled, int(res["target"]))
                if res["batting_team"] == res["team_a"]:
                    res["win_probability"] = {"team_a": chasing_win, "team_b": 100 - chasing_win}
                else:
                    res["win_probability"] = {"team_a": 100 - chasing_win, "team_b": chasing_win}
                res["predicted_score"] = int(res["target"])
                balls_remaining = max(0, 120 - overs_to_balls(overs_bowled))
                runs_needed = max(0, int(res["target"]) - runs)
                req_rr = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0
                res["prediction_note"] = f"{runs_needed} needed from {balls_remaining} balls at {req_rr:.2f} RPO"
            else:
                res["predicted_score"] = project_first_innings_score(runs, wickets, overs_bowled)
                batting_win = first_innings_win_probability(res["predicted_score"], overs_bowled, wickets)
                if res["batting_team"] == res["team_b"]:
                    res["win_probability"] = {"team_a": 100 - batting_win, "team_b": batting_win}
                else:
                    res["win_probability"] = {"team_a": batting_win, "team_b": 100 - batting_win}
                balls_remaining = max(0, 120 - overs_to_balls(overs_bowled))
                res["prediction_note"] = f"Projected total based on {balls_remaining} balls remaining"

        # 4. WIN PROBABILITY MATH FOR COMPLETED MATCHES
        if is_complete:
            winner = detect_winner(status, res["team_a"], res["team_b"])
            if winner == "team_a":
                res["win_probability"] = {"team_a": 100, "team_b": 0}
            elif winner == "team_b":
                res["win_probability"] = {"team_a": 0, "team_b": 100}
            else:
                res["win_probability"] = {"team_a": 50, "team_b": 50}
            res["predicted_score"] = runs
            res["prediction_note"] = status

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
            "batting_team": "MI",
            "predicted_score": 185,
            "prediction_note": "65 needed from 34 balls at 11.47 RPO",
            "win_probability": {"team_a": 42, "team_b": 58}
        })

    cached_data = get_cached(match_id)
    if cached_data is not None:
        return jsonify({**cached_data, "match_id": match_id, "from_cache": True})

    url = f"https://cricbuzz-cricket.p.rapidapi.com/mcenter/v1/{match_id}/hscard"
    headers = get_api_headers()

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200: return jsonify({"error": "API Error"}), response.status_code
        
        raw_json = response.json()
        sanitised = parse_score_response(raw_json, match_id=match_id)
        set_cached(match_id, sanitised)
        return jsonify({**sanitised, "match_id": match_id, "from_cache": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
