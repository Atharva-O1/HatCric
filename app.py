"""
HatCric Backend Proxy
=====================
"""

import time
import logging
import re
from flask import Flask, jsonify, abort
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import os
from prematch_model import predict_match_probability
from live_model import predict_live_win_probability

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
MATCH_LIST_CACHE_KEY = "__match_lists__"


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


def _alias_in_status(alias: str, status_l: str) -> bool:
    alias = (alias or "").strip().lower()
    if not alias:
        return False
    if len(alias) <= 4 and alias.replace(" ", "").isalpha():
        return re.search(rf"\b{re.escape(alias)}\b", status_l) is not None
    return alias in status_l


def detect_winner(
    status: str,
    team_a: str,
    team_b: str,
    team_a_full: str = "",
    team_b_full: str = "",
) -> str | None:
    status_l = (status or "").lower()
    if "won" not in status_l and "beat" not in status_l:
        return None

    if _alias_in_status(team_a, status_l) or _alias_in_status(team_a_full, status_l):
        return "team_a"
    if _alias_in_status(team_b, status_l) or _alias_in_status(team_b_full, status_l):
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


def build_prediction_explanation(
    *,
    state: str,
    is_complete: bool,
    team_a: str,
    team_b: str,
    team_a_pct: int,
    team_b_pct: int,
    status: str,
    batting_team: str | None = None,
    innings: int = 1,
    target: int = 0,
    runs: int = 0,
    wickets: int = 0,
    overs_bowled: float = 0.0,
    predicted_score: int = 0,
    used_lineups: bool = False,
) -> str:
    favored_team = team_a if team_a_pct >= team_b_pct else team_b

    if is_complete:
        return status or f"{favored_team} finished on top."

    if state == "Preview":
        if used_lineups:
            return f"HatCric favors {favored_team} from the announced XI balance, venue context, and recent team strength."
        return f"HatCric favors {favored_team} from recent form, venue fit, and historical team strength."

    balls_remaining = max(0, 120 - overs_to_balls(overs_bowled))
    if innings == 2 and target > 0:
        runs_needed = max(0, target - runs)
        req_rr = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0.0
        return f"HatCric leans {favored_team} based on the chase pressure, wickets in hand, and required rate at {req_rr:.2f}."

    current_rr = (runs / (overs_to_balls(overs_bowled) / 6)) if overs_to_balls(overs_bowled) > 0 else 0.0
    batting_side = batting_team or favored_team
    return f"HatCric leans {favored_team} because {batting_side} is scoring at {current_rr:.2f}, projects to {predicted_score}, and has {max(0, 10 - wickets)} wickets left."


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def extract_batters_at_crease(innings_data: dict | None) -> list[dict]:
    batsmen = (innings_data or {}).get("batsman") or []
    active = [
        batter for batter in batsmen
        if str(batter.get("outdec", "")).strip().lower() == "batting"
    ]

    if len(active) < 2:
        not_out = [
            batter for batter in batsmen
            if not str(batter.get("outdec", "")).strip()
        ]
        for batter in not_out:
            if batter not in active:
                active.append(batter)
            if len(active) >= 2:
                break

    return [
        {
            "name": batter.get("nickname") or batter.get("name") or "Batter",
            "runs": _safe_int(batter.get("runs")),
            "balls": _safe_int(batter.get("balls")),
            "fours": _safe_int(batter.get("fours")),
            "sixes": _safe_int(batter.get("sixes")),
            "strike_rate": str(batter.get("strkrate", "0")),
            "is_batting": str(batter.get("outdec", "")).strip().lower() == "batting",
        }
        for batter in active[:2]
    ]

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
    cached = get_cached(MATCH_LIST_CACHE_KEY)
    if cached is not None:
        return cached

    final_matches = _fetch_match_infos(("live", "recent", "upcoming"))
    set_cached(MATCH_LIST_CACHE_KEY, final_matches)
    set_cached(LIVE_MATCHES_CACHE_KEY, final_matches)
    return final_matches


def _fetch_match_infos(paths: tuple[str, ...]) -> list[dict]:
    matches = []
    for path in paths:
        url = f"https://cricbuzz-cricket.p.rapidapi.com/matches/v1/{path}"
        response = requests.get(url, headers=get_api_headers(), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        raw = response.json()

        for type_match in raw.get("typeMatches", []):
            for wrapper in type_match.get("seriesMatches", []):
                adapter = wrapper.get("seriesAdWrapper", {})
                for match in adapter.get("matches", []):
                    info = match.get("matchInfo", {}) or {}
                    if info:
                        matches.append(info)

    deduped = {}
    for info in matches:
        if info.get("matchId") is not None:
            deduped[str(info["matchId"])] = info

    return list(deduped.values())


def fetch_selectable_matches() -> list[dict]:
    live_and_upcoming = _fetch_match_infos(("live", "upcoming"))
    recent_matches = _fetch_match_infos(("recent",))

    latest_complete = None
    for match_info in recent_matches:
        if str(match_info.get("state") or "").lower() == "complete":
            latest_complete = match_info
            break

    if latest_complete is None:
        return live_and_upcoming

    merged = {}
    for match_info in live_and_upcoming + [latest_complete]:
        match_id = match_info.get("matchId")
        if match_id is not None:
            merged[str(match_id)] = match_info
    return list(merged.values())


def get_match_info_from_live(match_id: str) -> dict:
    try:
        for info in fetch_live_matches():
            if str(info.get("matchId")) == str(match_id):
                return info
    except Exception as exc:
        logger.warning("Live matches enrichment failed for %s: %s", match_id, exc)
    return {}


def serialize_match_info(match_info: dict) -> dict | None:
    match_id = match_info.get("matchId")
    if match_id is None:
        return None

    team1_info = match_info.get("team1", {}) or {}
    team2_info = match_info.get("team2", {}) or {}
    team_a = team1_info.get("shortName") or team1_info.get("teamSName") or team1_info.get("teamName")
    team_b = team2_info.get("shortName") or team2_info.get("teamSName") or team2_info.get("teamName")
    if not team_a or not team_b:
        return None

    state = match_info.get("state") or "Unknown"
    series_name = match_info.get("seriesName") or ""
    status = match_info.get("status") or ""
    label = f"{team_a} vs {team_b}"
    badge = state

    match_desc = match_info.get("matchDesc") or match_info.get("matchDescription") or ""
    if match_desc:
        badge = match_desc
    elif series_name:
        badge = series_name

    return {
        "match_id": str(match_id),
        "team_a": team_a,
        "team_b": team_b,
        "label": label,
        "badge": badge,
        "state": state,
        "status": status,
        "series_name": series_name,
        "venue": {
            "ground": (match_info.get("venueInfo", {}) or {}).get("ground", ""),
            "city": (match_info.get("venueInfo", {}) or {}).get("city", ""),
        },
    }


def _collect_player_name_list(value) -> list[str]:
    names: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    names.append(cleaned)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("fullName") or item.get("playerName") or item.get("nickName")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
    return names


def _extract_lineups_from_container(container: dict, aliases: set[str]) -> list[str]:
    for key in ("playingXI", "playingXi", "playing11", "players", "squad", "probableXI", "probableXi"):
        lineup = _collect_player_name_list(container.get(key))
        if len(lineup) >= 2:
            return lineup

    team_name = str(container.get("teamName") or container.get("name") or container.get("fullName") or "").strip().lower()
    if team_name and team_name in aliases:
        for key in ("players", "playingXI", "playingXi", "playing11", "squad", "probableXI", "probableXi"):
            lineup = _collect_player_name_list(container.get(key))
            if len(lineup) >= 2:
                return lineup
    return []


def _find_lineup_candidates(value, aliases: set[str], found: list[list[str]]) -> None:
    if isinstance(value, dict):
        lineup = _extract_lineups_from_container(value, aliases)
        if lineup:
            found.append(lineup)
        for nested in value.values():
            _find_lineup_candidates(nested, aliases, found)
    elif isinstance(value, list):
        for item in value:
            _find_lineup_candidates(item, aliases, found)


def extract_preview_lineups(data: dict, team_a: str, team_b: str, team_a_full: str = "", team_b_full: str = "") -> tuple[list[str], list[str]]:
    aliases_a = {alias.strip().lower() for alias in (team_a, team_a_full) if alias}
    aliases_b = {alias.strip().lower() for alias in (team_b, team_b_full) if alias}

    candidates_a: list[list[str]] = []
    candidates_b: list[list[str]] = []
    _find_lineup_candidates(data, aliases_a, candidates_a)
    _find_lineup_candidates(data, aliases_b, candidates_b)

    lineup_a = max(candidates_a, key=len, default=[])
    lineup_b = max(candidates_b, key=len, default=[])
    return lineup_a, lineup_b


def is_ipl_t20_match(match_info: dict) -> bool:
    series_name = str(match_info.get("seriesName") or "").lower()
    match_format = str(match_info.get("matchFormat") or match_info.get("matchType") or "").lower()
    if "indian premier league" not in series_name and "ipl" not in series_name:
        return False
    if match_format and "t20" not in match_format:
        return False
    return True

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
        team1_full = team1_info.get("teamName") or team1
        team2_full = team2_info.get("teamName") or team2
        venue_info = match_info.get("venueInfo", {})

        res = {
            "innings": 1, "target": 0, "current_score": 0, "wickets": 0, "overs_bowled": 0.0,
            "team_a": team1, "team_b": team2, "status": status,
            "is_complete": is_complete, "state": state,
            "batting_team": None,
            "wickets_in_hand": 10,
            "batters_at_crease": [],
            "predicted_score": 0,
            "prediction_note": "Awaiting live data",
            "prediction_explanation": "HatCric is waiting for enough match context to explain the edge.",
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

            res["batters_at_crease"] = extract_batters_at_crease(last_innings)
              
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
        res["wickets_in_hand"] = max(0, 10 - wickets)

        if res["state"] == "Preview":
            lineup_a, lineup_b = extract_preview_lineups(data, res["team_a"], res["team_b"], team1_full, team2_full)
            res["win_probability"] = predict_match_probability(
                res["team_a"],
                res["team_b"],
                venue_info,
                status=status,
                team_a_full=team1_full,
                team_b_full=team2_full,
                team_a_lineup=lineup_a,
                team_b_lineup=lineup_b,
            )
            if lineup_a and lineup_b:
                res["prediction_note"] = "ML pre-match model using announced lineups"
            else:
                res["prediction_note"] = "ML pre-match model based on historical IPL results"
            res["prediction_explanation"] = build_prediction_explanation(
                state=res["state"],
                is_complete=res["is_complete"],
                team_a=res["team_a"],
                team_b=res["team_b"],
                team_a_pct=int(res["win_probability"]["team_a"]),
                team_b_pct=int(res["win_probability"]["team_b"]),
                status=status,
                used_lineups=bool(lineup_a and lineup_b),
            )

        if not res["is_complete"] and res["state"] != "Preview":
            live_probability = predict_live_win_probability(
                team_a=res["team_a"],
                team_b=res["team_b"],
                batting_team=res["batting_team"] or res["team_a"],
                innings=int(res["innings"] or 1),
                runs=runs,
                wickets=wickets,
                overs_bowled=overs_bowled,
                target=int(res["target"] or 0),
                venue=venue_info,
            )
            if res["innings"] == 2 and res["target"] > 0:
                if live_probability is not None:
                    res["win_probability"] = live_probability
                else:
                    chasing_win = chase_win_probability(runs, wickets, overs_bowled, int(res["target"]))
                    if res["batting_team"] == res["team_a"]:
                        res["win_probability"] = {"team_a": chasing_win, "team_b": 100 - chasing_win}
                    else:
                        res["win_probability"] = {"team_a": 100 - chasing_win, "team_b": chasing_win}
                res["predicted_score"] = int(res["target"])
                balls_remaining = max(0, 120 - overs_to_balls(overs_bowled))
                runs_needed = max(0, int(res["target"]) - runs)
                req_rr = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0
                note_prefix = "Live ML chase model" if live_probability is not None else "Chase model"
                res["prediction_note"] = f"{note_prefix}: {runs_needed} needed from {balls_remaining} balls at {req_rr:.2f} RPO"
                res["prediction_explanation"] = build_prediction_explanation(
                    state=res["state"],
                    is_complete=res["is_complete"],
                    team_a=res["team_a"],
                    team_b=res["team_b"],
                    team_a_pct=int(res["win_probability"]["team_a"]),
                    team_b_pct=int(res["win_probability"]["team_b"]),
                    status=status,
                    batting_team=res["batting_team"],
                    innings=int(res["innings"]),
                    target=int(res["target"]),
                    runs=runs,
                    wickets=wickets,
                    overs_bowled=overs_bowled,
                    predicted_score=int(res["predicted_score"]),
                )
            else:
                res["predicted_score"] = project_first_innings_score(runs, wickets, overs_bowled)
                if live_probability is not None:
                    res["win_probability"] = live_probability
                else:
                    batting_win = first_innings_win_probability(res["predicted_score"], overs_bowled, wickets)
                    if res["batting_team"] == res["team_b"]:
                        res["win_probability"] = {"team_a": 100 - batting_win, "team_b": batting_win}
                    else:
                        res["win_probability"] = {"team_a": batting_win, "team_b": 100 - batting_win}
                balls_remaining = max(0, 120 - overs_to_balls(overs_bowled))
                note_prefix = "Live ML innings model" if live_probability is not None else "Projected total"
                res["prediction_note"] = f"{note_prefix} with {balls_remaining} balls remaining"
                res["prediction_explanation"] = build_prediction_explanation(
                    state=res["state"],
                    is_complete=res["is_complete"],
                    team_a=res["team_a"],
                    team_b=res["team_b"],
                    team_a_pct=int(res["win_probability"]["team_a"]),
                    team_b_pct=int(res["win_probability"]["team_b"]),
                    status=status,
                    batting_team=res["batting_team"],
                    innings=int(res["innings"]),
                    target=int(res["target"]),
                    runs=runs,
                    wickets=wickets,
                    overs_bowled=overs_bowled,
                    predicted_score=int(res["predicted_score"]),
                )

        # 4. WIN PROBABILITY MATH FOR COMPLETED MATCHES
        if is_complete:
            winner = detect_winner(status, res["team_a"], res["team_b"], team1_full, team2_full)
            if winner == "team_a":
                res["win_probability"] = {"team_a": 100, "team_b": 0}
            elif winner == "team_b":
                res["win_probability"] = {"team_a": 0, "team_b": 100}
            else:
                res["win_probability"] = {"team_a": 50, "team_b": 50}
            res["predicted_score"] = runs
            res["prediction_note"] = status
            res["prediction_explanation"] = build_prediction_explanation(
                state=res["state"],
                is_complete=res["is_complete"],
                team_a=res["team_a"],
                team_b=res["team_b"],
                team_a_pct=int(res["win_probability"]["team_a"]),
                team_b_pct=int(res["win_probability"]["team_b"]),
                status=status,
            )

        return res

    except Exception as e:
        logger.error(f"Parser Error: {e}")
        return {
            "innings": 1,
            "target": 0,
            "current_score": 0,
            "wickets": 0,
            "wickets_in_hand": 10,
            "overs_bowled": 0,
            "is_complete": False,
            "state": "Error",
            "batters_at_crease": [],
            "prediction_explanation": "HatCric could not explain the edge because the match feed failed to parse.",
            "win_probability": {"team_a": 50, "team_b": 50},
        }


@app.route("/api/matches", methods=["GET"])
def get_matches():
    try:
        serialized = []
        for match_info in fetch_selectable_matches():
            if not is_ipl_t20_match(match_info):
                continue
            item = serialize_match_info(match_info)
            if item:
                serialized.append(item)

        state_order = {"Live": 0, "In Progress": 0, "Preview": 1, "Complete": 2}
        serialized.sort(
            key=lambda item: (
                state_order.get(item["state"], 3),
                item["label"],
            )
        )

        return jsonify({
            "matches": serialized,
            "count": len(serialized),
        })
    except Exception as exc:
        logger.error("Failed to fetch match list: %s", exc)
        return jsonify({"error": "Unable to fetch match list", "matches": []}), 500


@app.route("/api/match/<string:match_id>", methods=["GET"])
def get_match_data(match_id: str):
    if match_id == "mock_test":
        return jsonify({
            "match_id": "mock_test", "innings": 2, "target": 185, "current_score": 120,
            "wickets": 3, "overs_bowled": 14.2, "team_a": "RR", "team_b": "MI",
            "is_complete": False, "state": "Live", "status": "Live Match",
            "batting_team": "MI",
            "wickets_in_hand": 7,
            "batters_at_crease": [
                {"name": "Yashasvi Jaiswal", "runs": 41, "balls": 24, "fours": 5, "sixes": 2, "strike_rate": "170.83", "is_batting": True},
                {"name": "Sanju Samson", "runs": 36, "balls": 29, "fours": 3, "sixes": 1, "strike_rate": "124.14", "is_batting": True},
            ],
            "predicted_score": 185,
            "prediction_note": "65 needed from 34 balls at 11.47 RPO",
            "prediction_explanation": "HatCric leans MI because the chase is alive, wickets are in hand, and the required rate is still manageable.",
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
