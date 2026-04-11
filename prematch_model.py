import json
import math
import random
import re
import zipfile
from datetime import date
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency managed by requirements
    pd = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - dependency managed by requirements
    CatBoostClassifier = None


TEAM_NAME_TO_CODE = {
    "chennai super kings": "CSK",
    "mumbai indians": "MI",
    "royal challengers bengaluru": "RCB",
    "royal challengers bangalore": "RCB",
    "kolkata knight riders": "KKR",
    "sunrisers hyderabad": "SRH",
    "delhi capitals": "DC",
    "delhi daredevils": "DC",
    "rajasthan royals": "RR",
    "gujarat titans": "GT",
    "gujarat lions": "GL",
    "punjab kings": "PBKS",
    "kings xi punjab": "PBKS",
    "lucknow super giants": "LSG",
    "deccan chargers": "DEC",
    "rising pune supergiant": "RPS",
    "rising pune supergiants": "RPS",
    "pune warriors": "PWI",
    "kochi tuskers kerala": "KTK",
}

CURRENT_TEAM_CODES = {"CSK", "MI", "RCB", "KKR", "SRH", "DC", "RR", "GT", "PBKS", "LSG"}

HOME_VENUE_KEYWORDS = {
    "CSK": ["chennai", "chepauk"],
    "MI": ["mumbai", "wankhede"],
    "RCB": ["bengaluru", "bangalore", "chinnaswamy"],
    "KKR": ["kolkata", "eden gardens"],
    "SRH": ["hyderabad", "uppal", "rajiv gandhi"],
    "DC": ["delhi", "arun jaitley", "feroz shah kotla"],
    "RR": ["jaipur", "sawai mansingh"],
    "GT": ["ahmedabad", "narendra modi"],
    "PBKS": ["mullanpur", "mohali", "new chandigarh", "dharamsala"],
    "LSG": ["lucknow", "ekana"],
}

MODEL_PATH = Path(__file__).with_name("prematch_model.json")
CATBOOST_MODEL_PATH = Path(__file__).with_name("prematch_model.cbm")
BASE_RATING = 1500.0
FORM_WINDOW = 5
VALIDATION_SPLIT = 0.15
SEASON_REGRESSION_FACTOR = 0.65
CATEGORICAL_FEATURES = ["team_a_code", "team_b_code", "venue_key", "venue_city", "venue_ground", "season"]


def normalize_team_code(name: str) -> str | None:
    if not name:
        return None
    name_l = name.strip().lower()
    if len(name_l) <= 4 and name_l.upper() in CURRENT_TEAM_CODES:
        return name_l.upper()
    return TEAM_NAME_TO_CODE.get(name_l)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _dot(weights: dict[str, float], features: dict[str, float | str]) -> float:
    return sum(
        weights.get(key, 0.0) * float(value)
        for key, value in features.items()
        if isinstance(value, (int, float))
    )


def _alias_in_text(alias: str, text_l: str) -> bool:
    alias = (alias or "").strip().lower()
    if not alias:
        return False
    if len(alias) <= 4 and alias.replace(" ", "").isalpha():
        return re.search(rf"\b{re.escape(alias)}\b", text_l) is not None
    return alias in text_l


def _normalize_rating(raw_rating: float) -> float:
    return (raw_rating - BASE_RATING) / 100.0


def _recent_form_score(results: list[int]) -> float:
    if not results:
        return 0.0
    return (sum(results) / len(results)) - 0.5


def _head_to_head_score(a_wins: int, b_wins: int) -> float:
    total = a_wins + b_wins
    if total == 0:
        return 0.0
    return (a_wins - b_wins) / total


def _normalize_venue_key(venue: dict | None) -> str:
    ground = str((venue or {}).get("ground", "")).strip().lower()
    city = str((venue or {}).get("city", "")).strip().lower()
    return " | ".join(part for part in (ground, city) if part)


def _smoothed_rate(wins: int, games: int) -> float:
    if games <= 0:
        return 0.5
    return (wins + 1.5) / (games + 3.0)


def _smoothed_nrr(runs_for: int, balls_faced: int, runs_against: int, balls_bowled: int) -> float:
    if balls_faced <= 0 or balls_bowled <= 0:
        return 0.0
    run_rate_for = runs_for / (balls_faced / 6.0)
    run_rate_against = runs_against / (balls_bowled / 6.0)
    return run_rate_for - run_rate_against


def _blank_season_stats() -> dict[str, float]:
    return {
        "matches": 0,
        "wins": 0,
        "points": 0,
        "runs_for": 0,
        "balls_faced": 0,
        "runs_against": 0,
        "balls_bowled": 0,
    }


def _blank_venue_season_stats() -> dict[str, float]:
    return {
        "matches": 0,
        "chase_wins": 0,
        "first_innings_runs": 0,
        "first_innings_count": 0,
    }


def _blank_player_stats() -> dict[str, float]:
    return {
        "bat_runs": 0,
        "bat_balls": 0,
        "bat_matches": 0,
        "bowl_runs": 0,
        "bowl_balls": 0,
        "wickets": 0,
        "bowl_matches": 0,
    }


def _normalize_player_name(name: str) -> str:
    return str(name or "").strip().lower()


def _batting_rating(stats: dict[str, float]) -> float:
    balls = float(stats.get("bat_balls", 0))
    runs = float(stats.get("bat_runs", 0))
    matches = float(stats.get("bat_matches", 0))
    if balls <= 0:
        return 0.0
    average_runs = runs / max(1.0, matches)
    strike_rate = (runs / balls) * 100.0
    return (average_runs / 18.0) + ((strike_rate - 120.0) / 80.0)


def _bowling_rating(stats: dict[str, float]) -> float:
    balls = float(stats.get("bowl_balls", 0))
    runs = float(stats.get("bowl_runs", 0))
    wickets = float(stats.get("wickets", 0))
    matches = float(stats.get("bowl_matches", 0))
    if balls <= 0:
        return 0.0
    overs = balls / 6.0
    economy = runs / max(overs, 1.0)
    wickets_per_match = wickets / max(1.0, matches)
    return (wickets_per_match / 1.2) + ((8.0 - economy) / 4.0)


def _compute_lineup_strength(player_names: list[str], player_stats: dict[str, dict[str, float]]) -> dict[str, float]:
    if not player_names:
        return {"batting": 0.0, "bowling": 0.0, "balance": 0.0}

    batting_scores = []
    bowling_scores = []
    for player in player_names:
        stats = player_stats.get(_normalize_player_name(player), _blank_player_stats())
        batting_scores.append(_batting_rating(stats))
        bowling_scores.append(_bowling_rating(stats))

    batting_scores.sort(reverse=True)
    bowling_scores.sort(reverse=True)
    batting_strength = sum(batting_scores[:7]) / max(1, min(7, len(batting_scores)))
    bowling_strength = sum(bowling_scores[:5]) / max(1, min(5, len(bowling_scores)))
    return {
        "batting": batting_strength,
        "bowling": bowling_strength,
        "balance": batting_strength - bowling_strength,
    }


def _player_strengths_from_stats(player_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    strengths: dict[str, dict[str, float]] = {}
    for player_name, stats in player_stats.items():
        strengths[player_name] = {
            "batting": round(_batting_rating(stats), 6),
            "bowling": round(_bowling_rating(stats), 6),
        }
    return strengths


def _compute_lineup_strength_from_priors(
    player_names: list[str],
    player_strengths: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not player_names:
        return {"batting": 0.0, "bowling": 0.0, "balance": 0.0}

    batting_scores = []
    bowling_scores = []
    for player in player_names:
        strengths = player_strengths.get(_normalize_player_name(player), {"batting": 0.0, "bowling": 0.0})
        batting_scores.append(float(strengths.get("batting", 0.0)))
        bowling_scores.append(float(strengths.get("bowling", 0.0)))

    batting_scores.sort(reverse=True)
    bowling_scores.sort(reverse=True)
    batting_strength = sum(batting_scores[:7]) / max(1, min(7, len(batting_scores)))
    bowling_strength = sum(bowling_scores[:5]) / max(1, min(5, len(bowling_scores)))
    return {
        "batting": batting_strength,
        "bowling": bowling_strength,
        "balance": batting_strength - bowling_strength,
    }


def _pair_key(team_a_code: str, team_b_code: str) -> tuple[str, str]:
    return tuple(sorted((team_a_code, team_b_code)))


def _pair_record_from_perspective(pair_record: dict[str, int], team_a_code: str, team_b_code: str) -> tuple[int, int]:
    wins_a = pair_record.get(team_a_code, 0)
    wins_b = pair_record.get(team_b_code, 0)
    return wins_a, wins_b


def infer_home_advantage(team_a_code: str, team_b_code: str, venue: dict | None) -> int:
    venue_text = " ".join([
        str((venue or {}).get("ground", "")),
        str((venue or {}).get("city", "")),
    ]).lower()
    team_a_home = any(keyword in venue_text for keyword in HOME_VENUE_KEYWORDS.get(team_a_code, []))
    team_b_home = any(keyword in venue_text for keyword in HOME_VENUE_KEYWORDS.get(team_b_code, []))
    if team_a_home and not team_b_home:
        return 1
    if team_b_home and not team_a_home:
        return -1
    return 0


def infer_toss_context(
    status: str,
    team_a_code: str,
    team_b_code: str,
    team_a_full: str = "",
    team_b_full: str = "",
) -> tuple[int, int]:
    status_l = (status or "").lower()
    if all(token not in status_l for token in ("opt to bat", "opt to bowl", "elected to bat", "elected to field")):
        return 0, 0

    toss_sign = 0
    if _alias_in_text(team_a_code, status_l) or _alias_in_text(team_a_full, status_l):
        toss_sign = 1
    elif _alias_in_text(team_b_code, status_l) or _alias_in_text(team_b_full, status_l):
        toss_sign = -1

    decision_sign = 0
    if "opt to bat" in status_l or "elected to bat" in status_l:
        decision_sign = 1
    elif "opt to bowl" in status_l or "elected to field" in status_l:
        decision_sign = -1

    return toss_sign, toss_sign * decision_sign if toss_sign else 0


def build_contextual_features(
    team_a_code: str,
    team_b_code: str,
    venue: dict | None = None,
    toss_sign: int = 0,
    toss_decision_sign: int = 0,
    team_ratings: dict[str, float] | None = None,
    recent_form: dict[str, list[int]] | None = None,
    season_form: dict[str, list[int]] | None = None,
    season_stats: dict[str, dict[str, float]] | None = None,
    season_venue_stats: dict[str, dict[str, float]] | None = None,
    team_lineup_strengths: dict[str, dict[str, float]] | None = None,
    head_to_head: dict[tuple[str, str], dict[str, int]] | None = None,
    venue_history: dict[str, dict[str, int]] | None = None,
) -> dict[str, float]:
    ratings = team_ratings or {}
    forms = recent_form or {}
    seasonal_forms = season_form or {}
    seasonal_stats = season_stats or {}
    seasonal_venues = season_venue_stats or {}
    lineup_strengths = team_lineup_strengths or {}
    pair_history = head_to_head or {}
    venues = venue_history or {}
    pair_key = _pair_key(team_a_code, team_b_code)
    pair_record = pair_history.get(pair_key, {})
    wins_a, wins_b = _pair_record_from_perspective(pair_record, team_a_code, team_b_code)

    rating_a = ratings.get(team_a_code, BASE_RATING)
    rating_b = ratings.get(team_b_code, BASE_RATING)
    form_a = _recent_form_score(forms.get(team_a_code, []))
    form_b = _recent_form_score(forms.get(team_b_code, []))
    season_form_a = _recent_form_score(seasonal_forms.get(team_a_code, []))
    season_form_b = _recent_form_score(seasonal_forms.get(team_b_code, []))
    stats_a = seasonal_stats.get(team_a_code, _blank_season_stats())
    stats_b = seasonal_stats.get(team_b_code, _blank_season_stats())
    lineup_a = lineup_strengths.get(team_a_code, {"batting": 0.0, "bowling": 0.0, "balance": 0.0})
    lineup_b = lineup_strengths.get(team_b_code, {"batting": 0.0, "bowling": 0.0, "balance": 0.0})
    venue_key = _normalize_venue_key(venue)
    venue_stats = seasonal_venues.get(venue_key, _blank_venue_season_stats())
    venue_a = venues.get(f"{team_a_code}|{venue_key}", {"wins": 0, "games": 0})
    venue_b = venues.get(f"{team_b_code}|{venue_key}", {"wins": 0, "games": 0})
    venue_rate_a = _smoothed_rate(venue_a.get("wins", 0), venue_a.get("games", 0))
    venue_rate_b = _smoothed_rate(venue_b.get("wins", 0), venue_b.get("games", 0))
    home_advantage = float(infer_home_advantage(team_a_code, team_b_code, venue))
    rating_diff = _normalize_rating(rating_a - rating_b + BASE_RATING) - _normalize_rating(BASE_RATING)
    form_diff = form_a - form_b
    season_form_diff = season_form_a - season_form_b
    venue_record_diff = venue_rate_a - venue_rate_b
    head_to_head_diff = _head_to_head_score(wins_a, wins_b)
    points_diff = float(stats_a.get("points", 0) - stats_b.get("points", 0)) / 4.0
    win_pct_diff = _smoothed_rate(int(stats_a.get("wins", 0)), int(stats_a.get("matches", 0))) - _smoothed_rate(
        int(stats_b.get("wins", 0)),
        int(stats_b.get("matches", 0)),
    )
    season_nrr_diff = _smoothed_nrr(
        int(stats_a.get("runs_for", 0)),
        int(stats_a.get("balls_faced", 0)),
        int(stats_a.get("runs_against", 0)),
        int(stats_a.get("balls_bowled", 0)),
    ) - _smoothed_nrr(
        int(stats_b.get("runs_for", 0)),
        int(stats_b.get("balls_faced", 0)),
        int(stats_b.get("runs_against", 0)),
        int(stats_b.get("balls_bowled", 0)),
    )
    venue_matches = int(venue_stats.get("matches", 0))
    venue_chase_bias = (
        (float(venue_stats.get("chase_wins", 0)) / venue_matches) - 0.5
        if venue_matches > 0 else 0.0
    )
    venue_avg_first_innings = (
        float(venue_stats.get("first_innings_runs", 0)) / float(venue_stats.get("first_innings_count", 0))
        if int(venue_stats.get("first_innings_count", 0)) > 0 else 0.0
    )
    venue_scoring_bias = (venue_avg_first_innings - 170.0) / 20.0 if venue_avg_first_innings else 0.0
    venue_sample_strength = min(1.0, venue_matches / 10.0)
    batting_strength_diff = float(lineup_a.get("batting", 0.0)) - float(lineup_b.get("batting", 0.0))
    bowling_strength_diff = float(lineup_a.get("bowling", 0.0)) - float(lineup_b.get("bowling", 0.0))
    balance_diff = float(lineup_a.get("balance", 0.0)) - float(lineup_b.get("balance", 0.0))

    features = {
        "bias": 1.0,
        "home_advantage": home_advantage,
        "toss_sign": float(toss_sign),
        "toss_decision_sign": float(toss_decision_sign),
        "rating_diff": rating_diff,
        "rating_ratio": (rating_a / rating_b) - 1.0 if rating_b else 0.0,
        "form_diff": form_diff,
        "season_form_diff": season_form_diff,
        "season_points_diff": points_diff,
        "season_win_pct_diff": win_pct_diff,
        "season_nrr_diff": season_nrr_diff,
        "batting_strength_diff": batting_strength_diff,
        "bowling_strength_diff": bowling_strength_diff,
        "balance_diff": balance_diff,
        "venue_chase_bias": venue_chase_bias,
        "venue_scoring_bias": venue_scoring_bias,
        "venue_sample_strength": venue_sample_strength,
        "venue_record_diff": venue_record_diff,
        "head_to_head_diff": head_to_head_diff,
        "experience_diff": float(len(forms.get(team_a_code, [])) - len(forms.get(team_b_code, []))) / FORM_WINDOW,
        "rating_x_home": rating_diff * home_advantage,
        "rating_x_form": rating_diff * form_diff,
        "form_x_head_to_head": form_diff * head_to_head_diff,
        "venue_x_home": venue_record_diff * home_advantage,
        "season_x_toss": season_form_diff * float(toss_sign),
        "season_points_x_home": points_diff * home_advantage,
        "season_nrr_x_form": season_nrr_diff * form_diff,
        "venue_chase_x_toss_decision": venue_chase_bias * float(toss_decision_sign),
        "venue_scoring_x_home": venue_scoring_bias * home_advantage,
        "venue_scoring_x_form": venue_scoring_bias * season_form_diff,
        "venue_chase_x_record": venue_chase_bias * venue_record_diff,
        "venue_strength_x_home": venue_sample_strength * home_advantage,
        "batting_vs_bowling_edge": batting_strength_diff - bowling_strength_diff,
        "balance_x_home": balance_diff * home_advantage,
    }
    return features


def build_feature_row(
    team_a_code: str,
    team_b_code: str,
    venue: dict | None = None,
    season_key: str | None = None,
    toss_sign: int = 0,
    toss_decision_sign: int = 0,
    team_ratings: dict[str, float] | None = None,
    recent_form: dict[str, list[int]] | None = None,
    season_form: dict[str, list[int]] | None = None,
    season_stats: dict[str, dict[str, float]] | None = None,
    season_venue_stats: dict[str, dict[str, float]] | None = None,
    team_lineup_strengths: dict[str, dict[str, float]] | None = None,
    head_to_head: dict[tuple[str, str], dict[str, int]] | None = None,
    venue_history: dict[str, dict[str, int]] | None = None,
) -> dict[str, float | str]:
    venue_ground = str((venue or {}).get("ground", "")).strip().lower() or "unknown"
    venue_city = str((venue or {}).get("city", "")).strip().lower() or "unknown"
    venue_key = _normalize_venue_key(venue) or "unknown"
    features = build_contextual_features(
        team_a_code,
        team_b_code,
        venue=venue,
        toss_sign=toss_sign,
        toss_decision_sign=toss_decision_sign,
        team_ratings=team_ratings,
        recent_form=recent_form,
        season_form=season_form,
        season_stats=season_stats,
        season_venue_stats=season_venue_stats,
        team_lineup_strengths=team_lineup_strengths,
        head_to_head=head_to_head,
        venue_history=venue_history,
    )
    row: dict[str, float | str] = {
        "team_a_code": team_a_code,
        "team_b_code": team_b_code,
        "venue_key": venue_key,
        "venue_city": venue_city,
        "venue_ground": venue_ground,
        "season": str(season_key or date.today().year),
    }
    row.update(features)
    return row


def load_model(model_path: Path | None = None) -> dict | None:
    path = model_path or MODEL_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_CATBOOST_CACHE: object | None | bool = None


def load_catboost_model(model_path: Path | None = None):
    global _CATBOOST_CACHE
    path = model_path or CATBOOST_MODEL_PATH
    if CatBoostClassifier is None or not path.exists():
        return None
    if _CATBOOST_CACHE is False:
        return None
    if _CATBOOST_CACHE is not None:
        return _CATBOOST_CACHE

    model = CatBoostClassifier()
    model.load_model(str(path))
    _CATBOOST_CACHE = model
    return model


def predict_match_probability(
    team_a_code: str,
    team_b_code: str,
    venue: dict | None = None,
    status: str = "",
    team_a_full: str = "",
    team_b_full: str = "",
    team_a_lineup: list[str] | None = None,
    team_b_lineup: list[str] | None = None,
    model: dict | None = None,
) -> dict[str, int]:
    loaded = model or load_model()
    if not loaded:
        return {"team_a": 50, "team_b": 50}

    team_ratings = loaded.get("team_ratings", {})
    recent_form = loaded.get("recent_form", {})
    serialized_h2h = loaded.get("head_to_head", {})
    head_to_head = {
        tuple(key.split("|")): value
        for key, value in serialized_h2h.items()
        if "|" in key
    }
    season_form = loaded.get("season_form", {})
    season_stats = loaded.get("season_stats", {})
    season_venue_stats = loaded.get("season_venue_stats", {})
    team_lineup_strengths = loaded.get("team_lineup_strengths", {})
    player_strengths = loaded.get("player_strengths", {})
    venue_history = loaded.get("venue_history", {})

    if team_a_lineup:
        team_lineup_strengths = dict(team_lineup_strengths)
        team_lineup_strengths[team_a_code] = _compute_lineup_strength_from_priors(team_a_lineup, player_strengths)
    if team_b_lineup:
        team_lineup_strengths = dict(team_lineup_strengths)
        team_lineup_strengths[team_b_code] = _compute_lineup_strength_from_priors(team_b_lineup, player_strengths)

    toss_sign, toss_decision_sign = infer_toss_context(status, team_a_code, team_b_code, team_a_full, team_b_full)
    rating_a = team_ratings.get(team_a_code, BASE_RATING)
    rating_b = team_ratings.get(team_b_code, BASE_RATING)
    elo_probability = _expected_from_ratings(rating_a, rating_b)
    blend_alpha = loaded.get("blend_alpha", 1.0)

    probability = None
    catboost_model = load_catboost_model()
    if loaded.get("model_type") == "catboost_classifier" and catboost_model is not None and pd is not None:
        row = build_feature_row(
            team_a_code,
            team_b_code,
            venue=venue,
            season_key=str(date.today().year),
            toss_sign=toss_sign,
            toss_decision_sign=toss_decision_sign,
            team_ratings=team_ratings,
            recent_form=recent_form,
            season_form=season_form,
            season_stats=season_stats,
            season_venue_stats=season_venue_stats,
            team_lineup_strengths=team_lineup_strengths,
            head_to_head=head_to_head,
            venue_history=venue_history,
        )
        catboost_probability = float(catboost_model.predict_proba(pd.DataFrame([row]))[0][1])
        probability = (blend_alpha * catboost_probability) + ((1.0 - blend_alpha) * elo_probability)
    else:
        features = build_contextual_features(
            team_a_code,
            team_b_code,
            venue,
            toss_sign,
            toss_decision_sign,
            team_ratings=team_ratings,
            recent_form=recent_form,
            season_form=season_form,
            season_stats=season_stats,
            season_venue_stats=season_venue_stats,
            team_lineup_strengths=team_lineup_strengths,
            head_to_head=head_to_head,
            venue_history=venue_history,
        )
        logistic_probability = _sigmoid(_dot(loaded.get("weights", {}), features))
        probability = (blend_alpha * logistic_probability) + ((1.0 - blend_alpha) * elo_probability)

    low = loaded.get("prediction_floor", 18)
    high = loaded.get("prediction_ceiling", 82)
    team_a_pct = round(max(low, min(high, probability * 100)))
    if low < team_a_pct < high and team_a_pct == 50:
        team_a_pct = 51 if probability >= 0.5 else 49
    return {"team_a": team_a_pct, "team_b": 100 - team_a_pct}


def _parse_match_date(raw_info: dict) -> date:
    dates = raw_info.get("dates") or []
    if dates:
        return date.fromisoformat(str(dates[0]))
    return date(2008, 1, 1)


def _extract_innings_totals(raw_match: dict) -> dict[str, dict[str, int]]:
    team_totals: dict[str, dict[str, int]] = {}
    for innings in raw_match.get("innings", []):
        team_code = normalize_team_code(innings.get("team", ""))
        if not team_code:
            continue

        runs = 0
        legal_balls = 0
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                runs += int(delivery.get("runs", {}).get("total", 0))
                extras = delivery.get("extras", {}) or {}
                if "wides" not in extras and "noballs" not in extras:
                    legal_balls += 1

        team_totals[team_code] = {"runs": runs, "balls": legal_balls}
    return team_totals


def _extract_match_sequence(raw_match: dict) -> list[str]:
    sequence = []
    for innings in raw_match.get("innings", []):
        team_code = normalize_team_code(innings.get("team", ""))
        if team_code:
            sequence.append(team_code)
    return sequence


def _extract_lineups(raw_match: dict) -> dict[str, list[str]]:
    players = raw_match.get("info", {}).get("players", {}) or {}
    lineups: dict[str, list[str]] = {}
    for team_name, squad in players.items():
        team_code = normalize_team_code(team_name)
        if team_code and isinstance(squad, list):
            lineups[team_code] = [str(player) for player in squad]
    return lineups


def _extract_player_match_stats(raw_match: dict) -> dict[str, dict[str, float]]:
    player_stats: dict[str, dict[str, float]] = {}
    appeared_as_batter: set[str] = set()
    appeared_as_bowler: set[str] = set()

    for innings in raw_match.get("innings", []):
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                batter = _normalize_player_name(delivery.get("batter", ""))
                bowler = _normalize_player_name(delivery.get("bowler", ""))
                extras = delivery.get("extras", {}) or {}
                runs = delivery.get("runs", {}) or {}

                if batter:
                    stats = player_stats.setdefault(batter, _blank_player_stats())
                    stats["bat_runs"] += int(runs.get("batter", 0))
                    if "wides" not in extras:
                        stats["bat_balls"] += 1
                    appeared_as_batter.add(batter)

                if bowler:
                    stats = player_stats.setdefault(bowler, _blank_player_stats())
                    conceded = int(runs.get("total", 0)) - int(extras.get("byes", 0)) - int(extras.get("legbyes", 0))
                    stats["bowl_runs"] += max(0, conceded)
                    if "wides" not in extras and "noballs" not in extras:
                        stats["bowl_balls"] += 1
                    for wicket in delivery.get("wickets", []) or []:
                        kind = str(wicket.get("kind", "")).lower()
                        if kind not in {"run out", "retired hurt", "retired out", "obstructing the field"}:
                            stats["wickets"] += 1
                    appeared_as_bowler.add(bowler)

    for player in appeared_as_batter:
        player_stats.setdefault(player, _blank_player_stats())["bat_matches"] += 1
    for player in appeared_as_bowler:
        player_stats.setdefault(player, _blank_player_stats())["bowl_matches"] += 1

    return player_stats


def _expected_from_ratings(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_elo(rating_a: float, rating_b: float, label: float, k_factor: float) -> tuple[float, float]:
    expected_a = _expected_from_ratings(rating_a, rating_b)
    delta = k_factor * (label - expected_a)
    return rating_a + delta, rating_b - delta


def _regress_ratings_for_new_season(team_ratings: dict[str, float], factor: float = SEASON_REGRESSION_FACTOR) -> None:
    for team_code, rating in list(team_ratings.items()):
        team_ratings[team_code] = BASE_RATING + ((rating - BASE_RATING) * factor)


def _append_recent_result(history: dict[str, list[int]], team_code: str, won: bool) -> None:
    bucket = history.setdefault(team_code, [])
    bucket.append(1 if won else 0)
    if len(bucket) > FORM_WINDOW:
        del bucket[0]


def _update_venue_history(
    venue_history: dict[str, dict[str, int]],
    team_code: str,
    venue: dict | None,
    won: bool,
) -> None:
    venue_key = _normalize_venue_key(venue)
    if not venue_key:
        return
    record = venue_history.setdefault(f"{team_code}|{venue_key}", {"wins": 0, "games": 0})
    record["games"] += 1
    if won:
        record["wins"] += 1


def _train_logistic(
    feature_rows: list[tuple[dict[str, float], float, float, float]],
    epochs: int,
    lr: float,
    l2: float,
) -> dict[str, float]:
    rng = random.Random(42)
    weights: dict[str, float] = {}
    for features, _, _, _ in feature_rows:
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                weights.setdefault(feature, 0.0)

    for _ in range(epochs):
        shuffled = feature_rows[:]
        rng.shuffle(shuffled)
        for features, label, sample_weight, _ in shuffled:
            prediction = _sigmoid(_dot(weights, features))
            error = (prediction - label) * sample_weight
            for feature, value in features.items():
                if isinstance(value, (int, float)):
                    weights[feature] -= lr * (error * float(value) + l2 * weights.get(feature, 0.0))
    return weights


def _evaluate_rows(
    feature_rows: list[tuple[dict[str, float], float, float, float]],
    weights: dict[str, float],
    blend_alpha: float = 1.0,
) -> tuple[float, float]:
    if not feature_rows:
        return 0.0, 0.0

    correct = 0
    weighted_loss = 0.0
    for features, label, sample_weight, elo_probability in feature_rows:
        logistic_probability = _sigmoid(_dot(weights, features))
        prediction = (blend_alpha * logistic_probability) + ((1.0 - blend_alpha) * elo_probability)
        predicted_label = 1.0 if prediction >= 0.5 else 0.0
        if predicted_label == label:
            correct += 1
        prediction = min(max(prediction, 1e-7), 1 - 1e-7)
        weighted_loss += sample_weight * (-(label * math.log(prediction) + (1 - label) * math.log(1 - prediction)))

    return round(correct / len(feature_rows), 4), round(weighted_loss / len(feature_rows), 4)


def _select_blend_alpha(
    validation_rows: list[tuple[dict[str, float], float, float, float]],
    weights: dict[str, float],
) -> float:
    if not validation_rows:
        return 1.0

    best_alpha = 1.0
    best_loss = None
    for step in range(0, 21):
        alpha = step / 20.0
        _, loss = _evaluate_rows(validation_rows, weights, blend_alpha=alpha)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_alpha = alpha
    return best_alpha


def _evaluate_probabilities(
    labels: list[float],
    probabilities: list[float],
    sample_weights: list[float],
) -> tuple[float, float]:
    if not labels:
        return 0.0, 0.0

    correct = 0
    weighted_loss = 0.0
    for label, prediction, sample_weight in zip(labels, probabilities, sample_weights):
        predicted_label = 1.0 if prediction >= 0.5 else 0.0
        if predicted_label == label:
            correct += 1
        clipped = min(max(prediction, 1e-7), 1 - 1e-7)
        weighted_loss += sample_weight * (-(label * math.log(clipped) + (1 - label) * math.log(1 - clipped)))

    return round(correct / len(labels), 4), round(weighted_loss / len(labels), 4)


def _select_blend_alpha_from_probs(
    labels: list[float],
    model_probabilities: list[float],
    elo_probabilities: list[float],
    sample_weights: list[float],
) -> float:
    if not labels:
        return 1.0

    best_alpha = 1.0
    best_loss = None
    for step in range(0, 21):
        alpha = step / 20.0
        blended = [
            (alpha * model_prob) + ((1.0 - alpha) * elo_prob)
            for model_prob, elo_prob in zip(model_probabilities, elo_probabilities)
        ]
        _, loss = _evaluate_probabilities(labels, blended, sample_weights)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_alpha = alpha
    return best_alpha


def train_from_cricsheet_zip(
    zip_path: Path,
    output_path: Path | None = None,
    epochs: int = 650,
    lr: float = 0.02,
    l2: float = 8.0,
    depth: int = 7,
    random_strength: float = 1.5,
) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        raw_matches = []
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue

            raw = json.loads(zf.read(name).decode("utf-8"))
            info = raw.get("info", {})
            event = info.get("event", {})
            if event.get("name") != "Indian Premier League":
                continue

            teams = info.get("teams", [])
            outcome = info.get("outcome", {})
            winner = outcome.get("winner")
            if len(teams) != 2 or not winner:
                continue

            team_a = normalize_team_code(teams[0])
            team_b = normalize_team_code(teams[1])
            winner_code = normalize_team_code(winner)
            if not team_a or not team_b or not winner_code:
                continue

            toss = info.get("toss", {})
            toss_winner = normalize_team_code(toss.get("winner", ""))
            toss_sign = 1 if toss_winner == team_a else -1 if toss_winner == team_b else 0
            toss_decision = toss.get("decision", "")
            toss_decision_sign = 0
            if toss_decision == "bat":
                toss_decision_sign = toss_sign
            elif toss_decision in {"field", "bowl"}:
                toss_decision_sign = -toss_sign

            raw_matches.append({
                "match_date": _parse_match_date(info),
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner_code,
                "venue": {"ground": info.get("venue", ""), "city": info.get("city", "")},
                "toss_sign": toss_sign,
                "toss_decision_sign": toss_decision_sign,
                "team_totals": _extract_innings_totals(raw),
                "innings_sequence": _extract_match_sequence(raw),
                "lineups": _extract_lineups(raw),
                "player_match_stats": _extract_player_match_stats(raw),
            })

    raw_matches.sort(key=lambda item: item["match_date"])

    team_ratings: dict[str, float] = {}
    recent_form: dict[str, list[int]] = {}
    season_form: dict[str, dict[str, list[int]]] = {}
    season_stats: dict[str, dict[str, dict[str, float]]] = {}
    season_venue_stats: dict[str, dict[str, dict[str, float]]] = {}
    player_stats: dict[str, dict[str, float]] = {}
    last_seen_lineups: dict[str, list[str]] = {}
    head_to_head: dict[tuple[str, str], dict[str, int]] = {}
    venue_history: dict[str, dict[str, int]] = {}
    feature_rows: list[tuple[dict[str, float | str], float, float, float]] = []

    min_date = raw_matches[0]["match_date"] if raw_matches else date(2008, 1, 1)
    max_date = raw_matches[-1]["match_date"] if raw_matches else min_date
    total_days = max(1, (max_date - min_date).days)
    current_season = None

    for match in raw_matches:
        team_a = match["team_a"]
        team_b = match["team_b"]
        label = 1.0 if match["winner"] == team_a else 0.0
        season_key = str(match["match_date"].year)

        if current_season is None:
            current_season = season_key
        elif season_key != current_season:
            _regress_ratings_for_new_season(team_ratings)
            current_season = season_key

        season_bucket = season_form.setdefault(season_key, {})
        season_stats_bucket = season_stats.setdefault(season_key, {})
        season_venue_bucket = season_venue_stats.setdefault(season_key, {})
        current_lineups = match.get("lineups", {})
        team_lineup_strengths = {
            team_a: _compute_lineup_strength(current_lineups.get(team_a, []), player_stats),
            team_b: _compute_lineup_strength(current_lineups.get(team_b, []), player_stats),
        }
        rating_a = team_ratings.get(team_a, BASE_RATING)
        rating_b = team_ratings.get(team_b, BASE_RATING)
        elo_probability = _expected_from_ratings(rating_a, rating_b)

        features = build_feature_row(
            team_a,
            team_b,
            venue=match["venue"],
            season_key=season_key,
            toss_sign=match["toss_sign"],
            toss_decision_sign=match["toss_decision_sign"],
            team_ratings=team_ratings,
            recent_form=recent_form,
            season_form=season_bucket,
            season_stats=season_stats_bucket,
            season_venue_stats=season_venue_bucket,
            team_lineup_strengths=team_lineup_strengths,
            head_to_head=head_to_head,
            venue_history=venue_history,
        )

        recency = (match["match_date"] - min_date).days / total_days
        sample_weight = 0.7 + (0.8 * recency)
        feature_rows.append((features, label, sample_weight, elo_probability))

        k_factor = 26 if match["match_date"].year >= 2020 else 20
        new_a, new_b = _update_elo(rating_a, rating_b, label, k_factor)
        team_ratings[team_a] = new_a
        team_ratings[team_b] = new_b

        pair_key = _pair_key(team_a, team_b)
        pair_record = head_to_head.setdefault(pair_key, {})
        pair_record[match["winner"]] = pair_record.get(match["winner"], 0) + 1

        _append_recent_result(recent_form, team_a, label == 1.0)
        _append_recent_result(recent_form, team_b, label == 0.0)
        _append_recent_result(season_bucket, team_a, label == 1.0)
        _append_recent_result(season_bucket, team_b, label == 0.0)
        _update_venue_history(venue_history, team_a, match["venue"], label == 1.0)
        _update_venue_history(venue_history, team_b, match["venue"], label == 0.0)

        totals = match.get("team_totals", {})
        team_a_total = totals.get(team_a, {"runs": 0, "balls": 0})
        team_b_total = totals.get(team_b, {"runs": 0, "balls": 0})

        stats_a = season_stats_bucket.setdefault(team_a, _blank_season_stats())
        stats_b = season_stats_bucket.setdefault(team_b, _blank_season_stats())
        stats_a["matches"] += 1
        stats_b["matches"] += 1
        stats_a["wins"] += 1 if label == 1.0 else 0
        stats_b["wins"] += 1 if label == 0.0 else 0
        stats_a["points"] += 2 if label == 1.0 else 0
        stats_b["points"] += 2 if label == 0.0 else 0
        stats_a["runs_for"] += int(team_a_total.get("runs", 0))
        stats_a["balls_faced"] += int(team_a_total.get("balls", 0))
        stats_a["runs_against"] += int(team_b_total.get("runs", 0))
        stats_a["balls_bowled"] += int(team_b_total.get("balls", 0))
        stats_b["runs_for"] += int(team_b_total.get("runs", 0))
        stats_b["balls_faced"] += int(team_b_total.get("balls", 0))
        stats_b["runs_against"] += int(team_a_total.get("runs", 0))
        stats_b["balls_bowled"] += int(team_a_total.get("balls", 0))

        venue_key = _normalize_venue_key(match["venue"])
        if venue_key:
            venue_stats = season_venue_bucket.setdefault(venue_key, _blank_venue_season_stats())
            innings_sequence = match.get("innings_sequence", [])
            if innings_sequence:
                first_batting_team = innings_sequence[0]
                first_total = totals.get(first_batting_team, {"runs": 0})
                venue_stats["first_innings_runs"] += int(first_total.get("runs", 0))
                venue_stats["first_innings_count"] += 1
                if len(innings_sequence) >= 2 and match["winner"] == innings_sequence[1]:
                    venue_stats["chase_wins"] += 1
            venue_stats["matches"] += 1

        for team_code, lineup in current_lineups.items():
            if lineup:
                last_seen_lineups[team_code] = lineup

        for player_name, stats_update in (match.get("player_match_stats", {}) or {}).items():
            player_bucket = player_stats.setdefault(player_name, _blank_player_stats())
            for stat_key, stat_value in stats_update.items():
                player_bucket[stat_key] += stat_value

    split_index = max(1, int(len(feature_rows) * (1 - VALIDATION_SPLIT)))
    train_rows = feature_rows[:split_index]
    validation_rows = feature_rows[split_index:]

    model_type = "contextual_logistic_regression"
    weights: dict[str, float] = {}
    blend_alpha = 1.0

    if CatBoostClassifier is not None and pd is not None and feature_rows:
        all_df = pd.DataFrame([row for row, _, _, _ in feature_rows]).fillna(0)
        train_df = all_df.iloc[:split_index].copy()
        train_labels = [label for _, label, _, _ in train_rows]
        train_weights = [weight for _, _, weight, _ in train_rows]
        val_df = all_df.iloc[split_index:].copy()
        val_labels = [label for _, label, _, _ in validation_rows]
        val_weights = [weight for _, _, weight, _ in validation_rows]
        val_elo = [elo for _, _, _, elo in validation_rows]

        catboost_model = CatBoostClassifier(
            iterations=epochs,
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2,
            random_strength=random_strength,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False,
        )
        fit_kwargs = {
            "X": train_df,
            "y": train_labels,
            "cat_features": CATEGORICAL_FEATURES,
            "sample_weight": train_weights,
            "verbose": False,
        }
        if validation_rows:
            fit_kwargs["eval_set"] = (val_df, val_labels)
            fit_kwargs["use_best_model"] = True
        catboost_model.fit(**fit_kwargs)

        val_model_probs = catboost_model.predict_proba(val_df)[:, 1].tolist() if validation_rows else []
        blend_alpha = _select_blend_alpha_from_probs(val_labels, val_model_probs, val_elo, val_weights) if validation_rows else 1.0
        blended_val_probs = [
            (blend_alpha * model_prob) + ((1.0 - blend_alpha) * elo_prob)
            for model_prob, elo_prob in zip(val_model_probs, val_elo)
        ]
        validation_accuracy, validation_log_loss = _evaluate_probabilities(val_labels, blended_val_probs, val_weights)

        best_iterations = catboost_model.get_best_iteration()
        if best_iterations is None or best_iterations <= 0:
            best_iterations = catboost_model.tree_count_

        full_df = all_df.copy()
        full_labels = [label for _, label, _, _ in feature_rows]
        full_weights = [weight for _, _, weight, _ in feature_rows]
        full_elo = [elo for _, _, _, elo in feature_rows]

        final_model = CatBoostClassifier(
            iterations=best_iterations,
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2,
            random_strength=random_strength,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False,
        )
        final_model.fit(
            full_df,
            full_labels,
            cat_features=CATEGORICAL_FEATURES,
            sample_weight=full_weights,
            verbose=False,
        )
        final_model.save_model(str(CATBOOST_MODEL_PATH))
        full_model_probs = final_model.predict_proba(full_df)[:, 1].tolist()
        blended_train_probs = [
            (blend_alpha * model_prob) + ((1.0 - blend_alpha) * elo_prob)
            for model_prob, elo_prob in zip(full_model_probs, full_elo)
        ]
        training_accuracy, weighted_loss = _evaluate_probabilities(full_labels, blended_train_probs, full_weights)
        model_type = "catboost_classifier"
    else:
        validation_weights = _train_logistic(train_rows, epochs=epochs, lr=lr, l2=l2)
        blend_alpha = _select_blend_alpha(validation_rows, validation_weights)
        validation_accuracy, validation_log_loss = _evaluate_rows(validation_rows, validation_weights, blend_alpha=blend_alpha)

        weights = _train_logistic(feature_rows, epochs=epochs, lr=lr, l2=l2)
        training_accuracy, weighted_loss = _evaluate_rows(feature_rows, weights, blend_alpha=blend_alpha)

    latest_season = max(season_form.keys(), default="")
    team_lineup_strengths = {
        team_code: _compute_lineup_strength(lineup, player_stats)
        for team_code, lineup in last_seen_lineups.items()
    }
    player_strengths = _player_strengths_from_stats(player_stats)

    model = {
        "model_type": model_type,
        "trained_on_matches": len(feature_rows),
        "training_accuracy": training_accuracy,
        "weighted_log_loss": weighted_loss,
        "validation_accuracy": validation_accuracy,
        "validation_log_loss": validation_log_loss,
        "blend_alpha": round(blend_alpha, 2),
        "prediction_floor": 18,
        "prediction_ceiling": 82,
        "catboost_params": {
            "iterations": epochs,
            "learning_rate": lr,
            "l2_leaf_reg": l2,
            "depth": depth,
            "random_strength": random_strength,
        } if model_type == "catboost_classifier" else {},
        "weights": {key: round(value, 8) for key, value in weights.items()},
        "catboost_model_path": CATBOOST_MODEL_PATH.name if model_type == "catboost_classifier" else "",
        "cat_features": CATEGORICAL_FEATURES if model_type == "catboost_classifier" else [],
        "team_ratings": {key: round(value, 3) for key, value in team_ratings.items()},
        "recent_form": recent_form,
        "season_form": season_form.get(latest_season, {}),
        "season_stats": season_stats.get(latest_season, {}),
        "season_venue_stats": season_venue_stats.get(latest_season, {}),
        "team_lineup_strengths": team_lineup_strengths,
        "player_strengths": player_strengths,
        "head_to_head": {"|".join(key): value for key, value in head_to_head.items()},
        "venue_history": venue_history,
    }

    final_path = output_path or MODEL_PATH
    final_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model
