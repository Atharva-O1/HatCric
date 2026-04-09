import json
import math
import random
import re
import zipfile
from datetime import date
from pathlib import Path


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
BASE_RATING = 1500.0
FORM_WINDOW = 5


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


def _dot(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(key, 0.0) * value for key, value in features.items())


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
    head_to_head: dict[tuple[str, str], dict[str, int]] | None = None,
) -> dict[str, float]:
    ratings = team_ratings or {}
    forms = recent_form or {}
    pair_history = head_to_head or {}
    pair_key = _pair_key(team_a_code, team_b_code)
    pair_record = pair_history.get(pair_key, {})
    wins_a, wins_b = _pair_record_from_perspective(pair_record, team_a_code, team_b_code)

    rating_a = ratings.get(team_a_code, BASE_RATING)
    rating_b = ratings.get(team_b_code, BASE_RATING)
    form_a = _recent_form_score(forms.get(team_a_code, []))
    form_b = _recent_form_score(forms.get(team_b_code, []))

    features = {
        "bias": 1.0,
        f"team:{team_a_code}": 1.0,
        f"team:{team_b_code}": -1.0,
        "home_advantage": float(infer_home_advantage(team_a_code, team_b_code, venue)),
        "toss_sign": float(toss_sign),
        "toss_decision_sign": float(toss_decision_sign),
        "rating_diff": _normalize_rating(rating_a - rating_b + BASE_RATING) - _normalize_rating(BASE_RATING),
        "rating_ratio": (rating_a / rating_b) - 1.0 if rating_b else 0.0,
        "form_diff": form_a - form_b,
        "head_to_head_diff": _head_to_head_score(wins_a, wins_b),
        "experience_diff": float(len(forms.get(team_a_code, [])) - len(forms.get(team_b_code, []))) / FORM_WINDOW,
    }
    return features


def load_model(model_path: Path | None = None) -> dict | None:
    path = model_path or MODEL_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict_match_probability(
    team_a_code: str,
    team_b_code: str,
    venue: dict | None = None,
    status: str = "",
    team_a_full: str = "",
    team_b_full: str = "",
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

    toss_sign, toss_decision_sign = infer_toss_context(status, team_a_code, team_b_code, team_a_full, team_b_full)
    features = build_contextual_features(
        team_a_code,
        team_b_code,
        venue,
        toss_sign,
        toss_decision_sign,
        team_ratings=team_ratings,
        recent_form=recent_form,
        head_to_head=head_to_head,
    )

    probability = _sigmoid(_dot(loaded.get("weights", {}), features))
    low = loaded.get("prediction_floor", 18)
    high = loaded.get("prediction_ceiling", 82)
    team_a_pct = round(max(low, min(high, probability * 100)))
    return {"team_a": team_a_pct, "team_b": 100 - team_a_pct}


def _parse_match_date(raw_info: dict) -> date:
    dates = raw_info.get("dates") or []
    if dates:
        return date.fromisoformat(str(dates[0]))
    return date(2008, 1, 1)


def _expected_from_ratings(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_elo(rating_a: float, rating_b: float, label: float, k_factor: float) -> tuple[float, float]:
    expected_a = _expected_from_ratings(rating_a, rating_b)
    delta = k_factor * (label - expected_a)
    return rating_a + delta, rating_b - delta


def _append_recent_result(history: dict[str, list[int]], team_code: str, won: bool) -> None:
    bucket = history.setdefault(team_code, [])
    bucket.append(1 if won else 0)
    if len(bucket) > FORM_WINDOW:
        del bucket[0]


def train_from_cricsheet_zip(
    zip_path: Path,
    output_path: Path | None = None,
    epochs: int = 700,
    lr: float = 0.02,
    l2: float = 0.0008,
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
            })

    raw_matches.sort(key=lambda item: item["match_date"])

    team_ratings: dict[str, float] = {}
    recent_form: dict[str, list[int]] = {}
    head_to_head: dict[tuple[str, str], dict[str, int]] = {}
    feature_rows: list[tuple[dict[str, float], float, float]] = []

    min_date = raw_matches[0]["match_date"] if raw_matches else date(2008, 1, 1)
    max_date = raw_matches[-1]["match_date"] if raw_matches else min_date
    total_days = max(1, (max_date - min_date).days)

    for match in raw_matches:
        team_a = match["team_a"]
        team_b = match["team_b"]
        label = 1.0 if match["winner"] == team_a else 0.0

        features = build_contextual_features(
            team_a,
            team_b,
            venue=match["venue"],
            toss_sign=match["toss_sign"],
            toss_decision_sign=match["toss_decision_sign"],
            team_ratings=team_ratings,
            recent_form=recent_form,
            head_to_head=head_to_head,
        )

        recency = (match["match_date"] - min_date).days / total_days
        sample_weight = 0.7 + (0.8 * recency)
        feature_rows.append((features, label, sample_weight))

        rating_a = team_ratings.get(team_a, BASE_RATING)
        rating_b = team_ratings.get(team_b, BASE_RATING)
        k_factor = 26 if match["match_date"].year >= 2020 else 20
        new_a, new_b = _update_elo(rating_a, rating_b, label, k_factor)
        team_ratings[team_a] = new_a
        team_ratings[team_b] = new_b

        pair_key = _pair_key(team_a, team_b)
        pair_record = head_to_head.setdefault(pair_key, {})
        pair_record[match["winner"]] = pair_record.get(match["winner"], 0) + 1

        _append_recent_result(recent_form, team_a, label == 1.0)
        _append_recent_result(recent_form, team_b, label == 0.0)

    rng = random.Random(42)
    weights: dict[str, float] = {}
    for features, _, _ in feature_rows:
        for feature in features:
            weights.setdefault(feature, 0.0)

    for _ in range(epochs):
        rng.shuffle(feature_rows)
        for features, label, sample_weight in feature_rows:
            prediction = _sigmoid(_dot(weights, features))
            error = (prediction - label) * sample_weight
            for feature, value in features.items():
                weights[feature] -= lr * (error * value + l2 * weights.get(feature, 0.0))

    correct = 0
    weighted_loss = 0.0
    for features, label, sample_weight in feature_rows:
        prediction = _sigmoid(_dot(weights, features))
        predicted_label = 1.0 if prediction >= 0.5 else 0.0
        if predicted_label == label:
            correct += 1
        prediction = min(max(prediction, 1e-7), 1 - 1e-7)
        weighted_loss += sample_weight * (-(label * math.log(prediction) + (1 - label) * math.log(1 - prediction)))

    model = {
        "model_type": "contextual_logistic_regression",
        "trained_on_matches": len(feature_rows),
        "training_accuracy": round(correct / len(feature_rows), 4) if feature_rows else 0.0,
        "weighted_log_loss": round(weighted_loss / len(feature_rows), 4) if feature_rows else 0.0,
        "prediction_floor": 18,
        "prediction_ceiling": 82,
        "weights": {key: round(value, 8) for key, value in weights.items()},
        "team_ratings": {key: round(value, 3) for key, value in team_ratings.items()},
        "recent_form": recent_form,
        "head_to_head": {"|".join(key): value for key, value in head_to_head.items()},
    }

    final_path = output_path or MODEL_PATH
    final_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model
