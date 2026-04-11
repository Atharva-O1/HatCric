import json
import math
import zipfile
from datetime import date
from pathlib import Path

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    Pool = None

from prematch_model import normalize_team_code


LIVE_MODEL_META_PATH = Path(__file__).with_name("live_model.json")
LIVE_MODEL_PATH = Path(__file__).with_name("live_model.cbm")
VALIDATION_SPLIT = 0.15

FEATURE_ORDER = [
    "innings",
    "runs",
    "wickets",
    "balls_bowled",
    "balls_remaining",
    "wickets_in_hand",
    "current_rr",
    "target",
    "runs_needed",
    "required_rr",
    "run_rate_edge",
    "projected_total",
    "target_buffer",
    "batting_team_code",
    "bowling_team_code",
    "venue_city",
    "venue_ground",
    "season",
]

CATEGORICAL_FEATURES = [
    "batting_team_code",
    "bowling_team_code",
    "venue_city",
    "venue_ground",
    "season",
]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_date_sort_key(raw_match: dict) -> str:
    dates = (raw_match.get("info", {}) or {}).get("dates", []) or []
    if not dates:
        return "0000-00-00"
    return str(dates[0])


def _is_legal_delivery(delivery: dict) -> bool:
    extras = delivery.get("extras", {}) or {}
    return "wides" not in extras and "noballs" not in extras


def _delivery_wicket_count(delivery: dict) -> int:
    wickets = delivery.get("wickets", []) or []
    return len(wickets)


def _extract_innings_totals(innings_data: dict) -> tuple[int, int, int]:
    runs = 0
    wickets = 0
    legal_balls = 0
    for over in innings_data.get("overs", []) or []:
        for delivery in over.get("deliveries", []) or []:
            runs += int(((delivery.get("runs", {}) or {}).get("total", 0)) or 0)
            wickets += _delivery_wicket_count(delivery)
            if _is_legal_delivery(delivery):
                legal_balls += 1
    return runs, wickets, legal_balls


def _overs_from_balls(balls: int) -> float:
    whole = balls // 6
    part = balls % 6
    return float(f"{whole}.{part}")


def _projected_total(runs: int, legal_balls: int) -> float:
    if legal_balls <= 0:
        return 0.0
    current_rr = runs / (legal_balls / 6.0)
    return runs + ((120 - legal_balls) / 6.0) * current_rr


def _build_feature_row(
    *,
    innings_number: int,
    runs: int,
    wickets: int,
    legal_balls: int,
    target: int,
    batting_team_code: str,
    bowling_team_code: str,
    venue_city: str,
    venue_ground: str,
    season: str,
) -> dict[str, int | float | str]:
    balls_remaining = max(0, 120 - legal_balls)
    wickets_in_hand = max(0, 10 - wickets)
    current_rr = (runs / (legal_balls / 6.0)) if legal_balls > 0 else 0.0
    runs_needed = max(0, target - runs) if innings_number == 2 and target > 0 else 0
    required_rr = (runs_needed / (balls_remaining / 6.0)) if innings_number == 2 and balls_remaining > 0 else 0.0
    projected_total = _projected_total(runs, legal_balls) if innings_number == 1 else 0.0
    target_buffer = (projected_total - 175.0) if innings_number == 1 else float(target - runs)
    run_rate_edge = current_rr - required_rr if innings_number == 2 else current_rr - 8.75

    return {
        "innings": innings_number,
        "runs": runs,
        "wickets": wickets,
        "balls_bowled": legal_balls,
        "balls_remaining": balls_remaining,
        "wickets_in_hand": wickets_in_hand,
        "current_rr": round(current_rr, 4),
        "target": target if innings_number == 2 else 0,
        "runs_needed": runs_needed,
        "required_rr": round(required_rr, 4),
        "run_rate_edge": round(run_rate_edge, 4),
        "projected_total": round(projected_total, 2),
        "target_buffer": round(target_buffer, 2),
        "batting_team_code": batting_team_code,
        "bowling_team_code": bowling_team_code,
        "venue_city": venue_city,
        "venue_ground": venue_ground,
        "season": season,
    }


def _extract_live_training_rows(raw_match: dict) -> list[tuple[dict[str, int | float | str], int]]:
    info = raw_match.get("info", {}) or {}
    outcome = info.get("outcome", {}) or {}
    winner_name = outcome.get("winner")
    if not winner_name:
        return []

    innings_list = raw_match.get("innings", []) or []
    if len(innings_list) < 2:
        return []

    winner_code = normalize_team_code(winner_name)
    if not winner_code:
        return []

    venue_city = str(info.get("city", "") or "")
    venue_ground = str(info.get("venue", "") or "")
    season = str(info.get("season", "") or "")

    first_innings_total, _, _ = _extract_innings_totals(innings_list[0])
    target = first_innings_total + 1
    rows: list[tuple[dict[str, int | float | str], int]] = []

    for innings_index, innings_data in enumerate(innings_list[:2], start=1):
        batting_team_name = innings_data.get("team", "")
        batting_code = normalize_team_code(batting_team_name)
        if not batting_code:
            continue

        team_names = info.get("teams", []) or []
        bowling_team_name = next((team for team in team_names if team != batting_team_name), "")
        bowling_code = normalize_team_code(bowling_team_name)
        if not bowling_code:
            continue

        snapshot_target = target if innings_index == 2 else 0
        runs = 0
        wickets = 0
        legal_balls = 0

        for over in innings_data.get("overs", []) or []:
            for delivery in over.get("deliveries", []) or []:
                runs += int(((delivery.get("runs", {}) or {}).get("total", 0)) or 0)
                wickets += _delivery_wicket_count(delivery)
                legal = _is_legal_delivery(delivery)
                if legal:
                    legal_balls += 1

                if legal_balls < 6:
                    continue

                feature_row = _build_feature_row(
                    innings_number=innings_index,
                    runs=runs,
                    wickets=wickets,
                    legal_balls=legal_balls,
                    target=snapshot_target,
                    batting_team_code=batting_code,
                    bowling_team_code=bowling_code,
                    venue_city=venue_city,
                    venue_ground=venue_ground,
                    season=season,
                )
                label = 1 if batting_code == winner_code else 0
                rows.append((feature_row, label))

                if innings_index == 2 and runs >= snapshot_target:
                    break
            if innings_index == 2 and runs >= snapshot_target:
                break

    return rows


def _rows_to_matrix(rows: list[dict[str, int | float | str]]) -> list[list[int | float | str]]:
    return [[row.get(feature) for feature in FEATURE_ORDER] for row in rows]


def _log_loss(y_true: list[int], probs: list[float]) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for truth, prob in zip(y_true, probs):
        clipped = _clamp(prob, 1e-6, 1 - 1e-6)
        total += -(truth * math.log(clipped) + (1 - truth) * math.log(1 - clipped))
    return total / len(y_true)


def load_live_model_meta(model_path: Path | None = None) -> dict | None:
    path = model_path or LIVE_MODEL_META_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_LIVE_CATBOOST_CACHE: object | None | bool = None


def load_live_catboost_model(model_path: Path | None = None):
    global _LIVE_CATBOOST_CACHE
    path = model_path or LIVE_MODEL_PATH
    if CatBoostClassifier is None or not path.exists():
        return None
    if _LIVE_CATBOOST_CACHE is False:
        return None
    if _LIVE_CATBOOST_CACHE is not None:
        return _LIVE_CATBOOST_CACHE

    model = CatBoostClassifier()
    model.load_model(str(path))
    _LIVE_CATBOOST_CACHE = model
    return model


def predict_live_win_probability(
    *,
    team_a: str,
    team_b: str,
    batting_team: str,
    innings: int,
    runs: int,
    wickets: int,
    overs_bowled: float,
    target: int,
    venue: dict | None = None,
):
    meta = load_live_model_meta()
    model = load_live_catboost_model()
    if not meta or model is None or Pool is None:
        return None

    team_a_code = normalize_team_code(team_a) or str(team_a or "TEAM_A")
    team_b_code = normalize_team_code(team_b) or str(team_b or "TEAM_B")
    batting_code = normalize_team_code(batting_team) or (
        team_a_code if str(batting_team or "").strip().lower() == str(team_a or "").strip().lower() else team_b_code
    )
    bowling_code = team_b_code if batting_code == team_a_code else team_a_code

    try:
        overs_float = float(overs_bowled or 0)
    except (TypeError, ValueError):
        overs_float = 0.0
    whole_overs = int(overs_float)
    ball_part = int(round((overs_float - whole_overs) * 10))
    legal_balls = max(0, whole_overs * 6 + min(max(ball_part, 0), 5))

    row = _build_feature_row(
        innings_number=int(innings or 1),
        runs=int(runs or 0),
        wickets=int(wickets or 0),
        legal_balls=legal_balls,
        target=int(target or 0),
        batting_team_code=batting_code,
        bowling_team_code=bowling_code,
        venue_city=str((venue or {}).get("city", "") or ""),
        venue_ground=str((venue or {}).get("ground", "") or ""),
        season=str(date.today().year),
    )
    matrix = _rows_to_matrix([row])
    cat_indexes = [FEATURE_ORDER.index(name) for name in CATEGORICAL_FEATURES]
    probability = float(model.predict_proba(Pool(matrix, cat_features=cat_indexes))[:, 1][0])
    batting_win_pct = int(round(_clamp(probability * 100.0, 1, 99)))

    if batting_code == team_a_code:
        return {"team_a": batting_win_pct, "team_b": 100 - batting_win_pct}
    return {"team_a": 100 - batting_win_pct, "team_b": batting_win_pct}


def train_live_model_from_cricsheet_zip(
    zip_path: Path,
    meta_output_path: Path | None = None,
    model_output_path: Path | None = None,
) -> dict:
    if CatBoostClassifier is None or Pool is None:
        raise RuntimeError("catboost is required to train the live model")

    with zipfile.ZipFile(zip_path) as zf:
        raw_matches = []
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            raw = json.loads(zf.read(name).decode("utf-8"))
            info = raw.get("info", {}) or {}
            event = info.get("event", {}) or {}
            if event.get("name") != "Indian Premier League":
                continue
            raw_matches.append(raw)

    raw_matches.sort(key=_safe_date_sort_key)

    feature_rows: list[dict[str, int | float | str]] = []
    labels: list[int] = []
    for raw_match in raw_matches:
        for row, label in _extract_live_training_rows(raw_match):
            feature_rows.append(row)
            labels.append(label)

    if not feature_rows:
        raise RuntimeError("No live training rows were extracted from the dataset")

    split_index = max(1, int(len(feature_rows) * (1 - VALIDATION_SPLIT)))
    train_rows = feature_rows[:split_index]
    val_rows = feature_rows[split_index:]
    train_labels = labels[:split_index]
    val_labels = labels[split_index:]

    cat_indexes = [FEATURE_ORDER.index(name) for name in CATEGORICAL_FEATURES]
    train_pool = Pool(_rows_to_matrix(train_rows), label=train_labels, cat_features=cat_indexes)
    val_pool = Pool(_rows_to_matrix(val_rows), label=val_labels, cat_features=cat_indexes) if val_rows else None

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=False,
        random_seed=42,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=bool(val_rows))

    train_probs = model.predict_proba(train_pool)[:, 1]
    train_preds = [1 if prob >= 0.5 else 0 for prob in train_probs]
    train_accuracy = sum(int(pred == truth) for pred, truth in zip(train_preds, train_labels)) / len(train_labels)

    validation_accuracy = None
    validation_log_loss = None
    if val_rows:
        val_probs = model.predict_proba(val_pool)[:, 1]
        val_preds = [1 if prob >= 0.5 else 0 for prob in val_probs]
        validation_accuracy = sum(int(pred == truth) for pred, truth in zip(val_preds, val_labels)) / len(val_labels)
        validation_log_loss = _log_loss(val_labels, list(val_probs))

    meta = {
        "model_type": "catboost_live_classifier",
        "trained_on_matches": len(raw_matches),
        "trained_on_rows": len(feature_rows),
        "training_accuracy": round(train_accuracy, 4),
        "validation_accuracy": round(validation_accuracy, 4) if validation_accuracy is not None else None,
        "validation_log_loss": round(validation_log_loss, 4) if validation_log_loss is not None else None,
        "feature_order": FEATURE_ORDER,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    (meta_output_path or LIVE_MODEL_META_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    model.save_model(str(model_output_path or LIVE_MODEL_PATH))
    return meta
