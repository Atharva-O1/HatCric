from pathlib import Path

from live_model import LIVE_MODEL_META_PATH, train_live_model_from_cricsheet_zip


def main():
    repo_root = Path(__file__).resolve().parent
    zip_path = repo_root / "ipl_json.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset not found: {zip_path}")

    model = train_live_model_from_cricsheet_zip(zip_path, LIVE_MODEL_META_PATH)
    print("Trained live model saved to:", LIVE_MODEL_META_PATH)
    print("Matches:", model["trained_on_matches"])
    print("Rows:", model["trained_on_rows"])
    print("Training accuracy:", model["training_accuracy"])
    print("Validation accuracy:", model.get("validation_accuracy"))
    print("Validation log loss:", model.get("validation_log_loss"))


if __name__ == "__main__":
    main()
