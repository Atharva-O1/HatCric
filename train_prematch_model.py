from pathlib import Path

from prematch_model import MODEL_PATH, train_from_cricsheet_zip


def main():
    repo_root = Path(__file__).resolve().parent
    zip_path = repo_root / "ipl_json.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset not found: {zip_path}")

    model = train_from_cricsheet_zip(zip_path, MODEL_PATH)
    print("Trained model saved to:", MODEL_PATH)
    print("Matches:", model["trained_on_matches"])
    print("Training accuracy:", model["training_accuracy"])
    print("Validation accuracy:", model.get("validation_accuracy"))
    print("Validation log loss:", model.get("validation_log_loss"))


if __name__ == "__main__":
    main()
