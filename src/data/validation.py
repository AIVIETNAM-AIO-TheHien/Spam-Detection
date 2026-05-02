import os


def validate_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}. "
            "Please place the dataset at the configured input_path."
        )


def validate_required_columns(df, required_columns) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_split_config(split_cfg: dict) -> None:
    train_size = split_cfg["train_size"]
    dev_size = split_cfg["dev_size"]
    test_size = split_cfg["test_size"]

    total = train_size + dev_size + test_size

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_size + dev_size + test_size must equal 1.0, got {total}"
        )

    if not split_cfg.get("shuffle", True) and split_cfg.get("stratify", True):
        raise ValueError("stratify=True requires shuffle=True.")