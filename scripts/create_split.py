import json
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

from src.data.validation import (
    validate_file_exists,
    validate_required_columns,
    validate_split_config,
)


def main():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    split_cfg = config["split"]

    input_path = data_cfg["input_path"]
    label_column = data_cfg["label_column"]
    indices_path = split_cfg["indices_path"]

    validate_split_config(split_cfg)
    validate_file_exists(input_path)

    if os.path.exists(indices_path):
        raise FileExistsError(
            f"Split file already exists: {indices_path}. "
            "This project uses a fixed split. Do not regenerate it."
        )

    df = pd.read_csv(input_path)

    validate_required_columns(df, [data_cfg["text_column"], label_column])

    all_indices = df.index.to_list()

    stratify_labels = None
    if split_cfg.get("stratify", True):
        stratify_labels = df[label_column]

    train_indices, temp_indices = train_test_split(
        all_indices,
        train_size=split_cfg["train_size"],
        random_state=split_cfg["random_seed"],
        stratify=stratify_labels,
        shuffle=split_cfg.get("shuffle", True),
    )

    temp_labels = None
    if split_cfg.get("stratify", True):
        temp_labels = df.loc[temp_indices, label_column]

    dev_ratio = split_cfg["dev_size"] / (
        split_cfg["dev_size"] + split_cfg["test_size"]
    )

    dev_indices, test_indices = train_test_split(
        temp_indices,
        train_size=dev_ratio,
        random_state=split_cfg["random_seed"],
        stratify=temp_labels,
        shuffle=split_cfg.get("shuffle", True),
    )

    split_data = {
        "random_seed": split_cfg["random_seed"],
        "shuffle": split_cfg.get("shuffle", True),
        "stratify": split_cfg.get("stratify", True),
        "train_size": split_cfg["train_size"],
        "dev_size": split_cfg["dev_size"],
        "test_size": split_cfg["test_size"],
        "train_indices": [int(i) for i in train_indices],
        "dev_indices": [int(i) for i in dev_indices],
        "test_indices": [int(i) for i in test_indices],
    }

    os.makedirs(os.path.dirname(indices_path), exist_ok=True)

    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    print(f"Saved fixed split to: {indices_path}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Dev samples: {len(dev_indices)}")
    print(f"Test samples: {len(test_indices)}")


if __name__ == "__main__":
    main()