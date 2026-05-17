import json
import os
import sys

import joblib
import pandas as pd
import yaml

sys.path.append(os.getcwd())

from src.data.validation import validate_file_exists, validate_required_columns
from src.evaluation.metrics import compute_classification_metrics
from src.data.preprocess import clean_text

def preprocess_text_series(texts, preprocess_cfg):
    texts = texts.fillna("").astype(str)

    if not preprocess_cfg.get("enabled", False):
        return texts

    clean_kwargs = {
        key: value
        for key, value in preprocess_cfg.items()
        if key != "enabled"
    }

    return texts.apply(lambda text: clean_text(text, **clean_kwargs))


def main():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    split_cfg = config["split"]
    output_cfg = config["output"]
    preprocess_cfg = config.get("preprocess", {})

    validate_file_exists(data_cfg["input_path"])

    if not os.path.exists(split_cfg["indices_path"]):
        raise FileNotFoundError(
            f"Missing fixed split file: {split_cfg['indices_path']}."
        )

    if not os.path.exists(output_cfg["model_path"]):
        raise FileNotFoundError(
            f"Missing trained model: {output_cfg['model_path']}. "
            "Run `python scripts/train_baseline.py` first."
        )

    if not os.path.exists(output_cfg["vectorizer_path"]):
        raise FileNotFoundError(
            f"Missing vectorizer: {output_cfg['vectorizer_path']}."
        )

    df = pd.read_csv(data_cfg["input_path"])

    validate_required_columns(
        df,
        [data_cfg["text_column"], data_cfg["label_column"]],
    )

    with open(split_cfg["indices_path"], "r", encoding="utf-8") as f:
        split = json.load(f)

    model = joblib.load(output_cfg["model_path"])
    vectorizer = joblib.load(output_cfg["vectorizer_path"])

    test_df = df.loc[split["test_indices"]].copy()

    X_test = preprocess_text_series(test_df[data_cfg["text_column"]], preprocess_cfg)
    y_test = test_df[data_cfg["label_column"]]

    X_test_vectorized = vectorizer.transform(X_test)
    test_predictions = model.predict(X_test_vectorized)

    test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=test_predictions,
        positive_label=data_cfg["positive_label"],
    )

    if os.path.exists(output_cfg["metrics_path"]):
        with open(output_cfg["metrics_path"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics["test"] = test_metrics

    with open(output_cfg["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()