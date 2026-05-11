import json
import os
import sys

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

sys.path.append(os.getcwd())

from src.data.validation import validate_file_exists, validate_required_columns
from src.evaluation.metrics import compute_classification_metrics


def main():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    split_cfg = config["split"]
    vectorizer_cfg = config["vectorizer"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    validate_file_exists(data_cfg["input_path"])

    if not os.path.exists(split_cfg["indices_path"]):
        raise FileNotFoundError(
            f"Missing fixed split file: {split_cfg['indices_path']}. "
            "Run `python scripts/create_split.py` first."
        )

    df = pd.read_csv(data_cfg["input_path"])

    validate_required_columns(
        df,
        [data_cfg["text_column"], data_cfg["label_column"]],
    )

    with open(split_cfg["indices_path"], "r", encoding="utf-8") as f:
        split = json.load(f)

    train_df = df.loc[split["train_indices"]].copy()
    dev_df = df.loc[split["dev_indices"]].copy()

    text_column = data_cfg["text_column"]
    label_column = data_cfg["label_column"]

    X_train = train_df[text_column].fillna("").astype(str)
    y_train = train_df[label_column]

    X_dev = dev_df[text_column].fillna("").astype(str)
    y_dev = dev_df[label_column]

    vectorizer = TfidfVectorizer(
        max_features=vectorizer_cfg["max_features"],
        ngram_range=tuple(vectorizer_cfg["ngram_range"]),
        min_df=vectorizer_cfg["min_df"],
        max_df=vectorizer_cfg["max_df"],
        lowercase=vectorizer_cfg["lowercase"],
        strip_accents=vectorizer_cfg["strip_accents"],
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_dev_vectorized = vectorizer.transform(X_dev)

    model = MultinomialNB(alpha=model_cfg["alpha"])
    model.fit(X_train_vectorized, y_train)

    dev_predictions = model.predict(X_dev_vectorized)

    dev_metrics = compute_classification_metrics(
        y_true=y_dev,
        y_pred=dev_predictions,
        positive_label=data_cfg["positive_label"],
    )

    os.makedirs(output_cfg["experiment_dir"], exist_ok=True)

    joblib.dump(model, output_cfg["model_path"])
    joblib.dump(vectorizer, output_cfg["vectorizer_path"])

    metrics = {
        "model": model_cfg["type"],
        "feature_extractor": "tfidf",
        "split_file": split_cfg["indices_path"],
        "dev": dev_metrics,
    }

    with open(output_cfg["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
