import json
import os
import sys

import joblib
import pandas as pd
import yaml

sys.path.append(os.getcwd())

from src.data.preprocess_fixed import clean_text
from src.data.validation import validate_file_exists, validate_required_columns
from src.evaluation.metrics import compute_classification_metrics


# Áp dụng clean_text lên toàn bộ Series văn bản theo config
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


# Dự đoán dùng ngưỡng xác suất tùy chỉnh thay vì mặc định 0.5
def predict_with_threshold(model, features, positive_label, threshold):
    probabilities = model.predict_proba(features)
    positive_class_index = list(model.classes_).index(positive_label)
    positive_probabilities = probabilities[:, positive_class_index]

    negative_label = next(label for label in model.classes_ if label != positive_label)

    predictions = [
        positive_label if probability >= threshold else negative_label
        for probability in positive_probabilities
    ]

    return predictions, positive_probabilities


# Quét threshold 0.10–0.90 trên dev, chọn ngưỡng cho spam_f1 cao nhất
def find_best_threshold(model, features, labels, positive_label):
    best_threshold = 0.5
    best_metrics = None

    for i in range(10, 91):
        threshold = i / 100
        predictions, _ = predict_with_threshold(
            model=model,
            features=features,
            positive_label=positive_label,
            threshold=threshold,
        )
        metrics = compute_classification_metrics(
            y_true=labels,
            y_pred=predictions,
            positive_label=positive_label,
        )

        if best_metrics is None or (
            metrics["spam_f1"],
            metrics["spam_recall"],
            metrics["spam_precision"],
        ) > (
            best_metrics["spam_f1"],
            best_metrics["spam_recall"],
            best_metrics["spam_precision"],
        ):
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


# Load model đã train, evaluate test set, append kết quả vào metrics.json
def main():
    with open("configs/logistic_regression.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    split_cfg = config["split"]
    output_cfg = config["output"]
    preprocess_cfg = config.get("preprocess", {})
    positive_label = data_cfg["positive_label"]

    validate_file_exists(data_cfg["input_path"])

    if not os.path.exists(split_cfg["indices_path"]):
        raise FileNotFoundError(
            f"Missing fixed split file: {split_cfg['indices_path']}."
        )

    if not os.path.exists(output_cfg["model_path"]):
        raise FileNotFoundError(
            f"Missing trained model: {output_cfg['model_path']}. "
            "Run `python scripts/train_logistic_regression.py` first."
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

    dev_df = df.loc[split["dev_indices"]].copy()
    test_df = df.loc[split["test_indices"]].copy()

    X_dev = preprocess_text_series(dev_df[data_cfg["text_column"]], preprocess_cfg)
    y_dev = dev_df[data_cfg["label_column"]]
    X_dev_vectorized = vectorizer.transform(X_dev)

    X_test = preprocess_text_series(test_df[data_cfg["text_column"]], preprocess_cfg)
    y_test = test_df[data_cfg["label_column"]]
    X_test_vectorized = vectorizer.transform(X_test)

    default_predictions = model.predict(X_test_vectorized)
    default_test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=default_predictions,
        positive_label=positive_label,
    )

    best_threshold, dev_threshold_metrics = find_best_threshold(
        model=model,
        features=X_dev_vectorized,
        labels=y_dev,
        positive_label=positive_label,
    )

    threshold_predictions, _ = predict_with_threshold(
        model=model,
        features=X_test_vectorized,
        positive_label=positive_label,
        threshold=best_threshold,
    )
    threshold_test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=threshold_predictions,
        positive_label=positive_label,
    )

    if os.path.exists(output_cfg["metrics_path"]):
        with open(output_cfg["metrics_path"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics["test"] = default_test_metrics
    metrics["threshold_tuning"] = {
        "selected_by": "best_dev_spam_f1",
        "best_threshold": best_threshold,
        "dev": dev_threshold_metrics,
        "test": threshold_test_metrics,
    }

    with open(output_cfg["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
