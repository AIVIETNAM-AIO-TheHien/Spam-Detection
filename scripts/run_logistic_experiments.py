import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

sys.path.append(os.getcwd())

from src.data.preprocess_fixed import clean_text
from src.data.validation import validate_file_exists, validate_required_columns
from src.evaluation.metrics import compute_classification_metrics


BASE_CONFIG_PATH = Path("configs/logistic_regression.yaml")
RUNS_DIR = Path("experiments/logistic_regression/runs")
SUMMARY_PATH = Path("experiments/logistic_regression/experiment_summary.csv")


EXPERIMENTS = [
    {
        "run_id": "lr_c0_5_ng12_min2",
        "model": {"C": 0.5},
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_features": 50000},
    },
    {
        "run_id": "lr_c1_0_ng12_min2",
        "model": {"C": 1.0},
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_features": 50000},
    },
    {
        "run_id": "lr_c2_0_ng12_min2",
        "model": {"C": 2.0},
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "max_features": 50000},
    },
    {
        "run_id": "lr_c1_0_ng13_min2",
        "model": {"C": 1.0},
        "vectorizer": {"ngram_range": [1, 3], "min_df": 2, "max_features": 80000},
    },
    {
        "run_id": "lr_c1_0_ng12_min1",
        "model": {"C": 1.0},
        "vectorizer": {"ngram_range": [1, 2], "min_df": 1, "max_features": 80000},
    },
]


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


def apply_overrides(config, experiment):
    run_config = deepcopy(config)

    for section in ["model", "vectorizer"]:
        for key, value in experiment.get(section, {}).items():
            run_config[section][key] = value

    return run_config


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

        score = (
            metrics["spam_f1"],
            metrics["spam_recall"],
            metrics["spam_precision"],
        )
        best_score = None
        if best_metrics is not None:
            best_score = (
                best_metrics["spam_f1"],
                best_metrics["spam_recall"],
                best_metrics["spam_precision"],
            )

        if best_metrics is None or score > best_score:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def build_prediction_dataframe(
    df,
    text_column,
    label_column,
    predictions,
    probabilities,
    positive_label,
):
    result_df = df.copy()
    result_df["true_label"] = result_df[label_column]
    result_df["predicted_label"] = predictions
    result_df["spam_probability"] = probabilities
    result_df["error_type"] = "correct"
    result_df.loc[
        (result_df["true_label"] != positive_label)
        & (result_df["predicted_label"] == positive_label),
        "error_type",
    ] = "false_positive"
    result_df.loc[
        (result_df["true_label"] == positive_label)
        & (result_df["predicted_label"] != positive_label),
        "error_type",
    ] = "false_negative"
    result_df["message_length"] = result_df[text_column].fillna("").astype(str).str.len()
    result_df["word_count"] = (
        result_df[text_column].fillna("").astype(str).str.split().str.len()
    )
    result_df["has_url"] = (
        result_df[text_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(r"\burl\b|http|www", regex=True)
    )
    result_df["has_digit"] = (
        result_df[text_column].fillna("").astype(str).str.contains(r"\d", regex=True)
    )
    result_df["has_spam_keyword"] = (
        result_df[text_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            r"\burl\b|free|win|winner|cash|prize|urgent|claim|call|txt|stop|reply",
            regex=True,
        )
    )

    return result_df


def label_name(label, positive_label):
    return "spam" if label == positive_label else "ham"


def short_text(text, max_length=220):
    text = " ".join(str(text).replace("|", "/").split())
    if len(text) <= max_length:
        return text

    return f"{text[:max_length].rstrip()}..."


def format_bool(value):
    return "có" if bool(value) else "không"


def save_error_analysis(
    run_dir,
    result_df,
    y_true,
    y_pred,
    labels,
    metrics,
    text_column,
):
    output_dir = run_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    errors_df = result_df[result_df["error_type"] != "correct"]
    errors_df.to_csv(output_dir / "prediction_errors.csv", index=True)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            rows.append(
                {
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_index][predicted_index]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "confusion_matrix.csv", index=False)

    summary = {
        "total_samples": int(len(result_df)),
        "total_errors": int(len(errors_df)),
        "correct": int((result_df["error_type"] == "correct").sum()),
        "false_positive": int((result_df["error_type"] == "false_positive").sum()),
        "false_negative": int((result_df["error_type"] == "false_negative").sum()),
        "error_rate": float(len(errors_df) / len(result_df)),
    }

    group_summary = (
        errors_df.groupby("error_type")
        .agg(
            count=("error_type", "size"),
            avg_message_length=("message_length", "mean"),
            avg_word_count=("word_count", "mean"),
            url_count=("has_url", "sum"),
            digit_count=("has_digit", "sum"),
            spam_keyword_count=("has_spam_keyword", "sum"),
        )
        .reset_index()
    )
    summary["error_groups"] = group_summary.to_dict(orient="records")

    with open(output_dir / "error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    label_display = {label: label_name(label, labels[-1]) for label in labels}
    confusion_rows = []
    for row in rows:
        confusion_rows.append(
            f"| {label_display[row['actual_label']]} | "
            f"{label_display[row['predicted_label']]} | {row['count']} |"
        )

    false_positive_df = errors_df[errors_df["error_type"] == "false_positive"]
    false_negative_df = errors_df[errors_df["error_type"] == "false_negative"]

    report = [
        "# Phân tích lỗi Logistic Regression",
        "",
        "## 1. Tóm tắt kết quả",
        "",
        "| Metric | Giá trị |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        report.append(f"| {key} | {value:.4f} |")

    report.extend(
        [
            "",
            "## 2. Tổng quan lỗi",
            "",
            "| Chỉ số | Giá trị |",
            "|---|---:|",
            f"| Tổng số mẫu test | {summary['total_samples']} |",
            f"| Dự đoán đúng | {summary['correct']} |",
            f"| Tổng số lỗi | {summary['total_errors']} |",
            f"| Error rate | {summary['error_rate']:.2%} |",
            f"| False positive | {summary['false_positive']} |",
            f"| False negative | {summary['false_negative']} |",
            "",
            "False positive là mẫu ham bị dự đoán nhầm thành spam. "
            "False negative là mẫu spam bị bỏ sót và dự đoán thành ham.",
            "",
            "## 3. Confusion matrix",
            "",
            "| Nhãn thật | Dự đoán | Số lượng |",
            "|---|---|---:|",
        ]
    )
    report.extend(confusion_rows)

    report.extend(
        [
            "",
            "## 4. Đặc điểm các nhóm lỗi",
            "",
        ]
    )

    if summary["error_groups"]:
        report.extend(
            [
                "| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group in summary["error_groups"]:
            report.append(
                f"| {group['error_type']} | {group['count']} | "
                f"{group['avg_message_length']:.1f} | "
                f"{group['avg_word_count']:.1f} | "
                f"{int(group['url_count'])} | "
                f"{int(group['digit_count'])} | "
                f"{int(group['spam_keyword_count'])} |"
            )
    else:
        report.append("Không có lỗi trên test set.")

    report.extend(
        [
            "",
            "## 5. Ví dụ false positive",
            "",
            "| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |",
            "|---:|---:|---|---|---|---|---|---|",
        ]
    )

    if false_positive_df.empty:
        report.append("| - | - | - | - | - | - | - | Không có false positive |")
    for index, (_, row) in enumerate(false_positive_df.head(8).iterrows(), start=1):
        report.append(
            f"| {index} | {row['spam_probability']:.4f} | "
            f"{label_name(row['true_label'], labels[-1])} | "
            f"{label_name(row['predicted_label'], labels[-1])} | "
            f"{format_bool(row['has_url'])} | "
            f"{format_bool(row['has_digit'])} | "
            f"{format_bool(row['has_spam_keyword'])} | "
            f"{short_text(row[text_column])} |"
        )

    report.extend(
        [
            "",
            "Nhận xét nhanh: false positive thường là các email ham có hình thức giống spam, "
            "ví dụ newsletter, thông báo có nhiều link, giá tiền, khuyến mãi, hoặc MIME/html.",
            "",
            "## 6. Ví dụ false negative",
            "",
            "| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |",
            "|---:|---:|---|---|---|---|---|---|",
        ]
    )

    if false_negative_df.empty:
        report.append("| - | - | - | - | - | - | - | Không có false negative |")
    for index, (_, row) in enumerate(false_negative_df.head(8).iterrows(), start=1):
        report.append(
            f"| {index} | {row['spam_probability']:.4f} | "
            f"{label_name(row['true_label'], labels[-1])} | "
            f"{label_name(row['predicted_label'], labels[-1])} | "
            f"{format_bool(row['has_url'])} | "
            f"{format_bool(row['has_digit'])} | "
            f"{format_bool(row['has_spam_keyword'])} | "
            f"{short_text(row[text_column])} |"
        )

    report.extend(
        [
            "",
            "Nhận xét nhanh: false negative là nhóm cần ưu tiên đọc kỹ, vì đây là spam bị bỏ sót. "
            "Nếu các mẫu này thiếu keyword spam rõ ràng, cần xem xét thêm feature về URL, phone, "
            "money token, ngôn ngữ lạ, hoặc bổ sung data cùng pattern.",
            "",
            "## 7. File chi tiết",
            "",
            "- Các mẫu sai: `prediction_errors.csv`",
            "- Confusion matrix dạng CSV: `confusion_matrix.csv`",
            "- Tóm tắt lỗi dạng JSON: `error_summary.json`",
        ]
    )

    with open(output_dir / "error_analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return summary


def train_and_evaluate_run(base_config, experiment, df, split):
    run_id = experiment["run_id"]
    run_config = apply_overrides(base_config, experiment)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = run_config["data"]
    vectorizer_cfg = run_config["vectorizer"]
    model_cfg = run_config["model"]
    preprocess_cfg = run_config.get("preprocess", {})
    text_column = data_cfg["text_column"]
    label_column = data_cfg["label_column"]
    positive_label = data_cfg["positive_label"]

    train_df = df.loc[split["train_indices"]].copy()
    dev_df = df.loc[split["dev_indices"]].copy()
    test_df = df.loc[split["test_indices"]].copy()

    X_train = preprocess_text_series(train_df[text_column], preprocess_cfg)
    y_train = train_df[label_column]
    X_dev = preprocess_text_series(dev_df[text_column], preprocess_cfg)
    y_dev = dev_df[label_column]
    X_test = preprocess_text_series(test_df[text_column], preprocess_cfg)
    y_test = test_df[label_column]

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
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=model_cfg.get("C", 1.0),
        class_weight=model_cfg.get("class_weight", "balanced"),
        max_iter=model_cfg.get("max_iter", 2000),
        solver=model_cfg.get("solver", "liblinear"),
    )
    model.fit(X_train_vectorized, y_train)

    dev_default_predictions = model.predict(X_dev_vectorized)
    dev_default_metrics = compute_classification_metrics(
        y_true=y_dev,
        y_pred=dev_default_predictions,
        positive_label=positive_label,
    )

    test_default_predictions = model.predict(X_test_vectorized)
    test_default_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=test_default_predictions,
        positive_label=positive_label,
    )

    best_threshold, dev_threshold_metrics = find_best_threshold(
        model=model,
        features=X_dev_vectorized,
        labels=y_dev,
        positive_label=positive_label,
    )

    test_threshold_predictions, test_spam_probabilities = predict_with_threshold(
        model=model,
        features=X_test_vectorized,
        positive_label=positive_label,
        threshold=best_threshold,
    )
    test_threshold_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=test_threshold_predictions,
        positive_label=positive_label,
    )

    metrics = {
        "run_id": run_id,
        "model": model_cfg,
        "vectorizer": vectorizer_cfg,
        "preprocess": preprocess_cfg,
        "split_file": run_config["split"]["indices_path"],
        "dev": dev_default_metrics,
        "test": test_default_metrics,
        "threshold_tuning": {
            "selected_by": "best_dev_spam_f1",
            "best_threshold": best_threshold,
            "dev": dev_threshold_metrics,
            "test": test_threshold_metrics,
        },
    }

    joblib.dump(model, run_dir / "model.joblib")
    joblib.dump(vectorizer, run_dir / "vectorizer.joblib")

    with open(run_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    result_df = build_prediction_dataframe(
        df=test_df,
        text_column=text_column,
        label_column=label_column,
        predictions=test_threshold_predictions,
        probabilities=test_spam_probabilities,
        positive_label=positive_label,
    )
    error_summary = save_error_analysis(
        run_dir=run_dir,
        result_df=result_df,
        y_true=y_test,
        y_pred=test_threshold_predictions,
        labels=list(model.classes_),
        metrics=test_threshold_metrics,
        text_column=text_column,
    )

    return {
        "run_id": run_id,
        "C": model_cfg.get("C"),
        "class_weight": model_cfg.get("class_weight"),
        "ngram_range": str(vectorizer_cfg.get("ngram_range")),
        "min_df": vectorizer_cfg.get("min_df"),
        "max_features": vectorizer_cfg.get("max_features"),
        "best_threshold": best_threshold,
        "dev_spam_f1": dev_threshold_metrics["spam_f1"],
        "dev_spam_recall": dev_threshold_metrics["spam_recall"],
        "test_accuracy": test_threshold_metrics["accuracy"],
        "test_macro_f1": test_threshold_metrics["macro_f1"],
        "test_spam_precision": test_threshold_metrics["spam_precision"],
        "test_spam_recall": test_threshold_metrics["spam_recall"],
        "test_spam_f1": test_threshold_metrics["spam_f1"],
        "test_total_errors": error_summary["total_errors"],
        "test_false_positive": error_summary["false_positive"],
        "test_false_negative": error_summary["false_negative"],
        "run_dir": str(run_dir),
    }


def main():
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    data_cfg = base_config["data"]
    split_cfg = base_config["split"]

    validate_file_exists(data_cfg["input_path"])
    validate_file_exists(split_cfg["indices_path"])

    df = pd.read_csv(data_cfg["input_path"])
    validate_required_columns(df, [data_cfg["text_column"], data_cfg["label_column"]])

    with open(split_cfg["indices_path"], "r", encoding="utf-8") as f:
        split = json.load(f)

    rows = []
    for experiment in EXPERIMENTS:
        print(f"Running {experiment['run_id']}...")
        rows.append(train_and_evaluate_run(base_config, experiment, df, split))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        by=["test_spam_f1", "test_spam_recall", "test_macro_f1"],
        ascending=False,
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"Saved experiment summary to: {SUMMARY_PATH}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
