import json
import os
import sys

import joblib
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

sys.path.append(os.getcwd())

from src.data.validation import validate_file_exists, validate_required_columns
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


def get_error_type(true_label, predicted_label, positive_label):
    if true_label == predicted_label:
        return "correct"

    if predicted_label == positive_label:
        return "false_positive"

    return "false_negative"


def has_url(text):
    text = text.lower()
    return "http" in text or "www" in text


def has_digit(text):
    return any(char.isdigit() for char in text)


def has_spam_keyword(text):
    spam_keywords = [
        "free",
        "win",
        "winner",
        "cash",
        "prize",
        "urgent",
        "claim",
        "call",
        "txt",
        "stop",
        "reply",
    ]

    text = text.lower()
    return any(keyword in text for keyword in spam_keywords)


def build_prediction_dataframe(test_df, text_column, label_column, predictions, positive_label):
    result_df = test_df.copy()

    result_df["true_label"] = result_df[label_column]
    result_df["predicted_label"] = predictions

    result_df["error_type"] = result_df.apply(
        lambda row: get_error_type(
            row["true_label"],
            row["predicted_label"],
            positive_label,
        ),
        axis=1,
    )

    result_df["message_length"] = result_df[text_column].fillna("").astype(str).str.len()
    result_df["word_count"] = (
        result_df[text_column].fillna("").astype(str).str.split().str.len()
    )
    result_df["has_url"] = result_df[text_column].fillna("").astype(str).apply(has_url)
    result_df["has_digit"] = result_df[text_column].fillna("").astype(str).apply(has_digit)
    result_df["has_spam_keyword"] = (
        result_df[text_column].fillna("").astype(str).apply(has_spam_keyword)
    )

    return result_df


def save_confusion_matrix(y_true, y_pred, labels, output_path):
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

    cm_df = pd.DataFrame(rows)
    cm_df.to_csv(output_path, index=False)


def save_error_summary(result_df, output_path):
    errors_df = result_df[result_df["error_type"] != "correct"]

    summary = {
        "total_samples": int(len(result_df)),
        "total_errors": int(len(errors_df)),
        "correct": int((result_df["error_type"] == "correct").sum()),
        "false_positive": int((result_df["error_type"] == "false_positive").sum()),
        "false_negative": int((result_df["error_type"] == "false_negative").sum()),
        "error_rate": float(len(errors_df) / len(result_df)),
    }

    error_group_summary = (
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

    summary["error_groups"] = error_group_summary.to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_markdown_report(result_df, metrics, output_path, text_column):
    errors_df = result_df[result_df["error_type"] != "correct"]
    false_positive_df = result_df[result_df["error_type"] == "false_positive"]
    false_negative_df = result_df[result_df["error_type"] == "false_negative"]

    ham_ham = int(
        ((result_df["true_label"] == "ham") & (result_df["predicted_label"] == "ham")).sum()
    )
    ham_spam = int(
        ((result_df["true_label"] == "ham") & (result_df["predicted_label"] == "spam")).sum()
    )
    spam_ham = int(
        ((result_df["true_label"] == "spam") & (result_df["predicted_label"] == "ham")).sum()
    )
    spam_spam = int(
        ((result_df["true_label"] == "spam") & (result_df["predicted_label"] == "spam")).sum()
    )

    total_errors = len(errors_df)
    avg_length = errors_df["message_length"].mean() if total_errors else 0.0
    avg_words = errors_df["word_count"].mean() if total_errors else 0.0
    digit_count = int(errors_df["has_digit"].sum()) if total_errors else 0
    keyword_count = int(errors_df["has_spam_keyword"].sum()) if total_errors else 0
    url_count = int(errors_df["has_url"].sum()) if total_errors else 0
    short_count = int((errors_df["word_count"] <= 5).sum()) if total_errors else 0
    medium_short_count = int((errors_df["word_count"] <= 12).sum()) if total_errors else 0
    no_keyword_count = int((~errors_df["has_spam_keyword"]).sum()) if total_errors else 0
    spam_count = int((result_df["true_label"] == "spam").sum())
    missed_spam_count = len(false_negative_df)

    report = [
        "# Baseline Error Analysis\n",
        "## 1. Mục tiêu\n",
        (
            "Phân tích các mẫu dự đoán sai của baseline TF-IDF + Multinomial Naive Bayes "
            "trên tập test cố định của SMS Spam Collection."
        ),
        (
            "\nBáo cáo này dùng cùng cấu hình preprocessing với bước evaluate baseline. "
            "Tin nhắn gốc và cột `preprocessed_text` đều được lưu trong các file CSV đầu ra."
        ),
        "\n## 2. Metrics\n",
    ]

    test_metrics = metrics.get("test", {})
    if test_metrics:
        for metric_name, metric_value in test_metrics.items():
            report.append(f"- {metric_name}: {metric_value:.4f}")
    else:
        report.append("- Không tìm thấy test metrics trong file metrics.")

    report.extend(
        [
            "\n## 3. Confusion Matrix\n",
            "| Nhãn thật | Dự đoán ham | Dự đoán spam |",
            "|---|---:|---:|",
            f"| ham | {ham_ham} | {ham_spam} |",
            f"| spam | {spam_ham} | {spam_spam} |",
            "\n## 4. Tổng quan lỗi\n",
            f"- Tổng số mẫu test: {len(result_df)}",
            f"- Tổng số mẫu sai: {total_errors}",
            f"- Error rate: {total_errors / len(result_df):.2%}",
            f"- False Positive: {len(false_positive_df)}",
            f"- False Negative: {len(false_negative_df)}",
        ]
    )

    if false_positive_df.empty and not false_negative_df.empty:
        report.append("\nTất cả lỗi đều là False Negative: spam bị dự đoán nhầm thành ham.")
    elif false_negative_df.empty and not false_positive_df.empty:
        report.append("\nTất cả lỗi đều là False Positive: ham bị dự đoán nhầm thành spam.")
    elif errors_df.empty:
        report.append("\nKhông có lỗi dự đoán trong tập test.")
    else:
        report.append("\nLỗi trong tập test gồm cả False Positive và False Negative.")

    report.extend(
        [
            "\n## 5. Đặc điểm của lỗi\n",
            f"- Tỷ lệ False Negative trong lỗi: {len(false_negative_df)}/{total_errors}",
            f"- Độ dài trung bình: {avg_length:.2f} ký tự",
            f"- Số từ trung bình: {avg_words:.2f} từ",
            f"- Mẫu có chữ số: {digit_count}/{total_errors}",
            f"- Mẫu có keyword spam: {keyword_count}/{total_errors}",
            f"- Mẫu không có keyword spam: {no_keyword_count}/{total_errors}",
            f"- Mẫu có URL: {url_count}/{total_errors}",
            f"- Mẫu có <= 5 từ: {short_count}/{total_errors}",
            f"- Mẫu có <= 12 từ: {medium_short_count}/{total_errors}",
            "\n## 6. Ví dụ False Positive\n",
        ]
    )

    if false_positive_df.empty:
        report.append("Không có False Positive trong tập test.")
    for _, row in false_positive_df.head(5).iterrows():
        report.append(f"- True: {row['true_label']} | Pred: {row['predicted_label']}")
        report.append(f"  Tin nhắn gốc: {row[text_column]}")
        report.append(f"  Sau preprocess: {row.get('preprocessed_text', '')}")

    report.append("\n## 7. Ví dụ False Negative\n")
    if false_negative_df.empty:
        report.append("Không có False Negative trong tập test.")
    for _, row in false_negative_df.head(5).iterrows():
        report.append(f"- True: {row['true_label']} | Pred: {row['predicted_label']}")
        report.append(f"  Tin nhắn gốc: {row[text_column]}")
        report.append(f"  Sau preprocess: {row.get('preprocessed_text', '')}")

    report.extend(
        [
            "\n## 8. Nhận xét\n",
            f"- Model bỏ sót {missed_spam_count}/{spam_count} mẫu spam trong tập test.",
            (
                "- Nên so sánh tin nhắn gốc với `preprocessed_text` trong `prediction_errors.csv` "
                "để kiểm tra preprocessing có làm mất tín hiệu spam hữu ích hay không."
            ),
            (
                "- Các tín hiệu cần xem kỹ gồm dấu câu, ký hiệu tiền tệ, số điện thoại, URL, "
                "mã dịch vụ và các keyword như `free`, `claim`, `call`, `txt`, `stop`, `reply`."
            ),
            "\n## 9. Hướng tiếp theo\n",
            "- So sánh báo cáo này với error analysis của baseline dùng raw text.",
            "- Chạy ablation cho preprocessing, đặc biệt là `remove_punct: false`.",
            "- Thử Logistic Regression hoặc Linear SVM với cùng bộ đặc trưng TF-IDF.",
            "- Nếu recall vẫn thấp, bổ sung feature thủ công cho URL, phone, money, digit và spam keyword.",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


def main():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    split_cfg = config["split"]
    output_cfg = config["output"]
    preprocess_cfg = config.get("preprocess", {})

    validate_file_exists(data_cfg["input_path"])
    validate_file_exists(split_cfg["indices_path"])
    validate_file_exists(output_cfg["model_path"])
    validate_file_exists(output_cfg["vectorizer_path"])

    df = pd.read_csv(data_cfg["input_path"])

    text_column = data_cfg["text_column"]
    label_column = data_cfg["label_column"]
    positive_label = data_cfg["positive_label"]

    validate_required_columns(df, [text_column, label_column])

    with open(split_cfg["indices_path"], "r", encoding="utf-8") as f:
        split = json.load(f)

    model = joblib.load(output_cfg["model_path"])
    vectorizer = joblib.load(output_cfg["vectorizer_path"])

    test_df = df.loc[split["test_indices"]].copy()

    X_test = preprocess_text_series(test_df[text_column], preprocess_cfg)
    test_df["preprocessed_text"] = X_test
    y_test = test_df[label_column]

    X_test_vectorized = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vectorized)

    result_df = build_prediction_dataframe(
        test_df=test_df,
        text_column=text_column,
        label_column=label_column,
        predictions=predictions,
        positive_label=positive_label,
    )

    output_dir = os.path.join(output_cfg["experiment_dir"], "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    all_predictions_path = os.path.join(output_dir, "all_predictions.csv")
    errors_path = os.path.join(output_dir, "prediction_errors.csv")
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.csv")
    summary_path = os.path.join(output_dir, "error_summary.json")
    report_path = os.path.join(output_dir, "error_analysis_report.md")

    result_df.to_csv(all_predictions_path, index=True)

    errors_df = result_df[result_df["error_type"] != "correct"]
    errors_df.to_csv(errors_path, index=True)

    labels = ["ham", "spam"]
    save_confusion_matrix(
        y_true=y_test,
        y_pred=predictions,
        labels=labels,
        output_path=confusion_matrix_path,
    )

    save_error_summary(result_df, summary_path)

    if os.path.exists(output_cfg["metrics_path"]):
        with open(output_cfg["metrics_path"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    save_markdown_report(result_df, metrics, report_path, text_column)

    print(f"Saved all predictions to: {all_predictions_path}")
    print(f"Saved prediction errors to: {errors_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")
    print(f"Saved error summary to: {summary_path}")
    print(f"Saved markdown report to: {report_path}")


if __name__ == "__main__":
    main()
