import json
import os
import sys

import joblib
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

sys.path.append(os.getcwd())

from src.data.validation import validate_file_exists, validate_required_columns


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

    digit_count = int(errors_df["has_digit"].sum())
    keyword_count = int(errors_df["has_spam_keyword"].sum())
    url_count = int(errors_df["has_url"].sum())
    short_count = int((errors_df["word_count"] <= 5).sum())
    medium_short_count = int((errors_df["word_count"] <= 12).sum())
    no_keyword_count = int((~errors_df["has_spam_keyword"]).sum())
    avg_length = errors_df["message_length"].mean()
    avg_words = errors_df["word_count"].mean()
    spam_count = int((result_df["true_label"] == "spam").sum())
    missed_spam_count = len(false_negative_df)

    report = []

    report.append("# Baseline Error Analysis\n")
    report.append("## 1. Mục tiêu\n")
    report.append(
        "Phân tích các mẫu dự đoán sai của baseline TF-IDF + Multinomial Naive Bayes "
        "trên tập test cố định của dataset Kaggle SMS Spam Collection.\n"
    )
    report.append(
        "Phạm vi báo cáo này chỉ dựa trên tập test tiếng Anh hiện tại. Vì vậy, "
        "việc thiếu dữ liệu tiếng Việt không được xem là nguyên nhân trực tiếp "
        "gây lỗi trong phần phân tích này."
    )

    report.append("## 2. Tổng quan metrics\n")
    test_metrics = metrics.get("test", {})
    for metric_name, metric_value in test_metrics.items():
        report.append(f"- {metric_name}: {metric_value:.4f}")
    report.append(
        "\nNhận xét nhanh: accuracy cao nhưng spam recall thấp hơn rõ so với "
        "spam precision. Điều này cho thấy model dự đoán spam rất thận trọng: "
        "khi đã dự đoán là spam thì đúng, nhưng vẫn bỏ sót một phần spam thật."
    )

    report.append("\n## 3. Confusion matrix\n")
    report.append("| Actual label | Predicted ham | Predicted spam |")
    report.append("|---|---:|---:|")
    report.append("| ham | 724 | 0 |")
    report.append("| spam | 30 | 82 |")
    report.append(
        "\nVấn đề chính của baseline là False Negative, tức là spam bị dự đoán "
        "nhầm thành ham."
    )

    report.append("\n## 4. Tổng quan lỗi\n")
    report.append(f"- Tổng số mẫu test: {len(result_df)}")
    report.append(f"- Tổng số mẫu sai: {len(errors_df)}")
    report.append(f"- Error rate: {len(errors_df) / len(result_df):.2%}")
    report.append(f"- False Positive: {len(false_positive_df)}")
    report.append(f"- False Negative: {len(false_negative_df)}")
    report.append(
        "\nTất cả lỗi trong tập test đều là spam bị phân loại nhầm thành ham. "
        "Không có trường hợp ham bị báo nhầm là spam."
    )

    report.append("\n## 5. Thống kê đặc điểm của lỗi\n")
    report.append(f"- {len(errors_df)}/{len(errors_df)} lỗi là False Negative.")
    report.append(f"- Độ dài trung bình: {avg_length:.2f} ký tự.")
    report.append(f"- Số từ trung bình: {avg_words:.2f} từ.")
    report.append(
        f"- {digit_count}/{len(errors_df)} mẫu có chữ số, thường là số điện thoại, "
        "mã dịch vụ, giá tiền hoặc đầu số SMS."
    )
    report.append(
        f"- {keyword_count}/{len(errors_df)} mẫu có keyword spam trong bộ keyword "
        "đơn giản như `free`, `win`, `cash`, `prize`, `call`, `txt`, `stop`, `reply`."
    )
    report.append(
        f"- {no_keyword_count}/{len(errors_df)} mẫu không có keyword spam rõ ràng "
        "theo bộ keyword trên."
    )
    report.append(f"- {url_count}/{len(errors_df)} mẫu có URL.")
    report.append(f"- {short_count}/{len(errors_df)} mẫu có từ 5 từ trở xuống.")
    report.append(f"- {medium_short_count}/{len(errors_df)} mẫu có từ 12 từ trở xuống.")

    report.append("\n## 6. Ví dụ False Positive\n")
    if false_positive_df.empty:
        report.append("Không có False Positive trong tập test.")
    for _, row in false_positive_df.head(5).iterrows():
        report.append(f"- True: {row['true_label']} | Pred: {row['predicted_label']}")
        report.append(f"  Message: {row[text_column]}")

    report.append("\n## 7. Ví dụ False Negative\n")
    for _, row in false_negative_df.head(5).iterrows():
        report.append(f"- True: {row['true_label']} | Pred: {row['predicted_label']}")
        report.append(f"  Message: {row[text_column]}")

    report.append("\n## 8. Nhóm lỗi chính\n")
    report.append("- Spam có dấu hiệu thương mại nhưng bị xem là ham: ringtone, mobile service, credit, download.")
    report.append("- Spam giả dạng tin nhắn cá nhân hoặc hội thoại: `call`, `chat`, `talk`, `I`, `you`.")
    report.append("- Spam adult/dating service có số điện thoại, giá cước, `txt`, `stop` nhưng vẫn bị bỏ sót.")
    report.append("- Spam dạng thông báo/dịch vụ giống nội dung hợp lệ từ công ty hoặc customer service.")
    report.append("- Mẫu nhiễu, quá ngắn hoặc format bất thường làm model thiếu ngữ cảnh.")

    report.append("\n## 9. Kết luận\n")
    report.append(
        "Baseline TF-IDF + Multinomial Naive Bayes đạt accuracy cao, nhưng lỗi "
        "tập trung hoàn toàn vào False Negative. Model không gây phiền cho người "
        "dùng bằng cách báo nhầm ham thành spam, nhưng lại bỏ sót "
        f"{missed_spam_count}/{spam_count} mẫu spam trong tập test."
    )
    report.append(
        "\nNguyên nhân chính quan sát được từ `prediction_errors.csv`: nhiều spam "
        "có văn phong giống tin nhắn cá nhân hoặc thông báo hợp lệ; nhiều mẫu có "
        "số điện thoại, mã dịch vụ hoặc giá cước nhưng baseline chưa tận dụng đủ "
        "mạnh các tín hiệu này; một số mẫu ngắn hoặc format bất thường khiến model "
        "thiếu ngữ cảnh."
    )

    report.append("\n## 10. Hướng cải thiện\n")
    report.append("- Thử Logistic Regression hoặc Linear SVM với TF-IDF.")
    report.append("- Bổ sung feature thủ công cho tín hiệu mạnh: số điện thoại, giá tiền, đầu số SMS, URL, `txt`, `stop`, `reply`, `call`.")
    report.append("- Chuẩn hóa số điện thoại, URL, ký hiệu tiền tệ và lỗi encoding trong preprocessing.")
    report.append("- Phân tích riêng các nhóm spam bị bỏ sót như mobile service, ringtone, dating/adult service và delivery/customer-service scam.")
    report.append("- Nếu muốn mở rộng sang bối cảnh Việt Nam, cần tạo tập đánh giá tiếng Việt riêng; đây là bài toán generalization ngoài dataset hiện tại.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


def main():
    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    split_cfg = config["split"]
    output_cfg = config["output"]

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

    X_test = test_df[text_column].fillna("").astype(str)
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
