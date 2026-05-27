import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.preprocess_fixed import clean_text


POSITIVE_LABEL = 1.0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    summary_path: Path
    default_model_path: Path
    default_vectorizer_path: Path
    pipeline_path: Optional[Path] = None
    pipeline_metrics_path: Optional[Path] = None


@dataclass
class ModelBundle:
    spec: ModelSpec
    estimator: Any
    vectorizer: Optional[Any]
    source_dir: Path
    metrics: dict
    best_threshold: Optional[float]


MODEL_SPECS = [
    ModelSpec(
        name="Linear SVC",
        kind="linear_svc",
        summary_path=ROOT_DIR / "experiments/linear_svc/experiment_summary.csv",
        default_model_path=ROOT_DIR / "experiments/linear_svc/model.joblib",
        default_vectorizer_path=ROOT_DIR / "experiments/linear_svc/vectorizer.joblib",
        pipeline_path=ROOT_DIR / "models/linear_svc_pipeline.joblib",
        pipeline_metrics_path=ROOT_DIR / "models/linear_svc_metrics.json",
    ),
    ModelSpec(
        name="Logistic Regression",
        kind="logistic_regression",
        summary_path=ROOT_DIR / "experiments/logistic_regression/experiment_summary.csv",
        default_model_path=ROOT_DIR / "experiments/logistic_regression/model.joblib",
        default_vectorizer_path=ROOT_DIR / "experiments/logistic_regression/vectorizer.joblib",
        pipeline_path=ROOT_DIR / "models/logistic_regression_pipeline.joblib",
        pipeline_metrics_path=ROOT_DIR / "models/logistic_regression_metrics.json",
    ),
]


LABEL_MAPPING = {
    0: "Ham / Không spam",
    0.0: "Ham / Không spam",
    "0": "Ham / Không spam",
    "0.0": "Ham / Không spam",
    "ham": "Ham / Không spam",
    "Ham": "Ham / Không spam",
    1: "Spam",
    1.0: "Spam",
    "1": "Spam",
    "1.0": "Spam",
    "spam": "Spam",
    "Spam": "Spam",
}


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def normalize_prediction(prediction: Any) -> str:
    if prediction in LABEL_MAPPING:
        return LABEL_MAPPING[prediction]

    prediction_str = str(prediction).strip()
    return LABEL_MAPPING.get(prediction_str, prediction_str)


def is_spam(prediction: Any) -> bool:
    return normalize_prediction(prediction).lower().startswith("spam")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def best_run_dir(summary_path: Path) -> Optional[Path]:
    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    if summary.empty or "run_dir" not in summary.columns:
        return None

    score_column = "test_spam_f1" if "test_spam_f1" in summary.columns else None
    if score_column is None:
        numeric_columns = summary.select_dtypes(include="number").columns
        score_column = numeric_columns[0] if len(numeric_columns) else None

    best_row = (
        summary.sort_values(score_column, ascending=False).iloc[0]
        if score_column
        else summary.iloc[0]
    )
    return ROOT_DIR / str(best_row["run_dir"])


def load_from_model_vectorizer(spec: ModelSpec, model_path: Path, vectorizer_path: Path) -> Optional[ModelBundle]:
    if not model_path.exists() or not vectorizer_path.exists():
        return None

    source_dir = model_path.parent
    metrics = load_json(source_dir / "metrics.json")
    threshold = (
        metrics.get("threshold_tuning", {}).get("best_threshold")
        if isinstance(metrics.get("threshold_tuning"), dict)
        else None
    )

    return ModelBundle(
        spec=spec,
        estimator=joblib.load(model_path),
        vectorizer=joblib.load(vectorizer_path),
        source_dir=source_dir,
        metrics=metrics,
        best_threshold=threshold,
    )


@st.cache_resource
def load_models() -> tuple[list[ModelBundle], list[str]]:
    loaded: list[ModelBundle] = []
    missing: list[str] = []

    for spec in MODEL_SPECS:
        bundle = None

        if spec.pipeline_path and spec.pipeline_path.exists():
            metrics = load_json(spec.pipeline_metrics_path) if spec.pipeline_metrics_path else {}
            threshold = (
                metrics.get("threshold_tuning", {}).get("best_threshold")
                if isinstance(metrics.get("threshold_tuning"), dict)
                else None
            )
            bundle = ModelBundle(
                spec=spec,
                estimator=joblib.load(spec.pipeline_path),
                vectorizer=None,
                source_dir=spec.pipeline_path.parent,
                metrics=metrics,
                best_threshold=threshold,
            )

        if bundle is None:
            bundle = load_from_model_vectorizer(
                spec,
                spec.default_model_path,
                spec.default_vectorizer_path,
            )

        if bundle is None:
            run_dir = best_run_dir(spec.summary_path)
            if run_dir is not None:
                bundle = load_from_model_vectorizer(
                    spec,
                    run_dir / "model.joblib",
                    run_dir / "vectorizer.joblib",
                )

        if bundle is None:
            missing.append(
                f"{spec.name}: chưa thấy model/vectorizer. "
                f"Hãy train hoặc chạy experiments để tạo artifact cho {spec.kind}."
            )
        else:
            loaded.append(bundle)

    return loaded, missing


def transform_input(bundle: ModelBundle, cleaned_text: str):
    if bundle.vectorizer is None:
        return [cleaned_text]

    return bundle.vectorizer.transform([cleaned_text])


def positive_scores(estimator: Any, features) -> Optional[np.ndarray]:
    if not hasattr(estimator, "decision_function"):
        return None

    scores = estimator.decision_function(features)
    scores = np.asarray(scores)

    classes = list(getattr(estimator, "classes_", []))
    if len(classes) == 2 and POSITIVE_LABEL in classes and classes.index(POSITIVE_LABEL) == 0:
        scores = -scores

    return scores


def positive_probabilities(estimator: Any, features) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not hasattr(estimator, "predict_proba"):
        return None, None

    probabilities = estimator.predict_proba(features)
    classes = np.asarray(getattr(estimator, "classes_", []))
    return probabilities, classes


def estimate_probabilities_from_score(
    estimator: Any,
    score: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    classes = list(getattr(estimator, "classes_", [])) or [0.0, POSITIVE_LABEL]
    clipped_score = float(np.clip(score, -60, 60))
    spam_probability = float(1 / (1 + np.exp(-clipped_score)))
    class_probabilities = [
        spam_probability if label == POSITIVE_LABEL else 1 - spam_probability
        for label in classes
    ]
    return np.asarray([class_probabilities]), np.asarray(classes), spam_probability


def predict_with_optional_threshold(bundle: ModelBundle, cleaned_text: str, use_threshold: bool) -> dict:
    features = transform_input(bundle, cleaned_text)
    estimator = bundle.estimator

    default_prediction = estimator.predict(features)[0]
    final_prediction = default_prediction
    score = None
    spam_probability = None
    probabilities = None
    classes = None
    probability_source = None

    probabilities, classes = positive_probabilities(estimator, features)
    if probabilities is not None and classes is not None and len(classes):
        probability_source = "model"
        class_list = list(classes)
        if POSITIVE_LABEL in class_list:
            spam_probability = float(probabilities[0][class_list.index(POSITIVE_LABEL)])

    scores = positive_scores(estimator, features)
    if scores is not None:
        score = float(scores[0])

    if probabilities is None and score is not None:
        probabilities, classes, spam_probability = estimate_probabilities_from_score(estimator, score)
        probability_source = "estimated from decision score"

    if use_threshold and bundle.best_threshold is not None:
        negative_label = next(
            (label for label in getattr(estimator, "classes_", []) if label != POSITIVE_LABEL),
            0.0,
        )

        if spam_probability is not None and probability_source == "model":
            final_prediction = POSITIVE_LABEL if spam_probability >= bundle.best_threshold else negative_label
        elif score is not None:
            final_prediction = POSITIVE_LABEL if score >= bundle.best_threshold else negative_label

    confidence = None
    if probabilities is not None:
        confidence = float(np.max(probabilities[0]))

    return {
        "model": bundle.spec.name,
        "label": normalize_prediction(final_prediction),
        "raw_prediction": final_prediction,
        "default_label": normalize_prediction(default_prediction),
        "confidence": confidence,
        "spam_probability": spam_probability,
        "score": score,
        "classes": classes,
        "probabilities": probabilities[0] if probabilities is not None else None,
        "probability_source": probability_source,
        "threshold": bundle.best_threshold if use_threshold else None,
        "source_dir": bundle.source_dir,
        "metrics": bundle.metrics,
    }


def metric_value(metrics: dict, key: str) -> Optional[float]:
    threshold_test = metrics.get("threshold_tuning", {}).get("test", {})
    test_metrics = threshold_test or metrics.get("test", {})
    value = test_metrics.get(key)
    return float(value) if value is not None else None


def percent_text(value: Optional[float]) -> str:
    return f"{value:.2%}" if value is not None else "N/A"


def number_text(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "Model": result["model"],
                "Prediction": result["label"],
                "Default prediction": result["default_label"],
                "Test accuracy": percent_text(metric_value(result["metrics"], "accuracy")),
                "Spam F1": percent_text(metric_value(result["metrics"], "spam_f1")),
                "Spam probability": percent_text(result["spam_probability"]),
                "Probability source": result["probability_source"] or "N/A",
                "Decision score": number_text(result["score"]),
                "Threshold": number_text(result["threshold"]),
                "Artifact": relative_path(result["source_dir"]),
            }
        )
    return pd.DataFrame(rows)


def show_input_email_section(show_cleaned_text: bool) -> tuple[str, bool]:
    with st.container(border=True):
        st.subheader("[1] Input Email")
        st.caption("Paste raw email content here. The app preprocesses it before sending it to the models.")

        text_input = st.text_area(
            "Email content",
            height=240,
            placeholder=(
                "Example: Congratulations! You have won a free prize. "
                "Click http://example.com now to claim your reward."
            ),
        )

        col_predict, col_hint = st.columns([1, 3])
        predict_button = col_predict.button("Classify email", type="primary", use_container_width=True)
        col_hint.caption("Input can include headers, URLs, phone numbers, HTML, or unprocessed text.")

        if show_cleaned_text:
            st.caption("Processed text will appear after classification.")

    return text_input, predict_button


def show_prediction_summary_section(results: list[dict], cleaned_text: str, show_cleaned_text: bool) -> None:
    with st.container(border=True):
        st.subheader("[2] Prediction Summary")

        spam_votes = sum(1 for result in results if is_spam(result["raw_prediction"]))
        total_votes = len(results)
        ham_votes = total_votes - spam_votes
        final_label = "Spam" if spam_votes >= ham_votes else "Ham / Không spam"
        agreement = len({result["label"] for result in results}) == 1

        col_verdict, col_vote, col_words, col_agreement = st.columns(4)
        col_verdict.metric("Final verdict", final_label)
        col_vote.metric("Model votes", f"{spam_votes} spam / {ham_votes} ham")
        col_words.metric("Processed words", len(cleaned_text.split()))
        col_agreement.metric("Agreement", "Yes" if agreement else "No")

        if final_label == "Spam":
            st.error("The email is classified as Spam by the model ensemble.")
        else:
            st.success("The email is classified as Ham / Không spam by the model ensemble.")

        if show_cleaned_text:
            with st.expander("Processed text", expanded=False):
                st.code(cleaned_text or "(empty)", language="text")


def show_model_details_section(results: list[dict]) -> None:
    with st.container(border=True):
        st.subheader("[3] Model Details / Comparison")
        st.dataframe(build_comparison_table(results), hide_index=True, use_container_width=True)

        for result in results:
            with st.expander(f"{result['model']} details", expanded=False):
                details = []
                if result["confidence"] is not None:
                    details.append(f"Confidence: {result['confidence']:.2%}")
                if result["spam_probability"] is not None:
                    details.append(f"Spam probability: {result['spam_probability']:.2%}")
                if result["score"] is not None:
                    details.append(f"Decision score: {result['score']:.4f}")
                if result["threshold"] is not None:
                    details.append(f"Threshold: {result['threshold']:.4f}")
                if result["probability_source"] is not None:
                    details.append(f"Probability source: {result['probability_source']}")

                if details:
                    st.caption(" | ".join(details))

                if result["probabilities"] is not None and result["classes"] is not None:
                    probability_table = pd.DataFrame(
                        {
                            "Class": [normalize_prediction(label) for label in result["classes"]],
                            "Probability": [f"{prob:.2%}" for prob in result["probabilities"]],
                        }
                    )
                    st.dataframe(probability_table, hide_index=True, use_container_width=True)


def show_prediction_card(result: dict) -> None:
    label = result["label"]
    status = "spam" if is_spam(result["raw_prediction"]) else "ham"

    with st.container(border=True):
        st.subheader(result["model"])
        if status == "spam":
            st.error(f"Dự đoán: {label}")
        else:
            st.success(f"Dự đoán: {label}")

        columns = st.columns(3)
        accuracy = metric_value(result["metrics"], "accuracy")
        spam_f1 = metric_value(result["metrics"], "spam_f1")

        columns[0].metric("Test accuracy", f"{accuracy:.2%}" if accuracy is not None else "N/A")
        columns[1].metric("Spam F1", f"{spam_f1:.2%}" if spam_f1 is not None else "N/A")
        columns[2].metric("Nguồn model", relative_path(result["source_dir"]))

        details = []
        if result["confidence"] is not None:
            details.append(f"Độ tin cậy: {result['confidence']:.2%}")
        if result["spam_probability"] is not None:
            details.append(f"Xác suất spam: {result['spam_probability']:.2%}")
        if result["score"] is not None:
            details.append(f"Decision score spam: {result['score']:.4f}")
        if result["threshold"] is not None:
            details.append(f"Threshold đang dùng: {result['threshold']:.4f}")

        if details:
            st.caption(" | ".join(details))

        if result["probabilities"] is not None and result["classes"] is not None:
            probability_table = pd.DataFrame(
                {
                    "Lớp": [normalize_prediction(label) for label in result["classes"]],
                    "Xác suất": [f"{prob:.2%}" for prob in result["probabilities"]],
                }
            )
            st.dataframe(probability_table, hide_index=True, use_container_width=True)


def main() -> None:
    os.chdir(ROOT_DIR)

    st.set_page_config(
        page_title="Spam Detection",
        layout="wide",
    )

    st.title("Spam Detection")
    st.caption("Nhập email thô, app sẽ tiền xử lý bằng preprocess_fixed.py rồi đưa qua Linear SVC và Logistic Regression.")

    bundles, missing_messages = load_models()

    with st.sidebar:
        st.header("Thiết lập")
        use_threshold = st.toggle(
            "Dùng threshold tốt nhất từ dev set",
            value=False,
            help="Áp dụng best_threshold đã lưu trong metrics của từng experiment, nếu có.",
        )
        show_cleaned_text = st.toggle("Hiển thị text sau tiền xử lý", value=True)

        st.divider()
        st.write("Model đã load")
        if bundles:
            for bundle in bundles:
                st.success(f"{bundle.spec.name}: {relative_path(bundle.source_dir)}")
        else:
            st.warning("Chưa load được model nào.")

        for message in missing_messages:
            st.info(message)

    if not bundles:
        st.error("Chưa có artifact model để dự đoán.")
        st.code(
            "python scripts/run_linear_svc_experiments.py\n"
            "python scripts/run_logistic_experiments.py",
            language="bash",
        )
        return

    st.caption("End-to-end email classification with preprocessing, prediction summary, and model comparison.")

    text_input, predict_button = show_input_email_section(show_cleaned_text)

    if not predict_button:
        st.info("Enter an email and click Classify email to see the prediction summary and model comparison.")
        return

    if not text_input.strip():
        st.warning("Please enter email content before classification.")
        return

    cleaned_text = clean_text(text_input)
    results = [
        predict_with_optional_threshold(bundle, cleaned_text, use_threshold)
        for bundle in bundles
    ]

    show_prediction_summary_section(results, cleaned_text, show_cleaned_text)
    show_model_details_section(results)
    return

    text_input = st.text_area(
        "Nội dung email",
        height=220,
        placeholder=(
            "Ví dụ: Congratulations! You have won a free prize. "
            "Click http://example.com now to claim your reward."
        ),
    )

    predict_button = st.button("Phân loại", type="primary", use_container_width=True)

    if not predict_button:
        return

    if not text_input.strip():
        st.warning("Vui lòng nhập nội dung email trước khi phân loại.")
        return

    cleaned_text = clean_text(text_input)

    if show_cleaned_text:
        with st.expander("Text sau tiền xử lý", expanded=True):
            st.code(cleaned_text or "(rỗng)", language="text")

    st.subheader("Kết quả phân loại")
    results = [
        predict_with_optional_threshold(bundle, cleaned_text, use_threshold)
        for bundle in bundles
    ]

    for result in results:
        show_prediction_card(result)


if __name__ == "__main__":
    main()
