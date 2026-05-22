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
    ),
    ModelSpec(
        name="Logistic Regression",
        kind="logistic_regression",
        summary_path=ROOT_DIR / "experiments/logistic_regression/experiment_summary.csv",
        default_model_path=ROOT_DIR / "experiments/logistic_regression/model.joblib",
        default_vectorizer_path=ROOT_DIR / "experiments/logistic_regression/vectorizer.joblib",
        pipeline_path=ROOT_DIR / "models/logistic_regression_pipeline.joblib",
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
            bundle = ModelBundle(
                spec=spec,
                estimator=joblib.load(spec.pipeline_path),
                vectorizer=None,
                source_dir=spec.pipeline_path.parent,
                metrics=load_json(spec.pipeline_path.parent / "metrics.json"),
                best_threshold=None,
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


def predict_with_optional_threshold(bundle: ModelBundle, cleaned_text: str, use_threshold: bool) -> dict:
    features = transform_input(bundle, cleaned_text)
    estimator = bundle.estimator

    default_prediction = estimator.predict(features)[0]
    final_prediction = default_prediction
    score = None
    spam_probability = None
    probabilities = None
    classes = None

    probabilities, classes = positive_probabilities(estimator, features)
    if probabilities is not None and classes is not None and len(classes):
        class_list = list(classes)
        if POSITIVE_LABEL in class_list:
            spam_probability = float(probabilities[0][class_list.index(POSITIVE_LABEL)])

    scores = positive_scores(estimator, features)
    if scores is not None:
        score = float(scores[0])

    if use_threshold and bundle.best_threshold is not None:
        negative_label = next(
            (label for label in getattr(estimator, "classes_", []) if label != POSITIVE_LABEL),
            0.0,
        )

        if spam_probability is not None:
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
        "threshold": bundle.best_threshold if use_threshold else None,
        "source_dir": bundle.source_dir,
        "metrics": bundle.metrics,
    }


def metric_value(metrics: dict, key: str) -> Optional[float]:
    threshold_test = metrics.get("threshold_tuning", {}).get("test", {})
    test_metrics = threshold_test or metrics.get("test", {})
    value = test_metrics.get(key)
    return float(value) if value is not None else None


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
            value=True,
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
