from pathlib import Path

import joblib
import numpy as np
import streamlit as st


# =========================
# Config
# =========================

MODEL_PATH = Path("models/baseline_pipeline.joblib")

LABEL_MAPPING = {
    0: "Ham / Không spam",
    1: "Spam",
    "0": "Ham / Không spam",
    "1": "Spam",
    "ham": "Ham / Không spam",
    "spam": "Spam",
    "Ham": "Ham / Không spam",
    "Spam": "Spam",
}


# =========================
# Load model
# =========================

@st.cache_resource
def load_model():
    """
    Load baseline model.

    Khuyến nghị model được lưu dưới dạng sklearn Pipeline:
        TfidfVectorizer / CountVectorizer + Naive Bayes

    Ví dụ:
        Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB())
        ])
    """

    if not MODEL_PATH.exists():
        st.error(f"Không tìm thấy model tại: {MODEL_PATH}")
        st.info("Hãy train baseline trước và lưu model vào thư mục models/.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    return model


# =========================
# Helper functions
# =========================

def normalize_prediction(prediction):
    """
    Chuyển output của model về nhãn dễ đọc.
    """

    if prediction in LABEL_MAPPING:
        return LABEL_MAPPING[prediction]

    prediction_str = str(prediction).strip()

    if prediction_str in LABEL_MAPPING:
        return LABEL_MAPPING[prediction_str]

    return prediction_str


def predict_single_text(model, text):
    """
    Dự đoán một văn bản đầu vào.
    """

    prediction = model.predict([text])[0]
    label = normalize_prediction(prediction)

    confidence = None
    probabilities = None
    classes = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(np.max(probabilities))

        if hasattr(model, "classes_"):
            classes = model.classes_

    return {
        "raw_prediction": prediction,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "classes": classes,
    }


def display_result(result):
    """
    Hiển thị kết quả dự đoán.
    """

    st.subheader("Kết quả dự đoán")

    label = result["label"]

    if "spam" in label.lower():
        st.error(f"Dự đoán: {label}")
    else:
        st.success(f"Dự đoán: {label}")

    if result["confidence"] is not None:
        st.write(f"Độ tin cậy: {result['confidence']:.2%}")

    if result["probabilities"] is not None and result["classes"] is not None:
        st.write("Xác suất theo từng lớp:")

        probability_table = {
            "Lớp": [normalize_prediction(cls) for cls in result["classes"]],
            "Xác suất": [f"{prob:.2%}" for prob in result["probabilities"]],
        }

        st.table(probability_table)


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(
        page_title="Spam Detection Baseline",
        layout="centered",
    )

    st.title("Spam Detection - Baseline Model")

    st.write(
        "Ứng dụng sử dụng baseline model để phân loại nội dung văn bản "
        "thành Spam hoặc Ham / Không spam."
    )

    model = load_model()

    st.divider()

    text_input = st.text_area(
        label="Nhập nội dung cần kiểm tra",
        height=180,
        placeholder="Ví dụ: Congratulations! You have won a free prize...",
    )

    predict_button = st.button("Dự đoán")

    if predict_button:
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung trước khi dự đoán.")
            return

        result = predict_single_text(model, text_input)
        display_result(result)

    st.divider()

    with st.expander("Thông tin model"):
        st.write("Đường dẫn model:")
        st.code(str(MODEL_PATH))

        st.write("Kiểu object được load:")
        st.code(type(model).__name__)


if __name__ == "__main__":
    main()