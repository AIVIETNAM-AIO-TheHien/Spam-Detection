from pathlib import Path
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "baseline_pipeline.joblib"


COMMON_TEXT_COLUMNS = [
    "text",
    "message",
    "sms",
    "content",
    "email",
    "body",
    "v2",
]

COMMON_LABEL_COLUMNS = [
    "label",
    "target",
    "class",
    "category",
    "spam",
    "v1",
]


def read_csv_file(data_path: Path) -> pd.DataFrame:
    """
    Đọc file CSV với một số encoding phổ biến.
    """

    encodings = ["utf-8", "utf-8-sig", "latin1"]

    for encoding in encodings:
        try:
            return pd.read_csv(data_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Không đọc được file CSV: {data_path}")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Tìm tên cột dựa trên danh sách tên thường gặp.
    """

    lower_to_original = {
        col.lower().strip(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    return None


def normalize_labels(y: pd.Series) -> pd.Series:
    """
    Chuẩn hóa nhãn về dạng 0/1 nếu dataset dùng ham/spam.
    0 = ham
    1 = spam
    """

    y = y.astype(str).str.strip().str.lower()

    label_mapping = {
        "ham": 0,
        "not spam": 0,
        "non-spam": 0,
        "legitimate": 0,
        "0": 0,

        "spam": 1,
        "1": 1,
    }

    y_mapped = y.map(label_mapping)

    if y_mapped.isna().any():
        unknown_labels = sorted(y[y_mapped.isna()].unique())
        raise ValueError(
            "Có nhãn chưa được hỗ trợ: "
            f"{unknown_labels}. Hãy kiểm tra lại cột label."
        )

    return y_mapped


def build_pipeline() -> Pipeline:
    """
    Baseline pipeline:
    Text -> TF-IDF -> Multinomial Naive Bayes
    """

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
            )
        ),
        (
            "model",
            MultinomialNB()
        )
    ])

    return pipeline


def train_baseline(data_path: Path, text_col: str | None, label_col: str | None):
    df = read_csv_file(data_path)

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    if text_col is None:
        text_col = find_column(df, COMMON_TEXT_COLUMNS)

    if label_col is None:
        label_col = find_column(df, COMMON_LABEL_COLUMNS)

    if text_col is None:
        raise ValueError(
            "Không tự tìm được cột text. "
            "Hãy truyền rõ bằng --text-col."
        )

    if label_col is None:
        raise ValueError(
            "Không tự tìm được cột label. "
            "Hãy truyền rõ bằng --label-col."
        )

    print("Text column:", text_col)
    print("Label column:", label_col)

    df = df[[text_col, label_col]].dropna()

    X = df[text_col].astype(str)
    y = normalize_labels(df[label_col])

    print("Label distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"]
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Đường dẫn đến file CSV chứa dataset."
    )

    parser.add_argument(
        "--text-col",
        default=None,
        help="Tên cột chứa nội dung văn bản."
    )

    parser.add_argument(
        "--label-col",
        default=None,
        help="Tên cột chứa nhãn spam/ham."
    )

    args = parser.parse_args()

    train_baseline(
        data_path=Path(args.data),
        text_col=args.text_col,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()