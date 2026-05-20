# Cấu trúc Dự án — Hiện trạng & Đề xuất

---

## 1. Cấu trúc Hiện tại

```
spam-detection/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── configs/
│   ├── baseline.yaml
│   └── augmentation_plan.json       ← file được sinh ra bởi script, không phải config tĩnh
│
├── docs/
│   ├── Plan.md
│   ├── QA_QC.md
│   └── feedback_preprocess.md
│
├── experiments/
│   ├── baseline/
│   │   ├── model.joblib
│   │   ├── vectorizer.joblib
│   │   ├── metrics.json
│   │   └── error_analysis/
│   │       ├── all_predictions.csv
│   │       ├── prediction_errors.csv
│   │       ├── confusion_matrix.csv
│   │       ├── error_summary.json
│   │       ├── error_analysis_report.md
│   │       └── preprocess_error_comparison_report.md
│   └── splits/
│       └── split_indices.json
│
├── notebooks/                        ← thư mục rỗng
│
├── scripts/
│   ├── create_split.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   └── analyze_errors.py
│
├── src/
│   ├── data/
│   │   ├── raw/
│   │   │   └── spam.csv              ← dữ liệu gốc nằm trong thư mục source code
│   │   ├── Assassin_cleaned.csv      ← file data nằm lẫn với code
│   │   ├── Assassin_preprocessed.csv ← file data nằm lẫn với code
│   │   ├── W1.zip                    ← file nén không rõ nguồn gốc/mục đích
│   │   ├── preprocess_fixed.py
│   │   └── validation.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/                       ← thư mục rỗng
│   ├── training/                     ← thư mục rỗng
│   └── utils/                        ← thư mục rỗng
│
├── tests/
│   ├── test_split_integrity.py
│   └── test_preprocess.py
│
├── augment_data.py                   ← script nằm ở root thay vì scripts/
├── datacollection_guide.md           ← tài liệu nằm ở root thay vì docs/
├── CLAUDE.md
├── README.md
└── requirements.txt
```

### Mô tả chức năng từng thành phần

| Đường dẫn | Loại | Chức năng |
| :--- | :---: | :--- |
| `.github/` | Config | GitHub Actions templates, PR/Issue templates |
| `configs/baseline.yaml` | Config | Toàn bộ hyperparameters: split, preprocess, TF-IDF, model, output paths |
| `configs/augmentation_plan.json` | Output | **File sinh ra** bởi `augment_data.py`, liệt kê domain cần bổ sung và kỹ thuật augment |
| `docs/Plan.md` | Docs | Kế hoạch 2 tuần, phân chia vai trò 4 nhóm |
| `docs/QA_QC.md` | Docs | Tiêu chí nghiệm thu P0/P1/P2 theo tuần |
| `docs/feedback_preprocess.md` | Docs | Phản hồi và quyết định thiết kế preprocessing |
| `datacollection_guide.md` | Docs | **Đặt sai vị trí** — hướng dẫn thu thập dữ liệu spam mới |
| `experiments/baseline/` | Output | Model, vectorizer, metrics của experiment baseline |
| `experiments/baseline/error_analysis/` | Output | Báo cáo phân tích lỗi, so sánh trước/sau preprocess |
| `experiments/splits/split_indices.json` | Output | Chỉ số phân tách train/dev/test cố định (seed=42) |
| `notebooks/` | — | **Rỗng** — chưa có notebook nào |
| `scripts/create_split.py` | Script | Tạo stratified split, từ chối ghi đè nếu đã tồn tại |
| `scripts/train_baseline.py` | Script | Huấn luyện TF-IDF + MultinomialNB, lưu model + dev metrics |
| `scripts/evaluate_baseline.py` | Script | Đánh giá trên test set, append kết quả vào metrics.json |
| `scripts/analyze_errors.py` | Script | Trích xuất FP/FN, phân tích pattern lỗi, xuất báo cáo |
| `augment_data.py` | Script | **Đặt sai vị trí** — sinh augmentation plan, lưu vào configs/ |
| `src/data/preprocess_fixed.py` | Library | Pipeline tiền xử lý: normalize entities, remove noise, clean_text, preprocess_pipeline |
| `src/data/validation.py` | Library | Validate file tồn tại, schema cột, config split |
| `src/data/raw/spam.csv` | Data | **Đặt sai vị trí** — dataset gốc UCI SMS Spam (5,574 dòng) |
| `src/data/Assassin_cleaned.csv` | Data | **Đặt sai vị trí** — custom dataset đã làm sạch |
| `src/data/Assassin_preprocessed.csv` | Data | **Đặt sai vị trí** — custom dataset đã tiền xử lý |
| `src/data/W1.zip` | Data | **Đặt sai vị trí** — file nén, không rõ nội dung/mục đích |
| `src/evaluation/metrics.py` | Library | Tính accuracy, macro/weighted F1, spam precision/recall/F1 |
| `src/models/` | — | **Rỗng** — dự kiến chứa model classes tương lai |
| `src/training/` | — | **Rỗng** — mục đích trùng lặp với scripts/ và src/models/ |
| `src/utils/` | — | **Rỗng** — dự kiến shared utilities |
| `tests/test_split_integrity.py` | Tests | 6 tests kiểm tra tính toàn vẹn của split (không overlap, đúng tỷ lệ) |
| `tests/test_preprocess.py` | Tests | 14 tests cho preprocess pipeline |

---

## 2. Vấn đề Hiện tại

### Vấn đề 1 — Data file lẫn vào source code (`src/data/`)
`src/` là thư mục **library code** (importable Python modules). Các file `.csv`, `.zip` là **dữ liệu**, không phải code. Việc đặt chung gây nhầm lẫn về ranh giới data ↔ code, khó gitignore dữ liệu lớn, khó quản lý version data sau này (DVC).

### Vấn đề 2 — Script `augment_data.py` nằm ở root
Root chỉ nên chứa file cấu hình dự án (`README.md`, `CLAUDE.md`, `requirements.txt`, `.gitignore`). Script thực thi thuộc về `scripts/`.

### Vấn đề 3 — `datacollection_guide.md` nằm ở root
Tài liệu thuộc về `docs/`. Root bị ô nhiễm bởi file không phải project config.

### Vấn đề 4 — `configs/augmentation_plan.json` là output, không phải config
File này được sinh ra bởi `augment_data.py` (generated output). Đặt trong `configs/` gây nhầm lẫn với file config tĩnh như `baseline.yaml`. Output của scripts nên ở `experiments/` hoặc `docs/`.

### Vấn đề 5 — `src/training/` mờ nhạt và trùng lặp
`scripts/` đã chứa training logic. `src/training/` rỗng và không rõ sẽ chứa gì khác với `src/models/`. Nếu sau này cần training utilities, nên gộp vào `src/models/` hoặc `src/utils/`.

### Vấn đề 6 — `W1.zip` không có tài liệu
File nén không rõ nguồn gốc, nội dung, liên quan đến dataset nào. Cần được giải nén, đặt tên rõ ràng, hoặc xóa.

---

## 3. Đề xuất Cấu trúc Mới

```
spam-detection/
├── .github/                           # Giữ nguyên
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── configs/                           # Chỉ chứa config tĩnh
│   └── baseline.yaml
│
├── data/                              # TẤT CẢ dữ liệu ra khỏi src/
│   ├── raw/                           # Dataset gốc, không chỉnh sửa
│   │   └── spam.csv
│   └── custom/                        # Dataset tùy chỉnh / augmented
│       ├── Assassin_cleaned.csv
│       ├── Assassin_preprocessed.csv
│       └── W1.zip                     # Hoặc giải nén, đặt tên rõ hơn
│
├── docs/                              # Toàn bộ tài liệu dự án
│   ├── Plan.md
│   ├── QA_QC.md
│   ├── feedback_preprocess.md
│   ├── datacollection_guide.md        # ← chuyển từ root
│   └── augmentation_plan.md           # ← chuyển từ configs/, đổi sang .md (human-readable)
│
├── experiments/                       # Giữ nguyên — output của pipeline
│   ├── baseline/
│   │   ├── model.joblib
│   │   ├── vectorizer.joblib
│   │   ├── metrics.json
│   │   └── error_analysis/
│   │       ├── all_predictions.csv
│   │       ├── prediction_errors.csv
│   │       ├── confusion_matrix.csv
│   │       ├── error_summary.json
│   │       ├── error_analysis_report.md
│   │       └── preprocess_error_comparison_report.md
│   └── splits/
│       └── split_indices.json
│
├── notebooks/                         # Exploration notebooks
│
├── scripts/                           # Tất cả script thực thi
│   ├── create_split.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── analyze_errors.py
│   └── augment_data.py                # ← chuyển từ root
│
├── src/                               # Chỉ chứa library code (importable)
│   ├── data/
│   │   ├── preprocess_fixed.py
│   │   └── validation.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/                        # Future: model classes
│   └── utils/                        # Future: shared utilities
│                                      # src/training/ bị xóa (gộp vào models/ hoặc utils/)
│
├── tests/
│   ├── test_split_integrity.py
│   └── test_preprocess.py
│
├── CLAUDE.md
├── README.md
└── requirements.txt
```

### Tóm tắt thay đổi

| Thay đổi | Từ | Đến | Lý do |
| :--- | :--- | :--- | :--- |
| Script augmentation | `augment_data.py` (root) | `scripts/augment_data.py` | Script thực thi thuộc `scripts/` |
| Tài liệu thu thập data | `datacollection_guide.md` (root) | `docs/datacollection_guide.md` | Tài liệu thuộc `docs/` |
| Dataset gốc | `src/data/raw/` | `data/raw/` | Data không phải code |
| Custom datasets | `src/data/*.csv`, `*.zip` | `data/custom/` | Data không phải code |
| Augmentation plan | `configs/augmentation_plan.json` | `docs/augmentation_plan.md` | Output/tài liệu, không phải config tĩnh |
| Xóa `src/training/` | `src/training/` (rỗng) | — | Trùng lặp với `scripts/` + `src/models/` |
| Config `data.input_path` | `"src/data/raw/spam.csv"` | `"data/raw/spam.csv"` | Cập nhật path trong `configs/baseline.yaml` |

### Nguyên tắc phân chia

```
src/       ← Library code: hàm Python có thể import, không chạy trực tiếp
scripts/   ← Executable scripts: chạy từ CLI để thực hiện một bước pipeline
data/      ← Dữ liệu: CSV, JSON, ZIP — không bao giờ lẫn vào src/
configs/   ← Config tĩnh: YAML/JSON định nghĩa hyperparameters
docs/      ← Tài liệu: kế hoạch, hướng dẫn, báo cáo dạng Markdown
experiments/ ← Output: model, metrics, error analysis — do pipeline sinh ra
tests/     ← Unit/integration tests cho code trong src/
notebooks/ ← Jupyter notebooks: EDA, prototyping
```
